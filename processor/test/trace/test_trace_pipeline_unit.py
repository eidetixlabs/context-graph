from __future__ import annotations

from types import SimpleNamespace

import pytest

from processor.resolution.models import ResolutionMap, ResolutionMapEntry, ResolvedEntity
from processor.schemas.extraction import DecisionNode
from processor.trace import pipeline
from processor.trace.transformer import trace_to_node


def test_trace_mentions_to_candidates_uses_payload_confidence() -> None:
    trace = {
        "mentions": [
            {
                "mention_text": "Acme Control Plane",
                "mention_type": "service",
                "confidence": 0.42,
            },
            {
                "mention_text": "Alice",
                "mention_type": "person",
            },
        ]
    }

    candidates = pipeline._trace_mentions_to_candidates(trace)

    assert len(candidates) == 2
    assert candidates[0].confidence == pytest.approx(0.42)
    assert candidates[1].confidence == pytest.approx(pipeline._TRACE_DEFAULT_MENTION_CONFIDENCE)


def test_build_author_mention_candidate_uses_trusted_author_fields() -> None:
    trace = {
        "_author_employee_id": "emp:test",
        "_author_name": "Alice Smith",
        "mentions": [
            {"mention_text": "Other Person", "mention_type": "person"},
        ],
    }

    candidate = pipeline._build_author_mention_candidate(trace)

    assert candidate is not None
    assert candidate.mention_text == "Alice Smith"
    assert candidate.mention_type == "person"
    assert candidate.confidence == pytest.approx(1.0)
    assert candidate.evidence == {"author_employee_id": "emp:test"}


def test_trace_to_node_sets_personal_source_scope() -> None:
    trace = {
        "trace_id": "abc-123",
        "_author_employee_id": "emp:test",
        "node": {
            "type": "Insight",
            "label": "Label",
            "body": "Body",
        },
    }

    node = trace_to_node(trace)

    assert getattr(node, "source_scope", None) == "scope:trace:emp:test"

@pytest.mark.asyncio
async def test_llm_trace_filter_fail_open_on_agent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = [
        pipeline.MentionCandidate(
            mention_text="Acme Control Plane",
            mention_type="service",
            chunk_index=0,
            confidence=0.97,
        )
    ]

    monkeypatch.setattr(pipeline, "create_agent", lambda *args, **kwargs: object())

    async def fake_run_agent(*args, **kwargs):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(pipeline, "run_agent", fake_run_agent)

    accepted, rejected = await pipeline._llm_trace_filter(candidates, "trace-err")

    assert accepted == candidates
    assert rejected == {"llm_filter_error_passthrough": 1}


class _AcquireContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePool:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


@pytest.mark.asyncio
async def test_run_trace_pipeline_persists_before_graph_write(monkeypatch: pytest.MonkeyPatch) -> None:
    call_order: list[str] = []

    fake_conn = object()
    monkeypatch.setattr(pipeline.db_pool, "get", lambda: _FakePool(fake_conn))

    async def fake_resolve_trace_mentions(*args, **kwargs) -> ResolutionMap:
        # Mirrors what the real resolve_trace_mentions + build_resolved_entities produces:
        # provisional IDs already assigned, author entry included.
        return ResolutionMap(
            doc_id="",
            org_id="org:test",
            entries={
                "Acme Service": ResolutionMapEntry(
                    mention_text="Acme Service",
                    entity_id="entity:service:org-test:acme-service",
                    resolution_method="new",
                    confidence=0.95,
                ),
                "Test Author": ResolutionMapEntry(
                    mention_text="Test Author",
                    entity_id="entity:person:org-test:test-author",
                    resolution_method="new",
                    confidence=1.0,
                ),
            },
            resolved_entities=[
                ResolvedEntity(
                    entity_id="entity:service:org-test:acme-service",
                    canonical_name="Acme Service",
                    entity_type="service",
                    is_new=True,
                    confidence=0.95,
                ),
                ResolvedEntity(
                    entity_id="entity:person:org-test:test-author",
                    canonical_name="Test Author",
                    entity_type="person",
                    is_new=True,
                    confidence=1.0,
                ),
            ],
        )

    async def fake_discover_edges(node, trace):
        return []

    async def fake_llm_trace_filter(candidates, trace_id: str):
        return candidates, {}

    async def fake_persist_trace_new_entities_from_map(trace_id, org_id, resolution_map, conn) -> None:
        call_order.append("persist_new_entities")
        for entity in resolution_map.resolved_entities:
            entity.is_new = False

    async def fake_to_thread(fn, *args, **kwargs):
        call_order.append("write_graph")
        return {"nodes_upserted": 1, "rels_upserted": 1}

    monkeypatch.setattr(pipeline, "resolve_trace_mentions", fake_resolve_trace_mentions)
    monkeypatch.setattr(pipeline, "trace_to_node", lambda trace: DecisionNode(id="decision:test", label="Test", rationale="r"))
    monkeypatch.setattr(pipeline, "discover_edges", fake_discover_edges)
    monkeypatch.setattr(pipeline, "_llm_trace_filter", fake_llm_trace_filter)
    monkeypatch.setattr(pipeline, "persist_trace_new_entities_from_map", fake_persist_trace_new_entities_from_map)
    monkeypatch.setattr(pipeline.asyncio, "to_thread", fake_to_thread)

    result = await pipeline.run_trace_pipeline(
        trace={
            "trace_id": "trace-123",
            "org_id": "org:test",
            "project": "test",
            "_author_name": "Test Author",
            "mentions": [
                {"mention_text": "Acme Service", "mention_type": "service", "confidence": 0.95},
            ],
        },
        neo4j_client=SimpleNamespace(),
    )

    assert any(getattr(node, "id", None) == "entity:service:org-test:acme-service" for node in result.nodes)
    assert any(edge.rel == "AUTHORED_BY" for edge in result.edges)
    assert call_order == ["persist_new_entities", "write_graph"]
