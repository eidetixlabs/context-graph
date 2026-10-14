"""Unit tests for extraction-result-driven entity persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from processor.schemas.extraction import EntityNode, ExtractionResult, SemanticEdge


class _AsyncTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _RecordingConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.alias_lookup_entity_id: str | None = None
        self.fuzzy_rows: list[dict[str, object]] = []

    def transaction(self) -> _AsyncTransaction:
        return _AsyncTransaction()

    async def execute(self, query: str, *args: object) -> None:
        self.calls.append((query, args))

    async def fetchrow(self, query: str, *args: object):
        if "FROM entity_aliases ea" in query and self.alias_lookup_entity_id:
            return {"entity_id": self.alias_lookup_entity_id}
        return None

    async def fetch(self, query: str, *args: object):
        if "array_agg(ea.alias_normalized)" in query:
            return self.fuzzy_rows
        return []


@pytest.mark.asyncio
async def test_persists_entity_nodes_without_mentions_edges() -> None:
    from processor.resolution import persistence

    doc = SimpleNamespace(id="doc:test", org_id="org:nodeops")
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:org-nodeops:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases='["Naman", "Naman K."]',
            ),
        ],
        edges=[],
    )
    conn = _RecordingConn()

    await persistence.persist_entities_from_extraction(
        doc=doc,
        extraction_result=extraction,
        conn=conn,
    )

    registry_calls = [
        args for query, args in conn.calls
        if "INSERT INTO entity_registry" in query
    ]
    alias_calls = [
        args for query, args in conn.calls
        if "INSERT INTO entity_aliases" in query
    ]

    assert registry_calls == [
        (
            "entity:person:nodeops:naman-k",
            "org:nodeops",
            "person",
            "Naman K",
        )
    ]
    assert {(args[0], args[3]) for args in alias_calls} == {
        ("naman k", "Naman K"),
        ("naman", "Naman"),
    }


@pytest.mark.asyncio
async def test_reconciles_with_resolution_map_before_insert() -> None:
    from processor.resolution import persistence
    from processor.resolution.models import ResolutionMap, ResolutionMapEntry

    doc = SimpleNamespace(id="doc:test", org_id="org:nodeops")
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases='["Naman"]',
            ),
        ],
        edges=[],
    )
    resolution_map = ResolutionMap(
        doc_id="doc:test",
        org_id="org:nodeops",
        entries={
            "Naman K": ResolutionMapEntry(
                mention_text="Naman K",
                entity_id="entity:person:nodeops:naman-k",
                resolution_method="llm_disambiguation",
                confidence=0.88,
            ),
        },
    )
    conn = _RecordingConn()

    await persistence.persist_entities_from_extraction(
        doc=doc,
        extraction_result=extraction,
        conn=conn,
        resolution_map=resolution_map,
    )

    registry_calls = [args for query, args in conn.calls if "INSERT INTO entity_registry" in query]
    assert registry_calls[0][0] == "entity:person:nodeops:naman-k"


@pytest.mark.asyncio
async def test_reconciles_with_alias_lookup_before_creating_new() -> None:
    from processor.resolution import persistence

    doc = SimpleNamespace(id="doc:test", org_id="org:nodeops")
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases='["Naman"]',
            ),
        ],
        edges=[],
    )
    conn = _RecordingConn()
    conn.alias_lookup_entity_id = "entity:person:nodeops:naman-k"

    await persistence.persist_entities_from_extraction(
        doc=doc,
        extraction_result=extraction,
        conn=conn,
    )

    registry_calls = [args for query, args in conn.calls if "INSERT INTO entity_registry" in query]
    assert registry_calls[0][0] == "entity:person:nodeops:naman-k"


@pytest.mark.asyncio
async def test_generates_canonical_entity_id_when_reconcile_misses() -> None:
    from processor.resolution import persistence

    doc = SimpleNamespace(id="doc:test", org_id="org:nodeops")
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases="[]",
            ),
        ],
        edges=[],
    )
    conn = _RecordingConn()

    await persistence.persist_entities_from_extraction(
        doc=doc,
        extraction_result=extraction,
        conn=conn,
    )

    registry_calls = [args for query, args in conn.calls if "INSERT INTO entity_registry" in query]
    assert registry_calls[0][0] == "entity:person:nodeops:naman-k"


@pytest.mark.asyncio
async def test_rewrites_extraction_result_entity_ids_to_canonical_ids() -> None:
    from processor.resolution import persistence

    doc = SimpleNamespace(id="doc:test", org_id="org:nodeops")
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:naman-k",
                label="Naman K",
                entity_type="person",
                canonical_name="Naman K",
                aliases="[]",
            ),
        ],
        edges=[
            SemanticEdge(
                from_id="entity:person:naman-k",
                rel="MENTIONS",
                to_id="topic:launch-plan",
                edge_context="Mentioned in launch planning context.",
            ),
        ],
    )
    conn = _RecordingConn()

    await persistence.persist_entities_from_extraction(
        doc=doc,
        extraction_result=extraction,
        conn=conn,
    )

    canonical_id = "entity:person:nodeops:naman-k"
    assert extraction.nodes[0].id == canonical_id
    assert extraction.edges[0].from_id == canonical_id
