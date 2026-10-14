"""Unit tests for processor.dual_write edge construction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from processor.dual_write import (
    IndexJob,
    IndexJobAlreadyClaimed,
    IndexingIncompleteError,
    StaleIndexWork,
    _build_index_payload,
    _build_edge_entries,
    _coerce_temporal_fields,
    _complete_index_sinks,
    _index_job_lease,
    _reconcile_graph,
    _write_graph_sink,
    _write_vector_sink,
    _upsert_doc_node,
    _upsert_semantic_nodes,
)
from processor.resolution.models import ResolutionMap, ResolvedEntity
from processor.schemas.extraction import (
    DecisionNode,
    EntityNode,
    ExtractionResult,
)
from processor.state_machine import DocumentRecord


DOC_ID = "doc:test-auth-design"


def _extracted_from_sources(entries: list[dict]) -> list[str]:
    return [
        e["from"]["id"]
        for e in entries
        if e["rel"] == "EXTRACTED_FROM"
    ]


def test_build_edge_entries_ignores_resolution_map_entities_absent_from_nodes() -> None:
    """Post-extraction writes use extraction_result.nodes as the final node list."""
    extraction_result = ExtractionResult(
        nodes=[
            DecisionNode(
                id="decision:use-postgres",
                label="Use Postgres",
                scope="system",
                trigger="session storage choice",
                rationale="ops familiarity",
            ),
        ],
        edges=[],
    )
    resolution_map = ResolutionMap(
        doc_id=DOC_ID,
        org_id="org:acme",
        resolved_entities=[
            ResolvedEntity(
                entity_id="entity:person:jane-doe",
                canonical_name="Jane Doe",
                entity_type="person",
                is_new=True,
            ),
        ],
    )
    client = MagicMock()

    entries = _build_edge_entries(extraction_result, DOC_ID, client, resolution_map)

    assert _extracted_from_sources(entries) == ["decision:use-postgres"]


def test_build_edge_entries_does_not_duplicate_extracted_from_for_entity_nodes() -> None:
    """Entity nodes already in extraction_result must not get a second EXTRACTED_FROM."""
    extraction_result = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:jane-doe",
                label="Jane Doe",
                entity_type="person",
                canonical_name="Jane Doe",
            ),
        ],
        edges=[],
    )
    resolution_map = ResolutionMap(
        doc_id=DOC_ID,
        org_id="org:acme",
        resolved_entities=[
            ResolvedEntity(
                entity_id="entity:person:jane-doe",
                canonical_name="Jane Doe",
                entity_type="person",
            ),
        ],
    )
    client = MagicMock()

    entries = _build_edge_entries(extraction_result, DOC_ID, client, resolution_map)

    assert _extracted_from_sources(entries) == ["entity:person:jane-doe"]


def test_coerce_temporal_fields_accepts_z_suffix() -> None:
    from neo4j.time import DateTime as Neo4jDateTime

    raw = {
        "id": "fact:example",
        "type": "Fact",
        "valid_from": "2026-04-01T00:00:00Z",
    }
    result = _coerce_temporal_fields(raw)
    assert isinstance(result["valid_from"], Neo4jDateTime)
    assert result["valid_from"].iso_format() == "2026-04-01T00:00:00.000000000+00:00"


def test_document_parent_relationship_carries_source_scope() -> None:
    client = MagicMock()
    doc = DocumentRecord(
        id="doc:gmail:org:message",
        org_id="org:acme",
        source_provider="gmail",
        source_scope="scope:gmail:org:message",
        source_ref="message",
        title="Subject",
        s3_key="raw/message.json",
        processing_state="extracted",
        signal_score=None,
        char_count=None,
        current_version=1,
        parent_ref="thread",
    )

    _upsert_doc_node(client, doc, "default")

    cypher, params = client.run_write.call_args.args
    assert "SET r.source_scope = $source_scope" in cypher
    assert params["source_scope"] == doc.source_scope


def test_profile_entries_carry_document_version_and_link_canonical_repository() -> None:
    extraction_result = ExtractionResult(
        nodes=[
            DecisionNode(
                id="decision:use-postgres",
                label="Use Postgres",
                scope="system",
                trigger="repository architecture",
                rationale="durable storage",
            ),
        ],
        edges=[],
    )
    client = MagicMock()

    entries = _build_edge_entries(
        extraction_result,
        "doc:github-profile:acme/service/_profile/main",
        client,
        source_scope="scope:github:acme/service",
        source_doc_version=3,
    )

    extracted = next(e for e in entries if e["rel"] == "EXTRACTED_FROM")
    assert extracted["properties"]["source_doc_id"] == (
        "doc:github-profile:acme/service/_profile/main"
    )
    assert extracted["properties"]["source_doc_version"] == 3


def test_index_payload_retains_embeddings_and_nodes_get_version_provenance() -> None:
    decision = DecisionNode(
        id="decision:storage",
        label="Storage",
        rationale="Durability",
        embedding=[0.1, 0.2],
    )
    extraction_result = ExtractionResult(nodes=[decision])

    payload = _build_index_payload("acme", [], extraction_result)

    assert payload["extraction_result"]["nodes"][0]["embedding"] == [0.1, 0.2]

    client = MagicMock()
    _upsert_semantic_nodes(
        client,
        extraction_result,
        "acme",
        source_scope="scope:github:acme/service",
        source_doc_id="doc:github-profile:acme/service/_profile/main",
        source_doc_version=4,
    )
    _, params = client.run_write.call_args.args
    props = params["rows"][0]["props"]
    assert props["source_doc_id"] == (
        "doc:github-profile:acme/service/_profile/main"
    )
    assert props["source_doc_version"] == 4


@pytest.mark.asyncio
async def test_vector_success_graph_failure_retries_only_graph_and_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"vector": 0, "graph": 0}
    states = {"vector": "pending", "graph": "pending"}
    job = IndexJob(
        doc_id="doc:github-profile:acme/service/_profile/main",
        document_version=2,
        payload_hash="payload-hash",
        payload_s3_key="index-payload.json",
        vector_state="pending",
        graph_state="pending",
    )
    doc = DocumentRecord(
        id=job.doc_id,
        org_id="org:acme",
        source_provider="github-profile",
        source_scope="scope:github:acme/service",
        source_ref="acme/service/_profile/main",
        title="Profile",
        s3_key="raw-profile.json",
        processing_state="indexing",
        signal_score=1.0,
        char_count=100,
        current_version=2,
        parent_ref="acme/service",
    )
    payload = {
        "schema_version": "document_index_payload.v1",
        "project": "acme",
        "chunks": [],
        "extraction_result": {"nodes": [], "edges": [], "extraction_notes": ""},
    }

    async def fake_vector(conn_arg, job_arg, payload_arg):
        calls["vector"] += 1
        states["vector"] = "completed"
        job_arg.vector_state = "completed"

    async def fake_graph(conn_arg, job_arg, doc_arg, payload_arg, client_arg):
        calls["graph"] += 1
        if calls["graph"] == 1:
            raise RuntimeError("injected graph failure")
        return {"nodes_upserted": 1, "rels_upserted": 2}

    async def fake_state(conn_arg, job_arg, sink, state, error=None):
        states[sink] = state
        object.__setattr__(job_arg, f"{sink}_state", state)

    async def fake_increment(conn_arg, job_arg):
        return None

    monkeypatch.setattr("processor.dual_write._write_vector_sink", fake_vector)
    monkeypatch.setattr("processor.dual_write._write_graph_sink", fake_graph)
    monkeypatch.setattr("processor.dual_write._set_sink_state", fake_state)
    monkeypatch.setattr("processor.dual_write._increment_attempt", fake_increment)

    with pytest.raises(IndexingIncompleteError):
        await _complete_index_sinks(MagicMock(), job, doc, payload, MagicMock())

    assert calls == {"vector": 1, "graph": 1}
    assert states == {"vector": "completed", "graph": "failed"}

    summary = await _complete_index_sinks(
        MagicMock(), job, doc, payload, MagicMock()
    )

    assert calls == {"vector": 1, "graph": 2}
    assert states == {"vector": "completed", "graph": "completed"}
    assert summary["vector_state"] == "completed"
    assert summary["graph_state"] == "completed"


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SinkConnection:
    def __init__(self, document_row=None, lease=True) -> None:
        self.document_row = document_row
        self.lease = lease
        self.executions: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self):
        return _Tx()

    async def fetchrow(self, query, *args):
        return self.document_row

    async def fetchval(self, query, *args):
        if "pg_try_advisory_lock" in query:
            return self.lease
        return True

    async def execute(self, query, *args):
        self.executions.append((query, args))
        return "UPDATE 1"

    async def executemany(self, query, args):
        self.executions.append((query, tuple(args)))


class _SinkAcquire:
    def __init__(self, connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SinkPool:
    def __init__(self, connection) -> None:
        self.connection = connection

    def acquire(self):
        return _SinkAcquire(self.connection)


class _BoundedPool:
    """A one-connection pool that fails instead of waiting on nested acquisition."""

    def __init__(self, connection) -> None:
        self.connection = connection
        self.in_use = False
        self.acquire_count = 0

    def acquire(self):
        pool = self

        class _BoundedAcquire:
            async def __aenter__(self):
                if pool.in_use:
                    raise AssertionError("nested pool acquisition would deadlock")
                pool.in_use = True
                pool.acquire_count += 1
                return pool.connection

            async def __aexit__(self, exc_type, exc, tb):
                pool.in_use = False
                return False

        return _BoundedAcquire()


@pytest.mark.asyncio
async def test_v1_vector_sink_is_cancelled_after_v2_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = IndexJob("doc:1", 1, "hash", "payload.json", "pending", "pending")
    connection = _SinkConnection(
        {"current_version": 2, "processing_state": "raw"}
    )
    monkeypatch.setattr(
        "processor.dual_write.db_pool.get", lambda: _SinkPool(connection)
    )

    with pytest.raises(StaleIndexWork):
        await _write_vector_sink(connection, job, {"chunks": []})

    assert not any("document_chunks" in query for query, _ in connection.executions)


@pytest.mark.asyncio
async def test_duplicate_index_claim_is_rejected_by_session_advisory_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _SinkConnection(lease=False)
    monkeypatch.setattr(
        "processor.dual_write.db_pool.get", lambda: _SinkPool(connection)
    )
    job = IndexJob("doc:1", 3, "hash", "payload.json", "pending", "pending")

    with pytest.raises(IndexJobAlreadyClaimed):
        async with _index_job_lease(job):
            pass


@pytest.mark.asyncio
async def test_index_lease_reuses_only_connection_for_all_postgres_sink_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = IndexJob("doc:1", 2, "hash", "payload.json", "pending", "pending")
    doc = DocumentRecord(
        id=job.doc_id,
        org_id="org:acme",
        source_provider="jira",
        source_scope="scope:jira:1",
        source_ref="JIRA-1",
        title="Issue",
        s3_key="raw.json",
        processing_state="indexing",
        signal_score=1.0,
        char_count=100,
        current_version=2,
    )
    payload = {
        "project": "acme",
        "chunks": [],
        "extraction_result": {"nodes": [], "edges": [], "extraction_notes": ""},
    }
    connection = _SinkConnection(
        {"current_version": 2, "processing_state": "indexing"}
    )
    pool = _BoundedPool(connection)
    monkeypatch.setattr("processor.dual_write.db_pool.get", lambda: pool)

    async with _index_job_lease(job) as leased_connection:
        await _complete_index_sinks(
            leased_connection,
            job,
            doc,
            payload,
            MagicMock(),
        )

    assert pool.acquire_count == 1


def _reconciliation_doc(provider: str = "jira") -> DocumentRecord:
    return DocumentRecord(
        id=f"doc:{provider}:1",
        org_id="org:acme",
        source_provider=provider,
        source_scope=f"scope:{provider}:1",
        source_ref="1",
        title="Document",
        s3_key="raw.json",
        processing_state="indexing",
        signal_score=1.0,
        char_count=100,
        current_version=2,
    )


def test_reconciliation_removes_only_stale_relationships_owned_by_document() -> None:
    doc = _reconciliation_doc()
    client = MagicMock()

    _reconcile_graph(client, doc, 2)

    relationship_query, params = client.run_write.call_args_list[0].args
    assert "r.source_doc_id = $doc_id" in relationship_query
    assert "coalesce(r.source_doc_version, -1) <> $document_version" in relationship_query
    assert "DELETE r" in relationship_query
    assert params == {"doc_id": doc.id, "document_version": 2}


def test_reconciliation_deletes_only_orphaned_stale_extracted_semantic_nodes() -> None:
    doc = _reconciliation_doc()
    client = MagicMock()

    _reconcile_graph(client, doc, 2)

    node_query = client.run_write.call_args_list[1].args[0]
    assert "n.source_doc_id = $doc_id" in node_query
    assert "coalesce(n.source_doc_version, -1) < $document_version" in node_query
    assert "(n:Decision OR n:Constraint OR n:Insight OR n:Signal)" in node_query
    assert "NOT (n)--()" in node_query
    assert "DELETE n" in node_query
    assert "DETACH DELETE" not in node_query


def test_reconciliation_preserves_cross_document_nodes_and_shared_directory_nodes() -> None:
    doc = _reconciliation_doc()
    client = MagicMock()

    _reconcile_graph(client, doc, 2)

    relationship_query = client.run_write.call_args_list[0].args[0]
    node_query = client.run_write.call_args_list[1].args[0]
    # Another document's relationships are not selected, so they keep a stale
    # semantic node non-orphaned and visible.
    assert "r.source_doc_id = $doc_id" in relationship_query
    assert "NOT (n)--()" in node_query
    # The positive allowlist can never select shared directory nodes.
    assert "n:Entity" not in node_query
    assert "n:Person" not in node_query
    assert "n:Topic" not in node_query


@pytest.mark.parametrize("provider", ["github-profile", "gmail", "jira"])
def test_reconciliation_retry_is_idempotent_and_preserves_current_nodes(
    provider: str,
) -> None:
    doc = _reconciliation_doc(provider)
    client = MagicMock()

    _reconcile_graph(client, doc, 2)
    _reconcile_graph(client, doc, 2)

    assert client.run_write.call_count == 4
    node_query = client.run_write.call_args_list[1].args[0]
    assert "coalesce(n.source_doc_version, -1) < $document_version" in node_query
    assert client.run_write.call_args_list[0].args == client.run_write.call_args_list[2].args
    assert client.run_write.call_args_list[1].args == client.run_write.call_args_list[3].args


@pytest.mark.asyncio
async def test_graph_failure_before_reconciliation_preserves_previous_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = IndexJob("doc:1", 2, "hash", "payload.json", "completed", "pending")
    doc = DocumentRecord(
        id=job.doc_id,
        org_id="org:acme",
        source_provider="jira",
        source_scope="scope:jira:1",
        source_ref="JIRA-1",
        title="Issue",
        s3_key="raw.json",
        processing_state="indexing",
        signal_score=1.0,
        char_count=100,
        current_version=2,
    )
    payload = {
        "project": "acme",
        "extraction_result": {"nodes": [], "edges": [], "extraction_notes": ""},
    }
    calls: list[str] = []
    client = MagicMock()
    connection = _SinkConnection(
        {"current_version": 2, "processing_state": "indexing"}
    )
    monkeypatch.setattr(
        "processor.dual_write.db_pool.get", lambda: _SinkPool(connection)
    )

    monkeypatch.setattr(
        "processor.dual_write._upsert_doc_node",
        lambda *args: calls.append("document"),
    )

    def fail_nodes(*args, **kwargs):
        calls.append("replacement")
        raise RuntimeError("injected replacement failure")

    monkeypatch.setattr("processor.dual_write._upsert_semantic_nodes", fail_nodes)
    monkeypatch.setattr(
        "processor.dual_write._reconcile_graph",
        lambda *args: calls.append("reconcile"),
    )

    with pytest.raises(RuntimeError, match="replacement failure"):
        await _write_graph_sink(connection, job, doc, payload, client)

    assert calls == ["document", "replacement"]
