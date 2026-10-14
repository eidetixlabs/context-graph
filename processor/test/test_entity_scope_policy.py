"""Regression tests for globally visible Entity nodes."""

from __future__ import annotations

from unittest.mock import MagicMock

from base.tools.ingest import _build_node_batch, _coerce_node
from processor.dual_write import _build_edge_entries, _upsert_semantic_nodes
from processor.schemas.extraction import DecisionNode, EntityNode, ExtractionResult


def test_gmail_entity_nodes_do_not_receive_source_scope() -> None:
    """Gmail entities stay global even when their source document is scoped."""
    client = MagicMock()
    client.run.return_value = []
    extraction = ExtractionResult(
        nodes=[
            EntityNode(
                id="entity:person:org:alice",
                label="Alice",
                entity_type="person",
                canonical_name="Alice",
            )
        ],
        edges=[],
    )

    _upsert_semantic_nodes(
        client,
        extraction,
        project="org",
        source_scope="scope:gmail:org:message-1",
        preserve_existing_scope=True,
    )

    cypher, params = client.run_write.call_args.args
    entity_row = params["rows"][0]
    assert "REMOVE n.source_scope" in cypher
    assert "source_scope" not in entity_row["props"]


def test_entity_search_ignores_document_scopes() -> None:
    """Global Entity lookup must not require a caller document scope."""
    from base.tools.query import search_entities

    client = MagicMock()
    client.run.return_value = []

    search_entities(client, "alice", allowed_scopes=["scope:gmail:org:message-1"])

    cypher, params = client.run.call_args.args
    assert "source_scope" not in cypher
    assert "allowed_scopes" not in params


def test_memory_search_allows_global_entities_but_scopes_sensitive_nodes() -> None:
    """Mixed full-text search keeps Entity global while filtering other labels."""
    from base.tools.query import search_memory

    client = MagicMock()
    client.run.return_value = []

    search_memory(client, "alice", allowed_scopes=["scope:gmail:org:message-1"])

    cypher, _params = client.run.call_args.args
    assert "n:Entity OR n.source_scope IN $allowed_scopes" in cypher


def test_traversal_allows_entity_seed_but_still_filters_relationship_scope() -> None:
    """Global entity must not bypass scope checks on sensitive relationships."""
    from base.tools.query import traverse

    client = MagicMock()
    client.run.return_value = []

    traverse(
        client,
        "entity:person:org:alice",
        allowed_scopes=["scope:gmail:org:message-1"],
    )

    cypher, _params = client.run.call_args.args
    assert "a:Entity OR a.source_scope IN $allowed_scopes" in cypher
    assert "b:Entity OR b.source_scope IN $allowed_scopes" in cypher
    assert "r.source_scope IN $allowed_scopes" in cypher
    assert "r.source_scope IS NULL" not in cypher


def test_direct_ingest_strips_entity_provenance_fields() -> None:
    """Legacy/direct ingestion cannot persist private provenance on an Entity."""
    raw = {
        "type": "Entity",
        "id": "entity:person:org:alice",
        "label": "Alice",
        "entity_type": "person",
        "canonical_name": "Alice",
        "source_scope": "scope:gmail:org:message-1",
        "source_doc_id": "doc:gmail:org:message-1",
        "source_s3_url": "s3://private/message-1",
        "source_chat": "private-chat",
    }
    props = _coerce_node(raw, project="org")

    assert "source_scope" not in props
    assert "source_doc_id" not in props
    assert "source_s3_url" not in props
    assert "source_chat" not in props

    row = _build_node_batch(
        {raw["id"]: raw},
        project="org",
        ingested_by="Private Employee Name",
    )["Entity"][0]
    assert "ingested_by" not in row["props"]
    assert "create_ingested_by" not in row


def test_document_derived_edges_are_scoped_for_every_provider() -> None:
    """Non-Gmail document writes must not create globally visible provenance."""
    client = MagicMock()
    extraction = ExtractionResult(
        nodes=[DecisionNode(id="decision:private", label="Private", rationale="Sensitive")],
        edges=[],
    )

    entries = _build_edge_entries(
        extraction,
        "doc:notion:private",
        client,
        source_scope="scope:notion:private",
    )

    assert entries
    assert all(entry["properties"]["source_scope"] == "scope:notion:private" for entry in entries)


def test_gmail_sensitive_node_ids_are_namespaced_per_document() -> None:
    """Same semantic slug in two Gmail messages must not share one scoped node."""
    from processor.dual_write import _namespace_gmail_semantic_ids

    extraction_a = ExtractionResult(
        nodes=[DecisionNode(id="decision:approve", label="Approve", rationale="A")],
        edges=[],
    )
    extraction_b = ExtractionResult(
        nodes=[DecisionNode(id="decision:approve", label="Approve", rationale="B")],
        edges=[],
    )

    _namespace_gmail_semantic_ids(extraction_a, "doc:gmail:org:message-a")
    _namespace_gmail_semantic_ids(extraction_b, "doc:gmail:org:message-b")

    id_a = extraction_a.nodes[0].id
    id_b = extraction_b.nodes[0].id
    assert id_a != id_b
    assert id_a.startswith("decision:approve:gmail:")
    assert id_b.startswith("decision:approve:gmail:")
