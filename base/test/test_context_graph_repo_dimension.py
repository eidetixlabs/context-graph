from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from base.tools import query as q
from processor.dual_write import _upsert_doc_node, _upsert_semantic_nodes
from processor.schemas.extraction import ExtractionResult
from processor.state_machine import DocumentRecord


ROOT = Path(__file__).resolve().parents[2]


def test_repo_dimension_migration_is_additive_and_packaged_in_both_charts() -> None:
    relative = "022_context_graph_documents.sql"
    copies = [
        ROOT / "base" / "db" / "init" / relative,
    ]

    assert all(path.exists() for path in copies)
    sql = copies[0].read_text()
    assert "ADD COLUMN IF NOT EXISTS repo TEXT" in sql
    assert "ADD COLUMN IF NOT EXISTS doc_type TEXT" in sql
    assert "idx_docs_org_repo" in sql
    assert "idx_docs_org_repo_type" in sql
    assert "'portal'" in sql
    assert "DROP CONSTRAINT IF EXISTS documents_doc_type_check" in sql
    assert "ADD CONSTRAINT documents_doc_type_check" in sql
    assert "doc_type IS NULL" in sql
    for doc_type in (
        "architecture",
        "hld",
        "lld",
        "design",
        "roadmap",
        "ticket_retrospective",
        "context",
    ):
        assert f"'{doc_type}'" in sql
    assert "INSERT INTO schema_migrations (id) VALUES ('022_context_graph_documents')" in sql


def test_portal_uses_generic_document_processor_worker() -> None:
    source = (ROOT / "processor" / "worker.py").read_text()

    assert "def _worker_for_provider(provider: str)" in source
    assert "return _GenericProcessorWorker(provider=provider)" in source
    assert "worker = _worker_for_provider(args.provider)" in source


def test_portal_processor_admits_uploads_without_generic_scoring() -> None:
    source = (ROOT / "processor" / "worker.py").read_text()
    assert 'if self.PROVIDER == "portal"' in source
    assert "signal_score=1.0" in source
    assert "Portal-uploaded context is deterministically admitted" in source


@pytest.mark.asyncio
async def test_pgvector_search_includes_untagged_documents_for_repo_scope() -> None:
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[])

    await q._pgvector_chunk_search(
        pool,
        [0.1, 0.2],
        "org:acme",
        10,
        repo="acme/widgets",
    )

    sql, *params = pool.fetch.await_args.args
    assert "d.repo IS NULL OR d.repo = $4" in sql
    assert params[3] == "acme/widgets"


def test_neo4j_vector_and_fulltext_search_apply_repo_filter() -> None:
    client = MagicMock()
    client.run.return_value = []

    q.query_context_vector(client, [0.1], repo="acme/widgets")
    vector_sql, vector_params = client.run.call_args.args
    assert "node.repo IS NULL OR node.repo = $repo" in vector_sql
    assert vector_params["repo"] == "acme/widgets"

    q.search_memory(client, "architecture", repo="acme/widgets")
    text_sql, text_params = client.run.call_args.args
    assert "n.repo IS NULL OR n.repo = $repo" in text_sql
    assert text_params["repo"] == "acme/widgets"


def test_document_and_semantic_nodes_preserve_repo_and_doc_type() -> None:
    doc = DocumentRecord(
        id="doc:portal:org:acme:architecture-v1",
        org_id="org:acme",
        source_provider="portal",
        source_scope="scope:github:acme/widgets",
        source_ref="architecture-v1",
        title="Architecture",
        s3_key="org:acme/portal/raw/architecture-v1/v1.json",
        processing_state="raw",
        signal_score=None,
        char_count=100,
        current_version=1,
        repo="acme/widgets",
        doc_type="architecture",
    )
    client = MagicMock()

    _upsert_doc_node(client, doc, "acme")
    _, doc_params = client.run_write.call_args.args
    assert doc_params["repo"] == "acme/widgets"
    assert doc_params["doc_type"] == "architecture"

    extraction = ExtractionResult(
        nodes=[
            {
                "id": "insight:gateway-owns-auth",
                "type": "Insight",
                "label": "Gateway owns auth",
                "body": "The gateway owns authentication.",
            }
        ]
    )
    _upsert_semantic_nodes(
        client,
        extraction,
        "acme",
        source_doc_id=doc.id,
        source_doc_version=1,
        repo=doc.repo,
        doc_type=doc.doc_type,
    )
    _, semantic_params = client.run_write.call_args.args
    props = semantic_params["rows"][0]["props"]
    assert props["repo"] == "acme/widgets"
    assert props["doc_type"] == "architecture"
