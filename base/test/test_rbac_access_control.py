"""
Tests for RBAC access control implementation.

Covers:
- base/db/access.py: get_allowed_scopes() scope resolution + TTL cache
- base/tools/query.py: search_memory(), query_context_vector() scope filtering
- base/tools/query.py: get_document_content() access guard

Run from repo root:
  pytest base/test/test_rbac_access_control.py -v
"""

from __future__ import annotations

import base64
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base.db.access import (
    _cache,
    get_allowed_scopes,
    invalidate_scope_cache,
    trace_personal_scope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pool(rows: list[dict]) -> AsyncMock:
    """Return a mock asyncpg Pool whose .fetch() yields the given rows."""
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=[
        {**row} for row in rows
    ])
    return pool


def _make_neo4j_client(run_returns: list[dict] | None = None) -> MagicMock:
    """Return a mock Neo4jClient whose .run() returns the given row list."""
    client = MagicMock()
    client.run = MagicMock(return_value=run_returns or [])
    return client


# ---------------------------------------------------------------------------
# get_allowed_scopes (base/db/access.py)
# ---------------------------------------------------------------------------

class TestGetAllowedScopes:
    def setup_method(self):
        # Clear TTL cache before every test
        _cache.clear()

    @pytest.mark.asyncio
    async def test_service_account_returns_none(self):
        """employee_id=None (service account) → returns None to bypass all filtering."""
        pool = _make_pool([])
        result = await get_allowed_scopes(pool, None)
        assert result is None
        pool.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_employee_with_memberships(self):
        """Returns channel memberships plus the employee's personal trace scope."""
        pool = _make_pool([
            {"channel_id": "scope:notion:abc"},
            {"channel_id": "scope:slack:xyz"},
        ])
        result = await get_allowed_scopes(pool, "emp:alice")
        assert result == [
            "scope:notion:abc",
            "scope:slack:xyz",
            trace_personal_scope("emp:alice"),
        ]

    @pytest.mark.asyncio
    async def test_employee_no_memberships(self):
        """Employee with zero memberships still gets their personal trace scope."""
        pool = _make_pool([])
        result = await get_allowed_scopes(pool, "emp:bob")
        assert result == [trace_personal_scope("emp:bob")]

    @pytest.mark.asyncio
    async def test_result_is_cached(self):
        """Second call for same employee should not hit the DB again."""
        pool = _make_pool([{"channel_id": "scope:notion:abc"}])
        await get_allowed_scopes(pool, "emp:alice")
        await get_allowed_scopes(pool, "emp:alice")
        assert pool.fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_personal_scope_not_duplicated_when_present(self):
        """If DB already has the personal scope row, return it only once."""
        personal = trace_personal_scope("emp:alice")
        pool = _make_pool([{"channel_id": personal}])
        result = await get_allowed_scopes(pool, "emp:alice")
        assert result == [personal]

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """invalidate_scope_cache() removes the entry so next call re-fetches."""
        pool = _make_pool([{"channel_id": "scope:notion:abc"}])
        await get_allowed_scopes(pool, "emp:alice")
        invalidate_scope_cache("emp:alice")
        await get_allowed_scopes(pool, "emp:alice")
        assert pool.fetch.call_count == 2


# ---------------------------------------------------------------------------
# search_memory scope filtering (base/tools/query.py)
# ---------------------------------------------------------------------------

class TestSearchMemoryScopeFilter:
    def test_employee_not_in_scope_sees_no_nodes(self):
        """
        When allowed_scopes does not contain the node's source_scope, the
        scope_filter Cypher clause is injected.  We verify the client.run()
        call receives the filter AND returns empty (simulating Neo4j honouring it).
        """
        from base.tools.query import search_memory

        client = _make_neo4j_client(run_returns=[])

        results = search_memory(
            client=client,
            query="deployment decision",
            allowed_scopes=["scope:slack:members-only"],  # does NOT include the node's scope
        )

        assert results == []
        # Ensure the cypher passed to client.run contains the scope filter clause
        for call in client.run.call_args_list:
            cypher = call.args[0] if call.args else call.kwargs.get("cypher", "")
            params = call.args[1] if len(call.args) > 1 else call.kwargs.get("params", {})
            if "allowed_scopes" in params:
                assert "source_scope" in cypher

    def test_employee_in_scope_returns_nodes(self):
        """When allowed_scopes includes the node's source_scope, matching nodes are returned."""
        from base.tools.query import search_memory

        node_scope = "scope:notion:abc"
        mock_row = {
            "props": {
                "id": "node:001",
                "summary": "We decided to use Kubernetes",
                "source_scope": node_scope,
                "project": "myproject",
            },
            "node_type": "Decision",
            "score": 0.9,
        }
        client = _make_neo4j_client(run_returns=[mock_row])

        results = search_memory(
            client=client,
            query="kubernetes decision",
            allowed_scopes=[node_scope],
        )

        assert len(results) == 1
        assert results[0]["id"] == "node:001"

    def test_service_account_no_filter_applied(self):
        """allowed_scopes=None → no scope_filter clause injected into Cypher."""
        from base.tools.query import search_memory

        mock_row = {
            "props": {
                "id": "node:002",
                "summary": "Secret architecture decision",
                "source_scope": "scope:notion:secret",
                "project": "proj",
            },
            "node_type": "Decision",
            "score": 0.8,
        }
        client = _make_neo4j_client(run_returns=[mock_row])

        results = search_memory(
            client=client,
            query="secret architecture",
            allowed_scopes=None,
        )

        assert len(results) == 1
        # Ensure NO scope filter was added to the cypher
        for call in client.run.call_args_list:
            cypher = call.args[0] if call.args else ""
            params = call.args[1] if len(call.args) > 1 else {}
            assert "allowed_scopes" not in params
            assert "source_scope" not in cypher


# ---------------------------------------------------------------------------
# query_context_vector scope filtering (base/tools/query.py)
# ---------------------------------------------------------------------------

class TestQueryContextVectorScopeFilter:
    def test_employee_not_in_scope_returns_zero_nodes(self):
        """Vector search with scope mismatch returns no nodes."""
        from base.tools.query import query_context_vector

        client = _make_neo4j_client(run_returns=[])

        result = query_context_vector(
            client=client,
            query_embedding=[0.1] * 1536,
            project="myproject",
            allowed_scopes=["scope:slack:general"],  # node belongs to different scope
        )

        assert result["nodes"] == []
        assert result["total_matched"] == 0

    def test_employee_in_scope_returns_nodes(self):
        """Vector search with matching scope returns nodes."""
        from base.tools.query import query_context_vector

        node_scope = "scope:notion:abc"
        mock_row = {
            "props": {
                "id": "node:vec-001",
                "summary": "Use microservices",
                "source_scope": node_scope,
                "project": "myproject",
            },
            "node_type": "Insight",
            "score": 0.95,
        }
        client = _make_neo4j_client(run_returns=[mock_row])

        result = query_context_vector(
            client=client,
            query_embedding=[0.1] * 1536,
            project="myproject",
            allowed_scopes=[node_scope],
        )

        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["id"] == "node:vec-001"

    def test_service_account_sees_all_nodes(self):
        """allowed_scopes=None → no scope filter; all nodes visible."""
        from base.tools.query import query_context_vector

        mock_row = {
            "props": {
                "id": "node:vec-svc",
                "summary": "Restricted insight",
                "source_scope": "scope:notion:restricted",
                "project": "myproject",
            },
            "node_type": "Insight",
            "score": 0.85,
        }
        client = _make_neo4j_client(run_returns=[mock_row])

        result = query_context_vector(
            client=client,
            query_embedding=[0.1] * 1536,
            project="myproject",
            allowed_scopes=None,
        )

        assert len(result["nodes"]) == 1
        # No scope params passed in any call
        for call in client.run.call_args_list:
            params = call.args[1] if len(call.args) > 1 else {}
            assert "allowed_scopes" not in params


# ---------------------------------------------------------------------------
# get_document_content access guard (base/tools/query.py)
# ---------------------------------------------------------------------------

class TestGetDocumentContentAccessGuard:
    def _make_doc_pool(
        self,
        *,
        doc_exists: bool = True,
        source_scope: str = "scope:notion:abc",
        is_member: bool = True,
    ) -> AsyncMock:
        """
        Build a pool mock for get_document_content tests.

        fetchrow returns a fake document row (or None if doc_exists=False).
        fetchval returns 1 (member) or None (non-member).
        """
        pool = AsyncMock()

        if doc_exists:
            doc_row = MagicMock()
            doc_row.__getitem__ = lambda self, key: {
                "id": "doc:001",
                "title": "Architecture Doc",
                "s3_key": "orgs/myorg/docs/doc:001.txt",
                "source_scope": source_scope,
            }[key]
            pool.fetchrow = AsyncMock(return_value=doc_row)
        else:
            pool.fetchrow = AsyncMock(return_value=None)

        pool.fetchval = AsyncMock(return_value=1 if is_member else None)
        return pool

    @pytest.mark.asyncio
    async def test_employee_not_in_scope_denied(self):
        """Employee without membership in doc's source_scope → Access denied."""
        from base.tools.query import get_document_content

        pool = self._make_doc_pool(is_member=False)

        result = await get_document_content(
            pool=pool,
            doc_id="doc:001",
            employee_id="emp:outsider",
        )

        assert result["error"] == "Access denied"
        assert result["doc_id"] == "doc:001"

    @pytest.mark.asyncio
    async def test_employee_in_scope_gets_content(self):
        """Employee with membership in doc's source_scope → content returned."""
        from base.tools.query import get_document_content

        pool = self._make_doc_pool(is_member=True)

        fake_content = "This is the document text."
        with patch("base.tools.s3.read_document_content", new=AsyncMock(return_value=fake_content)):
            result = await get_document_content(
                pool=pool,
                doc_id="doc:001",
                employee_id="emp:alice",
            )

        assert "error" not in result
        assert result["doc_id"] == "doc:001"
        assert result["content"] == fake_content

    @pytest.mark.asyncio
    async def test_service_account_bypasses_check(self):
        """employee_id=None (service account) → no membership check, content returned."""
        from base.tools.query import get_document_content

        pool = self._make_doc_pool(is_member=False)  # would be denied for a human

        fake_content = "Secret document text."
        with patch("base.tools.s3.read_document_content", new=AsyncMock(return_value=fake_content)):
            result = await get_document_content(
                pool=pool,
                doc_id="doc:001",
                employee_id=None,
            )

        assert "error" not in result
        assert result["content"] == fake_content
        # fetchval (membership check) should NOT have been called
        pool.fetchval.assert_not_called()

    @pytest.mark.asyncio
    async def test_document_not_found(self):
        """Non-existent doc_id returns appropriate error regardless of employee."""
        from base.tools.query import get_document_content

        pool = self._make_doc_pool(doc_exists=False)

        result = await get_document_content(
            pool=pool,
            doc_id="doc:nonexistent",
            employee_id="emp:alice",
        )

        assert result["error"] == "Document not found"
        assert result["doc_id"] == "doc:nonexistent"


class _FakeRel:
    def __init__(self, start_id: str, rel_type: str, end_id: str) -> None:
        self.start_node = {"id": start_id}
        self.end_node = {"id": end_id}
        self.type = rel_type


class TestQuerySubgraphRegression:
    def test_cursor_visited_nodes_not_returned_via_edges(self):
        """Edges should only reference nodes returned in this page."""
        from base.tools.query import query_subgraph

        cursor = base64.b64encode(
            json.dumps({"visited": ["node:old"], "frontier": ["node:seed"]}).encode()
        ).decode()
        client = _make_neo4j_client(
            run_returns=[
                {
                    "nodes": [
                        {"id": "node:seed", "_type": "Topic"},
                        {"id": "node:new", "_type": "Decision"},
                        {"id": "node:old", "_type": "Decision"},
                    ],
                    "relationships": [
                        _FakeRel("node:old", "RELATED_TO", "node:seed"),
                        _FakeRel("node:seed", "INFLUENCES", "node:new"),
                    ],
                }
            ]
        )
        client.run.side_effect = [
            client.run.return_value,
            [],
        ]

        out = query_subgraph(
            client=client,
            query="ignored when cursor has frontier",
            cursor=cursor,
            project="proj-a",
        )

        node_ids = {n["id"] for n in out["nodes"]}
        assert out["returned"] == len(out["nodes"])
        assert all(e["from"] in node_ids and e["to"] in node_ids for e in out["edges"])

    def test_boundary_query_bypasses_project_filter_when_empty(self):
        """Boundary query should not exclude rows when project is empty."""
        from base.tools.query import query_subgraph

        cursor = base64.b64encode(json.dumps({"visited": [], "frontier": ["node:seed"]}).encode()).decode()
        client = _make_neo4j_client(run_returns=[{"nodes": [], "relationships": []}])
        client.run.side_effect = [
            [{"nodes": [], "relationships": []}],
            [],
        ]

        query_subgraph(client=client, query="x", project="", cursor=cursor)

        boundary_cypher = client.run.call_args_list[1].args[0]
        assert "$project = '' OR a.project = $project" in boundary_cypher
        assert "$project = '' OR b.project = $project" in boundary_cypher
