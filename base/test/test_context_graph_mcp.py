from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from base.mcp import handler as handler_module
from base.mcp.handler import AgentMcpHandler, McpContext, ResourceNotFound
from base.tools.schema import SCHEMA
from webserver import main as webserver_main


class _Embeddings:
    async def embed_large(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    async def embed_small(self, texts: list[str]) -> list[list[float]]:
        return [[0.2] for _ in texts]


class _Acquire:
    def __init__(self, value) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return False


def _handler(*, scopes: list[str]) -> AgentMcpHandler:
    return AgentMcpHandler(
        McpContext(
            client=MagicMock(),
            employee={
                "id": "service-token-id",
                "org_id": "org:acme",
                "role": "service",
                "principal_type": "service",
                "auth_scopes": scopes,
            },
            db_pool=AsyncMock(),
            org_id="org:acme",
            inference_client_getter=MagicMock(),
        )
    )


def test_recall_and_search_schemas_expose_optional_repo() -> None:
    tools = {tool["name"]: tool for tool in _handler(scopes=["mcp:write"]).tool_list()}

    for name in ("recall", "search"):
        repo_schema = tools[name]["inputSchema"]["properties"]["repo"]
        assert repo_schema["type"] == "string"
        assert "owner/repo" in repo_schema["description"]
        assert "repo" not in tools[name]["inputSchema"]["required"]


def test_trace_schema_matches_node_embedded_repo_contract() -> None:
    fields = SCHEMA["trace_format"]["fields"]
    assert "node" in fields
    assert "repo" in fields
    assert "structured" not in fields
    assert "body" not in fields

    payload = webserver_main.TraceEntry(
        trace_id="trace-1",
        ts="2026-09-16T00:00:00Z",
        repo="Acme/Widgets",
        node={"type": "Insight", "label": "Ownership", "body": "Gateway owns auth"},
    )
    assert payload.repo == "acme/widgets"


def test_read_only_service_token_only_lists_recall_tools() -> None:
    handler = _handler(scopes=["mcp:read"])
    names = {tool["name"] for tool in handler.tool_list()}
    assert names == {"recall", "search", "get_content", "get_graph"}
    assert "cannot ingest" in handler.initialize_instructions()
    assert "ingest_traces" not in handler.initialize_instructions()
    assert {resource["uri"] for resource in handler.resources()} == {
        "eidetix://schema/semantics"
    }
    with pytest.raises(ResourceNotFound):
        handler.read_resource(handler.TRACE_GUIDE_URI)


@pytest.mark.asyncio
async def test_read_only_service_token_denies_ingest() -> None:
    handler = _handler(scopes=["mcp:read"])

    with pytest.raises(PermissionError, match="not allowed"):
        await handler.call_tool("ingest_traces", {"traces": []})
    with pytest.raises(PermissionError, match="not allowed"):
        await handler.call_tool("ingest_document", {"title": "x", "content": "y"})


def test_write_token_lists_ingest_document() -> None:
    names = {tool["name"] for tool in _handler(scopes=["mcp:write"]).tool_list()}
    assert "ingest_document" in names
    assert "ingest_traces" in names


@pytest.mark.asyncio
async def test_ingest_document_persists_portal_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = _handler(scopes=["mcp:write"])
    captured: dict = {}

    async def fake_persist(**kwargs):
        captured.update(kwargs)
        return {"id": "doc:portal:org:acme:notes", "status": "processing"}

    monkeypatch.setattr(handler_module, "persist_portal_document", fake_persist)
    result = json.loads(
        await handler.call_tool(
            "ingest_document",
            {"title": "Auth notes", "content": "We chose Postgres for sessions."},
        )
    )
    assert result["id"] == "doc:portal:org:acme:notes"
    assert captured["title"] == "Auth notes"
    assert captured["doc_type"] == "context"
    assert captured["org_id"] == "org:acme"


@pytest.mark.asyncio
async def test_read_only_service_token_denies_unlisted_custom_tool() -> None:
    handler = _handler(scopes=["mcp:read"])

    with pytest.raises(PermissionError, match="not allowed"):
        await handler.call_tool("get_schema", {})


@pytest.mark.asyncio
async def test_service_boundary_uses_principal_type_not_role_label() -> None:
    handler = _handler(scopes=["mcp:read"])
    handler.ctx.employee["role"] = "admin"

    assert {tool["name"] for tool in handler.tool_list()} == {
        "recall",
        "search",
        "get_content",
        "get_graph",
    }
    assert await handler._allowed_scopes() is None
    handler.ctx.db_pool.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_auth_resolves_scoped_service_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_row = {
        "id": "token-id",
        "org_id": "org:acme",
        "name": "Harness read only",
        "scopes": ["mcp:read"],
        "expires_at": datetime(2030, 1, 1, tzinfo=timezone.utc),
    }
    env_conn = AsyncMock()
    env_conn.fetchrow = AsyncMock(side_effect=[None, service_row])
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_Acquire(env_conn))
    org_ctx = SimpleNamespace(org_id="org:acme", pool=pool, neo4j=MagicMock())
    monkeypatch.setattr(
        webserver_main,
        "resolve_org_from_token",
        AsyncMock(return_value=org_ctx),
    )

    principal = await webserver_main.get_mcp_caller(
        authorization="Bearer scoped-service-token",
        central_db=AsyncMock(),
    )

    assert principal["principal_type"] == "service"
    assert principal["auth_scopes"] == ["mcp:read"]
    assert principal["client_type"] == "agent"
    env_conn.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["search", "recall"])
async def test_mcp_repo_argument_propagates_to_composite_search(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
) -> None:
    handler = _handler(scopes=["mcp:read"])
    captured: dict = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return {"results": []}

    async def fake_recall(**kwargs):
        captured.update(kwargs)
        return {"documents": [], "nodes": []}

    monkeypatch.setattr(handler_module, "_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(handler_module, "_get_openai_embed_client", lambda: _Embeddings())
    monkeypatch.setattr(handler_module.q, "search", fake_search)
    monkeypatch.setattr(handler_module.q, "recall", fake_recall)

    result = json.loads(
        await handler.call_tool(
            tool_name,
            {"query": "gateway ownership", "repo": "acme/widgets"},
        )
    )

    assert result
    assert captured["repo"] == "acme/widgets"
