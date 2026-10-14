"""
processor/tools/eidetix.py — Eidetix HTTP client + shared MCP server.

Two interfaces:

1. EidetixClient — async HTTP client for pre-fetching data *before* agent
   invocation (e.g., Phase 1 entity pre-fetch). Calls the webserver REST API
   directly, bypassing MCP transport.

2. mcp_server — module-level MCPServerSSE that pydantic-ai agents use as a
   toolset. Phase 1 and Phase 2 each filter it to only the tools they need.
   Call close_shared_client() on worker shutdown to release the HTTP connection.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from pydantic_ai.mcp import MCPServerStreamableHTTP as _MCPServerHTTPClass

from processor.config import config

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, doubles each retry


# ---------------------------------------------------------------------------
# EidetixClient — direct HTTP client for pre-agent data fetching
# ---------------------------------------------------------------------------


class EidetixClient:
    """
    Async HTTP client for Eidetix graph operations.

    Uses the webserver's REST API (/api/*) for data that must be fetched
    before the agent is invoked (e.g., entity pre-fetch in Phase 1).
    Not used for tool calls during agent execution — that goes through MCP.

    Sends X-Eidetix-Org on every request so the shared webserver routes
    queries to the correct org's Postgres + Neo4j env.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = (base_url or config.eidetix_url).rstrip("/")
        self._api_key = api_key or config.eidetix_api_key
        _org = org_id or config.org_id
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if _org:
            headers["X-Eidetix-Org"] = _org
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "EidetixClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        return await self._request("GET", path, params=params)

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json: dict | None = None,
    ) -> dict | list:
        """Execute HTTP request with exponential backoff retry on transient errors."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        backoff = _RETRY_BACKOFF

        for attempt in range(_MAX_RETRIES):
            try:
                if method == "GET":
                    resp = await self._client.get(url, params=params or {})
                else:
                    resp = await self._client.post(url, json=json)
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if 400 <= status < 500 and status != 429:
                    _log.error("Eidetix %s %s → %s: %s", method, path, status, exc.response.text)
                    raise
                last_exc = exc
                _log.warning(
                    "Eidetix %s %s → %s (attempt %d/%d), retrying in %.1fs",
                    method, path, status, attempt + 1, _MAX_RETRIES, backoff,
                )

            except httpx.RequestError as exc:
                last_exc = exc
                _log.warning(
                    "Eidetix %s %s network error (attempt %d/%d): %s, retrying in %.1fs",
                    method, path, attempt + 1, _MAX_RETRIES, exc, backoff,
                )

            await asyncio.sleep(backoff)
            backoff *= 2

        _log.error("Eidetix %s %s failed after %d retries", method, path, _MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    # ── Pre-fetch methods (called before agent invocation) ────────────────────

    async def list_entities_for_scope(
        self,
        org_id: str,
        source_scope: str,
        source_provider: str,
    ) -> list[dict]:
        """
        Return Entity nodes associated with documents from this scope.

        Used in Phase 1 to give the identity agent a warm-start list of entities
        it's likely to encounter, reducing tool call volume.
        """
        result = await self._get(
            "/api/entities/search",
            params={
                "q": source_provider,
                "project": org_id.replace("org:", ""),
            },
        )
        if isinstance(result, dict):
            return result.get("results", [])
        return result if isinstance(result, list) else []

    async def search_context(
        self,
        query: str,
        limit: int = 20,
    ) -> dict:
        """
        Search the knowledge graph using the new unified search MCP tool.

        Calls the search MCP tool on the webserver and returns a dict with key
        ``results`` containing the unified ranked results list.
        Raises on network or server error (caller should catch and fall through).
        """
        return await self._call_tool_json(
            "search",
            {
                "query": query,
                "limit": limit,
            },
        )

    async def gather_context_batch(
        self,
        query: str,
        project: str,
        node_ids: list[str] | None = None,
    ) -> dict:
        """
        Best-effort context prefetch composed from redesign MCP tools.

        Returns a legacy-shaped payload consumed by Phase 2 prefetch:
        {
          "search_results": {"merged": [...]},
          "subgraph": {"nodes": [...], "edges": [...]},
          "traversal": {"results": [{"node_id": "...", "nodes": [...], "edges": [...]}]},
        }
        """
        del project  # project is resolved server-side from auth context

        search_payload = await self.search_context(query=query, limit=20)
        search_results = search_payload.get("results", []) if isinstance(search_payload, dict) else []

        deduped_ids: list[str] = []
        seen: set[str] = set()
        for raw in (node_ids or []):
            nid = str(raw or "").strip()
            if nid and nid not in seen:
                seen.add(nid)
                deduped_ids.append(nid)
        for row in search_results:
            nid = str(row.get("id") or "").strip()
            if nid and nid not in seen:
                seen.add(nid)
                deduped_ids.append(nid)
            if len(deduped_ids) >= 20:
                break

        subgraph: dict = {}
        traversal: dict = {}
        if deduped_ids:
            subgraph = await self._call_tool_json(
                "get_graph_expanded",
                {"node_ids": deduped_ids[:12], "max_hops": 1},
            )
            graph = await self._call_tool_json(
                "get_graph",
                {"node_ids": deduped_ids[:20], "max_hops": 1},
            )
            traversal = {
                "results": [{
                    "node_id": "batch",
                    "nodes": graph.get("nodes", []) if isinstance(graph, dict) else [],
                    "edges": graph.get("edges", []) if isinstance(graph, dict) else [],
                }],
            }

        return {
            "search_results": {"merged": search_results},
            "subgraph": subgraph if isinstance(subgraph, dict) else {},
            "traversal": traversal if isinstance(traversal, dict) else {"results": []},
        }

    async def _call_tool_json(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Call one MCP tool and decode the first text content item as JSON."""
        payload: dict = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        result = await self._request("POST", "/mcp/sse", json=payload)
        if not isinstance(result, dict):
            return {}
        content = result.get("result", {}).get("content", [])
        if not (isinstance(content, list) and content):
            return {}
        text = content[0].get("text", "{}")
        if not isinstance(text, str):
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            _log.debug("Eidetix tool %s returned non-JSON text payload", tool_name)
            return {}
        return decoded if isinstance(decoded, dict) else {}



# ---------------------------------------------------------------------------
# Shared MCP server — SSE transport to the webserver's /mcp/sse endpoint.
#
# Phase 1 and Phase 2 agents each call .filtered() on this to expose only
# the tools they need. pydantic-ai auto-connects on the first agent run and
# reuses the connection for subsequent runs.
# ---------------------------------------------------------------------------

def _build_mcp_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if config.eidetix_api_key:
        headers["Authorization"] = f"Bearer {config.eidetix_api_key}"
    if config.org_id:
        headers["X-Eidetix-Org"] = config.org_id
    return headers


_mcp_http_client: httpx.AsyncClient = httpx.AsyncClient(
    headers=_build_mcp_headers(),
    timeout=_DEFAULT_TIMEOUT,
)

mcp_server = _MCPServerHTTPClass(
    f"{config.eidetix_url.rstrip('/')}/mcp/sse",
    http_client=_mcp_http_client,
)


async def close_shared_client() -> None:
    """Close the shared MCP HTTP client. Called on worker shutdown."""
    await _mcp_http_client.aclose()
