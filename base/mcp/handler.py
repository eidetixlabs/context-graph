from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)

import asyncpg

import os

from base.db.access import get_allowed_scopes
from base.db.identity import search_entities_pg
from base.inference_client import InferenceClient
from base.neo4j_client import Neo4jClient
from base.tools import query as q
from base.tools.repository_context import (
    get_repository_context,
    render_repository_profile_markdown,
)
from base.tools.repository_context_notes import (
    NOTE_CATEGORIES,
    add_repository_context_note,
    supersede_repository_context_note,
)
from base.tools.repository_refresh import request_repository_refresh
from base.tools.repository_refresh import normalize_repository_name
from base.tools.schema import get_schema
from base.tools.document_ingest import persist_portal_document
from base.tools.trace_ingest import normalize_trace_entries, store_traces_s3

_EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai")

_openai_embed_client = None


def _get_openai_embed_client():
    global _openai_embed_client
    if _openai_embed_client is None:
        from base.openai_embedding_client import OpenAIEmbeddingClient  # noqa: PLC0415
        _openai_embed_client = OpenAIEmbeddingClient()
    return _openai_embed_client


class ResourceNotFound(Exception):
    def __init__(self, uri: str) -> None:
        super().__init__(f"Resource not found: {uri}")
        self.uri = uri


_TRACE_GUIDE_PATH = Path(__file__).with_name("trace_extraction_guide.md")


@lru_cache(maxsize=1)
def _get_trace_extraction_guide() -> str:
    return _TRACE_GUIDE_PATH.read_text(encoding="utf-8")


@dataclass(slots=True)
class McpContext:
    client: Neo4jClient
    employee: dict[str, Any]
    db_pool: asyncpg.Pool
    org_id: str
    inference_client_getter: Callable[[], InferenceClient]


class BaseMcpHandler:
    SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-06-18"}
    PREFERRED_PROTOCOL_VERSION = "2025-06-18"
    READ_ONLY_SERVICE_TOOLS = {"recall", "search", "get_content", "get_graph"}

    def __init__(self, ctx: McpContext) -> None:
        self.ctx = ctx

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            client_version = params.get("protocolVersion", self.PREFERRED_PROTOCOL_VERSION)
            negotiated = (
                client_version
                if client_version in self.SUPPORTED_PROTOCOL_VERSIONS
                else self.PREFERRED_PROTOCOL_VERSION
            )
            return {
                "protocolVersion": negotiated,
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "context-graph", "version": "0.1.0"},
                "instructions": self.initialize_instructions(),
            }

        if method == "tools/list":
            return {"tools": self.tool_list()}

        if method == "tools/call":
            name = params.get("name", "")
            args = params.get("arguments", {})
            text = await self.call_tool(name, args)
            return {"content": [{"type": "text", "text": text}]}

        if method == "resources/list":
            return {"resources": self.resources()}

        if method == "resources/read":
            uri = params.get("uri", "")
            return self.read_resource(uri)

        if method in ("notifications/initialized", "ping"):
            return {}

        raise ValueError(f"Unknown method: {method}")

    def initialize_instructions(self) -> str:
        return (
            "Context Graph is local long-term memory: past Decisions, Constraints, Insights, "
            "and Signals, linked by how they influenced each other. "
            "Use recall first — it returns ranked hits, 1-hop neighbors, and source excerpts "
            "in one call (detail=full by default). "
            "For drill-down: search (ranked hits only), get_graph (topology map), "
            "get_graph_expanded (detailed content + edges), get_content (full source document). "
            "resolve_entities turns names/aliases into canonical IDs before searching. "
            "Index a markdown/text document with ingest_document. "
            "At the end of a session, extract Decision/Constraint/Insight/Signal traces "
            "via get_schema then ingest_traces."
        )

    def resources(self) -> list[dict[str, Any]]:
        return [
            {
                "uri": "eidetix://schema/semantics",
                "name": "semantics_schema",
                "description": (
                    "Authoritative semantics.json schema: trace format, tag values, "
                    "category types, and ingestion rules for Context Graph observations."
                ),
                "mimeType": "application/json",
            }
        ]

    def read_resource(self, uri: str) -> dict[str, Any]:
        if uri == "eidetix://schema/semantics":
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": get_schema()}]}
        raise ResourceNotFound(uri)

    def tool_list(self) -> list[dict[str, Any]]:
        return self._filter_tools([
            {
                "name": "get_repository_context",
                "description": (
                    "Read the canonical current GitHub repository profile directly "
                    "by owner/repo, augment it with task-relevant repository-scoped "
                    "Decisions, Constraints, and Insights, and assess it against "
                    "the caller's local git state. This tool never writes the repository."
                ),
                "inputSchema": self._repository_context_schema(require_task=True),
            },
            {
                "name": "export_repository_profile",
                "description": (
                    "Read and deterministically render the canonical repository "
                    "profile as cited Markdown, including generation time, ref, "
                    "commit, freshness warnings, and local verification guidance. "
                    "This tool never writes the repository."
                ),
                "inputSchema": self._repository_context_schema(require_task=False),
            },
            {
                "name": "refresh_repository_profile",
                "description": (
                    "Request fresh GitHub evidence collection for an accessible "
                    "repository and force profile regeneration after collection. "
                    "When multiple active connectors match the repository, pass "
                    "connector_id to select which connector to refresh. "
                    "This never derives shared context from local feature branches."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repository": {
                            "type": "string",
                            "description": "Canonical GitHub identity in owner/repo form",
                        },
                        "connector_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Optional active GitHub connector ID; required to "
                                "disambiguate when multiple connectors match"
                            ),
                        },
                    },
                    "required": ["repository"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "recall",
                "description": (
                    "One-shot cited recall: returns two independent sections. "
                    "documents[]: document chunks ranked by vector similarity, each with doc_id, title, "
                    "chunk_index, similarity, and (detail=full) source_excerpt pulled directly from S3. "
                    "nodes[]: semantic graph hits (Claims, Entities, Facts, …) ranked by RRF, each with "
                    "snippet, source_doc pointer, and 1-hop neighbors. "
                    "Use this first for any question. Fall back to search/get_graph/get_content for drill-down."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Topic or question to look up"},
                        "repo": self._repo_schema(),
                        "limit": {
                            "type": "number",
                            "description": "Max semantic nodes to return (default 10, max 50)",
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["full", "light"],
                            "description": (
                                "full (default): includes source_excerpt in documents[] fetched from S3. "
                                "light: skips S3 reads, returns chunk pointers and node metadata only."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search",
                "description": (
                    "Find anything the team has recorded about a topic, decision, problem, person, "
                    "or system. Always start here. Returns ranked matches with id, type, label, "
                    "score, and snippet."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Topic or question to look up"},
                        "repo": self._repo_schema(),
                        "limit": {"type": "number", "description": "Max results to return (default 20, max 50)"},
                        "chunk_limit": {
                            "type": "number",
                            "description": "Max document candidates to consider (default 10)",
                        },
                        "node_limit": {
                            "type": "number",
                            "description": "Max graph candidates to consider (default 20)",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs from search results to map related context around",
                        },
                        "max_hops": {"type": "number", "description": "Connection depth (default 2, max 4)"},
                    },
                    "required": ["node_ids"],
                },
                "description": (
                    "Given ids from search, see what else is connected to them and how. "
                    "Returns a lightweight map of related items and links, with no detailed content."
                ),
            },
            {
                "name": "get_graph_expanded",
                "description": (
                    "Like get_graph, but focused and detailed. Use when you know the relevant area "
                    "and want full content for each item. Pass max_hops=0 to read just the requested "
                    "nodes with no traversal; default 1 adds their immediate connections."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "node_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "IDs to expand with detailed content",
                        },
                        "max_hops": {
                            "type": "number",
                            "description": "Connection depth: 0 = nodes only, 1 = default, max 2",
                        },
                    },
                    "required": ["node_ids"],
                },
            },
            {
                "name": "get_content",
                "description": "Read the original source document behind a result.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Document ID from search/get_graph_expanded source_doc",
                        },
                        "chunk_index": {"type": "number", "description": "Optional document chunk index"},
                    },
                    "required": ["doc_id"],
                },
            },
            {
                "name": "resolve_entities",
                "description": (
                    "Turn people/service/tool names into canonical entity records before searching "
                    "or expanding context. Supports batch lookups by default."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names or aliases to resolve in one request",
                        },
                        "name": {"type": "string", "description": "Single name convenience input"},
                        "query": {"type": "string", "description": "Backward-compatible single-name input"},
                    },
                },
            },
        ])

    @staticmethod
    def _repo_schema() -> dict[str, Any]:
        return {
            "type": "string",
            "description": (
                "Optional canonical GitHub repository in owner/repo form. "
                "Omit for organization-wide cross-repository recall."
            ),
        }

    def _tool_allowed(self, name: str) -> bool:
        if self.ctx.employee.get("principal_type") != "service":
            return True
        scopes = set(self.ctx.employee.get("auth_scopes") or [])
        if "*" in scopes or "mcp:write" in scopes:
            return True
        return "mcp:read" in scopes and name in self.READ_ONLY_SERVICE_TOOLS

    def _is_read_only_service(self) -> bool:
        if self.ctx.employee.get("principal_type") != "service":
            return False
        scopes = set(self.ctx.employee.get("auth_scopes") or [])
        return "mcp:read" in scopes and "mcp:write" not in scopes and "*" not in scopes

    def _filter_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [tool for tool in tools if self._tool_allowed(str(tool.get("name") or ""))]

    @staticmethod
    def _repo_arg(args: dict[str, Any]) -> str | None:
        raw = args.get("repo")
        if raw is None or raw == "":
            return None
        normalized, _ = normalize_repository_name(raw)
        return normalized

    @staticmethod
    def _repository_context_schema(*, require_task: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "repository": {
                "type": "string",
                "description": "Canonical GitHub identity in owner/repo form",
            },
            "task": {
                "type": "string",
                "description": "Current implementation, debugging, or review task",
            },
            "local_head": {
                "type": "string",
                "description": "Optional local git HEAD commit SHA",
            },
            "merge_base": {
                "type": "string",
                "description": (
                    "Optional merge-base SHA between the local checkout and "
                    "the profiled upstream ref; compute and provide it whenever "
                    "local_head differs from the profile so feature-branch commits "
                    "are not mistaken for stale or disposable work"
                ),
            },
            "working_tree_dirty": {
                "type": "boolean",
                "description": "Whether the local working tree has uncommitted changes",
            },
            "changed_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional locally changed or task-relevant paths",
            },
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["repository", "task"] if require_task else ["repository"],
            "additionalProperties": False,
        }

    async def _allowed_scopes(self) -> list[str] | None:
        if self.ctx.employee.get("principal_type") == "service":
            return None
        return await get_allowed_scopes(self.ctx.db_pool, self.ctx.employee.get("id"))

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        if not self._tool_allowed(name):
            raise PermissionError(f"Tool '{name}' is not allowed for this service token")
        allowed_scopes = await self._allowed_scopes()

        match name:
            case "get_repository_context" | "export_repository_profile":
                task = args.get("task")
                if name == "export_repository_profile" and not task:
                    task = "export repository profile"
                result = await get_repository_context(
                    pool=self.ctx.db_pool,
                    client=self.ctx.client,
                    org_id=self.ctx.org_id,
                    allowed_scopes=allowed_scopes,
                    repository=args.get("repository"),
                    task=task,
                    local_head=args.get("local_head"),
                    merge_base=args.get("merge_base"),
                    working_tree_dirty=args.get("working_tree_dirty", False),
                    changed_paths=args.get("changed_paths"),
                )
                if name == "export_repository_profile":
                    return render_repository_profile_markdown(result)
                return json.dumps(result)

            case "refresh_repository_profile":
                connector_id = args.get("connector_id")
                if connector_id is not None:
                    if not isinstance(connector_id, str) or not connector_id.strip():
                        raise ValueError(
                            "connector_id must be a non-empty string when supplied"
                        )
                    connector_id = connector_id.strip()
                return json.dumps(
                    await request_repository_refresh(
                        pool=self.ctx.db_pool,
                        org_id=self.ctx.org_id,
                        repository=args.get("repository"),
                        reason="manual",
                        connector_id=connector_id,
                        allowed_scopes=allowed_scopes,
                        force_generation=True,
                    )
                )

            case "search":
                query_text = args.get("query", "")
                repo = self._repo_arg(args)
                chunk_limit = max(1, min(int(args.get("chunk_limit", 10)), 50))
                node_limit = max(1, min(int(args.get("node_limit", 20)), 50))
                result_limit = max(1, min(int(args.get("limit", 20)), 50))
                _t_embed = time.perf_counter()
                if _EMBEDDING_PROVIDER == "inference":
                    bge_embs, minilm_embs = await asyncio.gather(
                        self.ctx.inference_client_getter().embed_bge(["query: " + query_text]),
                        self.ctx.inference_client_getter().embed_minilm([query_text]),
                    )
                else:
                    bge_embs, minilm_embs = await asyncio.gather(
                        _get_openai_embed_client().embed_large(["query: " + query_text]),
                        _get_openai_embed_client().embed_small([query_text]),
                    )
                _log.info("[search:profile] embed_gather %.1fms", (time.perf_counter() - _t_embed) * 1000)
                return json.dumps(
                    await q.search(
                        client=self.ctx.client,
                        pool=self.ctx.db_pool,
                        query=query_text,
                        chunk_embedding=bge_embs[0],
                        neo4j_embedding=minilm_embs[0],
                        org_id=self.ctx.org_id,
                        limit=result_limit,
                        chunk_limit=chunk_limit,
                        node_limit=node_limit,
                        allowed_scopes=allowed_scopes,
                        repo=repo,
                    )
                )

            case "get_graph":
                raw_hops = args.get("max_hops", 2)
                max_hops = max(1, min(int(raw_hops), 4))
                return json.dumps(
                    await asyncio.to_thread(
                        q.get_graph,
                        client=self.ctx.client,
                        node_ids=args.get("node_ids", []),
                        max_hops=max_hops,
                        allowed_scopes=allowed_scopes,
                    )
                )

            case "get_graph_expanded":
                raw_hops = args.get("max_hops", 1)
                max_hops = max(0, min(int(raw_hops), 2))
                return json.dumps(
                    await asyncio.to_thread(
                        q.get_graph_expanded,
                        client=self.ctx.client,
                        node_ids=args.get("node_ids", []),
                        max_hops=max_hops,
                        allowed_scopes=allowed_scopes,
                    )
                )

            case "get_content":
                raw_chunk = args.get("chunk_index")
                chunk_index = int(raw_chunk) if raw_chunk is not None else None
                return json.dumps(
                    await q.get_document_content(
                        self.ctx.db_pool,
                        args["doc_id"],
                        chunk_index,
                        employee_id=(
                            None
                            if self.ctx.employee.get("principal_type") == "service"
                            else self.ctx.employee.get("id")
                        ),
                    )
                )

            case "recall":
                query_text = args.get("query", "")
                repo = self._repo_arg(args)
                recall_limit = max(1, min(int(args.get("limit", 10)), 50))
                detail = args.get("detail", "full")
                if detail not in ("full", "light"):
                    detail = "full"
                _t_embed = time.perf_counter()
                if _EMBEDDING_PROVIDER == "inference":
                    bge_embs, minilm_embs = await asyncio.gather(
                        self.ctx.inference_client_getter().embed_bge(["query: " + query_text]),
                        self.ctx.inference_client_getter().embed_minilm([query_text]),
                    )
                else:
                    bge_embs, minilm_embs = await asyncio.gather(
                        _get_openai_embed_client().embed_large(["query: " + query_text]),
                        _get_openai_embed_client().embed_small([query_text]),
                    )
                _log.info("[recall:profile] embed_gather %.1fms", (time.perf_counter() - _t_embed) * 1000)
                return json.dumps(
                    await q.recall(
                        client=self.ctx.client,
                        pool=self.ctx.db_pool,
                        query=query_text,
                        chunk_embedding=bge_embs[0],
                        neo4j_embedding=minilm_embs[0],
                        org_id=self.ctx.org_id,
                        limit=recall_limit,
                        detail=detail,
                        allowed_scopes=allowed_scopes,
                        employee_id=(
                            None
                            if self.ctx.employee.get("principal_type") == "service"
                            else self.ctx.employee.get("id")
                        ),
                        repo=repo,
                    )
                )

            case "resolve_entities":
                queries = args.get("names")
                if not isinstance(queries, list):
                    single_query = (args.get("name") or args.get("query") or "").strip()
                    queries = [single_query] if single_query else []
                deduped: list[str] = []
                seen_q: set[str] = set()
                for q_item in queries:
                    q_s = str(q_item).strip()
                    if q_s and q_s not in seen_q:
                        seen_q.add(q_s)
                        deduped.append(q_s)
                batch_results = []
                for q_item in deduped[:50]:
                    matches = await search_entities_pg(self.ctx.db_pool, self.ctx.org_id, q_item)
                    batch_results.append({"query": q_item, "matches": matches, "matched": len(matches)})
                return json.dumps(
                    {"results": batch_results, "requested": len(queries), "executed": len(batch_results)}
                )

            case _:
                custom = await self.call_custom_tool(name=name, args=args)
                if custom is not None:
                    return custom
                raise ValueError(f"Unknown tool: {name}")

    async def call_custom_tool(self, *, name: str, args: dict[str, Any]) -> str | None:
        return None


class DesktopMcpHandler(BaseMcpHandler):
    @staticmethod
    def _repository_note_schema(*, supersede: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "owner": {
                "type": "string",
                "description": "GitHub repository owner or organization",
            },
            "repo": {
                "type": "string",
                "description": "GitHub repository name",
            },
            "category": {
                "type": "string",
                "enum": sorted(NOTE_CATEGORIES),
                "description": "The repository profile area this note informs",
            },
            "body": {
                "type": "string",
                "minLength": 1,
                "description": "Concise repository context to preserve",
            },
        }
        required = ["owner", "repo", "category", "body"]
        if supersede:
            properties["supersedes_id"] = {
                "type": "string",
                "minLength": 1,
                "description": "Active note ID being replaced",
            }
            required = ["owner", "repo", "supersedes_id", "category", "body"]
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def tool_list(self) -> list[dict[str, Any]]:
        return self._filter_tools(super().tool_list() + [
            {
                "name": "add_repository_context_note",
                "description": (
                    "Add attributed context to an accessible GitHub repository "
                    "and queue its repository profile for regeneration."
                ),
                "inputSchema": self._repository_note_schema(supersede=False),
            },
            {
                "name": "supersede_repository_context_note",
                "description": (
                    "Replace one active context note in the same accessible "
                    "GitHub repository while preserving its history."
                ),
                "inputSchema": self._repository_note_schema(supersede=True),
            },
        ])

    async def call_custom_tool(self, *, name: str, args: dict[str, Any]) -> str | None:
        if name not in {
            "add_repository_context_note",
            "supersede_repository_context_note",
        }:
            return None

        employee_id = str(self.ctx.employee.get("id") or "")
        common = {
            "pool": self.ctx.db_pool,
            "org_id": self.ctx.org_id,
            "employee_id": employee_id,
            "owner": args.get("owner"),
            "repo": args.get("repo"),
            "category": args.get("category"),
            "body": args.get("body"),
        }
        if name == "supersede_repository_context_note":
            result = await supersede_repository_context_note(
                **common,
                supersedes_id=args.get("supersedes_id"),
            )
        else:
            result = await add_repository_context_note(**common)

        concise_result = {
            key: result[key]
            for key in (
                "id",
                "repository",
                "category",
                "supersedes_id",
                "author_employee_id",
                "created_at",
                "enqueued_profiles",
            )
            if key in result
        }
        return json.dumps(concise_result)


class AgentMcpHandler(BaseMcpHandler):
    TRACE_GUIDE_URI = "eidetix://guide/trace-extraction"

    def initialize_instructions(self) -> str:
        if self._is_read_only_service():
            return (
                "Context Graph provides read-only organization memory. Use recall first, "
                "optionally passing repo in owner/repo form for repository-specific "
                "business and architecture context. Use search, get_graph, and "
                "get_content for drill-down. This token cannot ingest or modify data."
            )
        return (
            "Context Graph is local long-term memory. "
            "Use recall first for knowledge lookups. "
            "For drill-down use search, get_graph, get_graph_expanded, get_content, and resolve_entities. "
            "Index markdown/text with ingest_document. "
            "At the end of EVERY completed action/chat, run semantic extraction as a hard gate before finalizing. "
            "If there is any signal (Decision/Constraint/Insight/Signal), you MUST write trace(s). "
            "Do not skip extraction because it seems small; omission is the failure mode. "
            "Before writing traces, read resources/read for eidetix://guide/trace-extraction or call get_schema "
            "for the exact schema and required fields. Then submit traces via ingest_traces. "
            "A task/chat is not complete until this is done. If truly no signal exists, explicitly state "
            "'Semantic extraction: no signal in this chat.'"
        )

    def resources(self) -> list[dict[str, Any]]:
        resources = super().resources()
        if self._is_read_only_service():
            return resources
        return resources + [
            {
                "uri": self.TRACE_GUIDE_URI,
                "name": "trace_extraction_guide",
                "description": (
                    "How to extract semantic traces: what to capture, hard gates, "
                    "node envelope, mention types, and tag usage."
                ),
                "mimeType": "text/markdown",
            }
        ]

    def read_resource(self, uri: str) -> dict[str, Any]:
        if uri == self.TRACE_GUIDE_URI:
            if self._is_read_only_service():
                raise ResourceNotFound(uri)
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": _get_trace_extraction_guide(),
                    }
                ]
            }
        return super().read_resource(uri)

    def tool_list(self) -> list[dict[str, Any]]:
        tools = super().tool_list()
        tools.append(
            {
                "name": "get_schema",
                "description": (
                    "Return the authoritative Context Graph trace schema. "
                    "Call this before writing any observation trace."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            }
        )
        tools.append(
            {
                "name": "ingest_traces",
                "description": (
                    "Directly ingest observation traces from an agent MCP client. "
                    "Use this at the end of each action/chat after semantic extraction."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "Optional project override"},
                        "traces": {
                            "type": "array",
                            "description": "Trace entries to ingest",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "trace_id": {"type": "string"},
                                    "ts": {"type": "string"},
                                    "session_id": {"type": "string"},
                                    "project": {"type": "string"},
                                    "repo": self._repo_schema(),
                                    "node": {"type": "object"},
                                    "keywords": {"type": "array", "items": {"type": "string"}},
                                    "mentions": {"type": "array", "items": {"type": "object"}},
                                },
                                "required": ["trace_id", "ts", "node"],
                            },
                        },
                    },
                    "required": ["traces"],
                },
            }
        )
        tools.append(
            {
                "name": "ingest_document",
                "description": (
                    "Index a markdown or text document into Context Graph. "
                    "The portal processor extracts Decisions, Constraints, Insights, "
                    "and Signals so they become recallable. Prefer this for notes, "
                    "architecture docs, and meeting write-ups."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Document title"},
                        "content": {
                            "type": "string",
                            "description": "Full markdown or plain-text body",
                        },
                        "doc_type": {
                            "type": "string",
                            "enum": [
                                "architecture",
                                "hld",
                                "lld",
                                "design",
                                "roadmap",
                                "ticket_retrospective",
                                "context",
                            ],
                            "description": "Optional document kind. Defaults to context.",
                        },
                        "source_ref": {
                            "type": "string",
                            "description": "Optional stable id for idempotent re-ingest",
                        },
                        "repo": self._repo_schema(),
                    },
                    "required": ["title", "content"],
                },
            }
        )
        return self._filter_tools(tools)

    async def call_custom_tool(self, *, name: str, args: dict[str, Any]) -> str | None:
        if name == "get_schema":
            return get_schema()

        if name == "ingest_document":
            title = str(args.get("title") or "").strip()
            content = str(args.get("content") or "").strip()
            if not title or not content:
                raise ValueError("ingest_document requires title and content")
            doc_type = str(args.get("doc_type") or "context").strip() or "context"
            repo = args.get("repo")
            if isinstance(repo, str) and not repo.strip():
                repo = None
            result = await persist_portal_document(
                pool=self.ctx.db_pool,
                org_id=self.ctx.org_id,
                employee_id=(
                    str(self.ctx.employee.get("id"))
                    if self.ctx.employee.get("principal_type") == "employee"
                    else None
                ),
                repo=repo,
                doc_type=doc_type,
                title=title,
                content=content,
                source_ref=args.get("source_ref"),
            )
            return json.dumps(result)

        if name != "ingest_traces":
            return None

        traces = args.get("traces")
        if not isinstance(traces, list):
            raise ValueError("ingest_traces requires a 'traces' array")

        org_id = self.ctx.employee.get("org_id") or ""
        project = org_id.removeprefix("org:") if org_id else str(args.get("project") or "")
        author_name = self.ctx.employee.get("canonical_name") or ""
        author_employee_id = str(self.ctx.employee.get("id") or "")
        normalized = normalize_trace_entries(
            traces,
            project=project,
            org_id=org_id,
            author_name=author_name,
            author_employee_id=author_employee_id,
        )
        storage_employee_id = str(self.ctx.employee.get("id") or "unknown")
        result = await store_traces_s3(
            project=project,
            org_id=org_id,
            traces=normalized,
            employee_id=storage_employee_id,
        )
        return json.dumps(
            {
                "ok": result["failed"] == 0,
                "accepted": result["stored"],
                "failed": result["failed"],
            }
        )
