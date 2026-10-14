"""
main.py — Context Graph MCP server.

Exposes:
  POST /ingest/documents            — index markdown/text (API_KEY)
  POST /ingest/traces               — store observation traces for the processor
  GET  /mcp/sse                     — MCP SSE transport (Cursor, Claude Desktop)
  POST /mcp/sse                     — MCP Streamable HTTP transport
  POST /mcp/message                 — MCP legacy HTTP transport
  GET  /health                      — unauthenticated liveness

Authentication:
  /mcp/* and /ingest/documents — Bearer API_KEY (synthetic service principal
  with mcp:write) or a DB session / service token.
  /api/* — API_KEY only.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Literal

_ROOT = Path(__file__).resolve().parent
if (_ROOT / "base").is_dir():
    _repo_root = _ROOT
else:
    _repo_root = _ROOT.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from base.auth.deps import CurrentEmployee, CurrentPrincipal
from base.db.client import DbConn, db_pool, get_db, org_pool_registry
from base.db.identity import search_entities_pg
from base.db.migrations import migrate_all_databases
from base.deprecated_api import deprecated_api
from base.inference_client import InferenceClient
from base.log_level import log_level_from_env
from base.mcp import AgentMcpHandler, McpContext, ResourceNotFound
from base.neo4j_client import Neo4jClient, org_neo4j_registry
from base.routing import OrgCtx, resolve_org_from_header, resolve_org_from_token
from base.tools import query as q
from base.tools.document_ingest import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentNotRetryableError,
    document_status,
    normalize_repo,
    normalize_source_ref,
    persist_portal_document,
    retry_portal_document,
)
from base.tools.ingest import SEMANTIC_TYPES, build_embed_text, ingest_entries
from base.tools.trace_ingest import normalize_trace_entries, store_traces_s3

_log = logging.getLogger(__name__)
_access_log = logging.getLogger("uvicorn.error")


def _configure_app_logging() -> None:
    level = log_level_from_env()
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:\t  %(message)s"))
        root.addHandler(handler)


def _http_status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""


INFERENCE_SERVICE_URL = os.getenv("INFERENCE_SERVICE_URL", "http://localhost:9111")
_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

_inference_client: InferenceClient | None = None
_openai_embed_client = None


def _get_inference_client() -> InferenceClient:
    global _inference_client
    if _inference_client is None:
        _inference_client = InferenceClient(INFERENCE_SERVICE_URL)
    return _inference_client


def _get_openai_embed_client():
    global _openai_embed_client
    if _openai_embed_client is None:
        from base.openai_embedding_client import OpenAIEmbeddingClient  # noqa: PLC0415

        _openai_embed_client = OpenAIEmbeddingClient()
    return _openai_embed_client


async def _embed_small(texts: list[str]) -> list[list[float]]:
    if _EMBEDDING_PROVIDER == "inference":
        return await _get_inference_client().embed_minilm(texts)
    return await _get_openai_embed_client().embed_small(texts)


async def _embed_nodes(entries: list[dict]) -> None:
    seen: dict[str, list[float]] = {}
    for entry in entries:
        for side in ("from", "to"):
            node = entry[side]
            node_type = node.get("type", "")
            if node_type not in SEMANTIC_TYPES:
                continue
            if node.get("embedding"):
                continue
            node_id = node["id"]
            if node_id in seen:
                node["embedding"] = seen[node_id]
                continue
            text = build_embed_text(node)
            if not text:
                continue
            embeddings = await _embed_small([text])
            node["embedding"] = embeddings[0]
            seen[node_id] = node["embedding"]


_API_KEY = os.environ.get("API_KEY", "")

_raw_org = os.environ.get("ORGANISATION_ID", "").strip()
DEFAULT_ORG_ID = _raw_org if _raw_org.startswith("org:") else f"org:{_raw_org}" if _raw_org else ""


def _service_principal(org_ctx: OrgCtx) -> dict[str, Any]:
    return {
        "id": "emp:local-service",
        "org_id": org_ctx.org_id,
        "email": "service@local",
        "canonical_name": "local-service",
        "role": "service",
        "status": "active",
        "entity_id": None,
        "session_id": None,
        "client_type": "agent",
        "expires_at": None,
        "principal_type": "service",
        "auth_scopes": ["*"],
        "resolved_org_ctx": org_ctx,
    }


async def _ensure_default_org_env_row() -> None:
    if not DEFAULT_ORG_ID:
        return

    pg_dsn = os.environ.get("DATABASE_URL", "").strip()
    neo4j_uri = os.environ.get("NEO4J_URI", "").strip()
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j").strip() or "neo4j"
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "").strip()
    neo4j_database = os.environ.get("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    slug = DEFAULT_ORG_ID.removeprefix("org:")

    if not (pg_dsn and neo4j_uri and neo4j_password):
        _log.warning(
            "Skipping default org_env seed for org=%s (missing DATABASE_URL/NEO4J_URI/NEO4J_PASSWORD)",
            DEFAULT_ORG_ID,
        )
        return

    try:
        await db_pool.get().execute(
            """
            INSERT INTO organizations (id, name, slug)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO NOTHING
            """,
            DEFAULT_ORG_ID,
            "Local Context Graph",
            slug,
        )
        await db_pool.get().execute(
            """
            INSERT INTO org_env
                (org_id, slug, email_domain, status, pg_dsn, neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
            VALUES
                ($1, $2, NULL, 'active', $3, $4, $5, $6, $7)
            ON CONFLICT (org_id) DO UPDATE
                SET slug           = EXCLUDED.slug,
                    pg_dsn         = EXCLUDED.pg_dsn,
                    neo4j_uri      = EXCLUDED.neo4j_uri,
                    neo4j_user     = EXCLUDED.neo4j_user,
                    neo4j_password = EXCLUDED.neo4j_password,
                    neo4j_database = EXCLUDED.neo4j_database,
                    status         = 'active',
                    updated_at     = now()
            """,
            DEFAULT_ORG_ID,
            slug,
            pg_dsn,
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            neo4j_database,
        )
        await db_pool.get().execute(
            """
            INSERT INTO employees
                (id, org_id, email, canonical_name, role, status, auth_provider)
            VALUES
                ('emp:local-service', $1, 'service@local', 'local-service', 'member', 'active', 'local')
            ON CONFLICT (id) DO NOTHING
            """,
            DEFAULT_ORG_ID,
        )
    except Exception as exc:
        _log.warning("Failed to seed default org for org=%s: %s", DEFAULT_ORG_ID, exc)


def verify_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
    if not _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on server",
        )
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    if not hmac.compare_digest(authorization[7:], _API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


_SESSION_QUERY = """
    SELECT
        e.id,
        e.org_id,
        e.email,
        e.canonical_name,
        e.role,
        e.status,
        e.entity_id,
        s.id         AS session_id,
        s.client_type,
        s.expires_at
    FROM sessions s
    JOIN employees e ON e.id = s.employee_id
    WHERE s.token_hash  = $1
      AND s.revoked_at  IS NULL
      AND s.expires_at  > now()
      AND e.status      = 'active'
"""

_SERVICE_TOKEN_QUERY = """
    SELECT id, org_id, name, scopes, expires_at
    FROM service_tokens
    WHERE token_hash = $1
      AND revoked_at IS NULL
      AND expires_at > now()
"""


async def get_mcp_caller(
    authorization: Annotated[str | None, Header()] = None,
    x_eidetix_org: Annotated[str | None, Header(alias="X-Eidetix-Org")] = None,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Use Bearer <API_KEY>.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_token = authorization[len("Bearer "):]

    if _API_KEY and hmac.compare_digest(raw_token, _API_KEY):
        org_ctx = await resolve_org_from_header(central_db, x_eidetix_org, DEFAULT_ORG_ID)
        return _service_principal(org_ctx)

    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    org_ctx = await resolve_org_from_token(central_db, raw_token, DEFAULT_ORG_ID)

    async with org_ctx.pool.acquire() as env_conn:
        row = await env_conn.fetchrow(_SESSION_QUERY, token_hash)
        if row is None:
            service_row = await env_conn.fetchrow(_SERVICE_TOKEN_QUERY, token_hash)
            if service_row is not None:
                await env_conn.execute(
                    "UPDATE service_tokens SET last_used_at = now() WHERE id = $1",
                    service_row["id"],
                )
                return {
                    "id": str(service_row["id"]),
                    "org_id": str(service_row["org_id"]),
                    "email": None,
                    "canonical_name": str(service_row["name"]),
                    "role": "service",
                    "status": "active",
                    "entity_id": None,
                    "session_id": None,
                    "client_type": "agent",
                    "expires_at": service_row["expires_at"],
                    "principal_type": "service",
                    "auth_scopes": list(service_row["scopes"] or []),
                    "resolved_org_ctx": org_ctx,
                }
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token. Use Bearer <API_KEY>.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        result = dict(row)
        result["principal_type"] = "employee"
        result["auth_scopes"] = []
        result["resolved_org_ctx"] = org_ctx
        return result


McpCaller = Annotated[dict[str, Any], Depends(get_mcp_caller)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    _configure_app_logging()
    if not DEFAULT_ORG_ID:
        raise RuntimeError("ORGANISATION_ID environment variable is required")

    database_url = os.environ.get("DATABASE_URL", "")
    if database_url:
        await migrate_all_databases(database_url)

    default_neo4j = Neo4jClient()
    default_neo4j.verify()
    org_neo4j_registry.set_default(DEFAULT_ORG_ID, default_neo4j)
    org_pool_registry.set_default_org(DEFAULT_ORG_ID)

    try:
        await db_pool.open()
        await _ensure_default_org_env_row()
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("Postgres pool not started: %s", exc)

    from base.tools.s3 import ensure_bucket_async

    for attempt in range(30):
        try:
            await ensure_bucket_async()
            break
        except Exception as exc:
            _log.warning("S3 bucket not ready (%s); retry %s/30", exc, attempt + 1)
            await asyncio.sleep(2)
    else:
        _log.warning("S3 bucket ensure failed; document and trace ingest may fail")

    _log.info(
        "Context Graph MCP ready at http://0.0.0.0:8080/mcp/sse "
        "(Authorization: Bearer <API_KEY>)"
    )
    yield

    default_neo4j.close()
    org_neo4j_registry.close_all()
    await org_pool_registry.close_all()
    await db_pool.close()


app = FastAPI(title="Context Graph", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def strip_trailing_slash(request: Request, call_next):
    if request.url.path.startswith("/mcp/") and request.url.path.endswith("/"):
        path = request.url.path.rstrip("/")
        scope = dict(request.scope)
        scope["path"] = path
        request = Request(scope, request.receive, request._send)
    return await call_next(request)


@app.middleware("http")
async def log_request_duration(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        client = request.client
        client_addr = f"{client.host}:{client.port}" if client else "-"
        http_version = request.scope.get("http_version", "1.1")
        _access_log.info(
            '%s - "%s %s HTTP/%s" %d %s (%.1fms)',
            client_addr,
            request.method,
            path,
            http_version,
            status_code,
            _http_status_phrase(status_code),
            elapsed_ms,
        )


@app.exception_handler(HTTPException)
async def mcp_http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/mcp/") and exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32001,
                    "message": exc.detail,
                },
            },
            headers=dict(exc.headers or {}),
        )
    return await http_exception_handler(request, exc)


DOCUMENT_CONTENT_MAX_BYTES = 2 * 1024 * 1024
DocumentType = Literal[
    "architecture",
    "hld",
    "lld",
    "design",
    "roadmap",
    "ticket_retrospective",
    "context",
]


class DocumentIngestPayload(BaseModel):
    repo: str | None = None
    doc_type: DocumentType = "context"
    title: str
    content: str
    source_ref: str | None = None
    idempotency_key: str | None = None

    @field_validator("repo")
    @classmethod
    def validate_repo(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        repository, _ = normalize_repo(value)
        return repository

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("title must not be empty")
        if len(normalized) > 300:
            raise ValueError("title must be at most 300 characters")
        return normalized

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be empty")
        if len(normalized.encode("utf-8")) > DOCUMENT_CONTENT_MAX_BYTES:
            raise ValueError(
                f"content must be at most {DOCUMENT_CONTENT_MAX_BYTES} UTF-8 bytes"
            )
        return normalized

    @field_validator("source_ref", "idempotency_key")
    @classmethod
    def validate_stable_ref(cls, value: str | None) -> str | None:
        return normalize_source_ref(value) if value is not None else None


def _require_document_write(principal: dict[str, Any]) -> None:
    if principal.get("principal_type") == "employee":
        return
    scopes = set(principal.get("auth_scopes") or [])
    if principal.get("principal_type") == "service" and (
        "*" in scopes or "mcp:write" in scopes
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Document ingest requires an employee session or mcp:write service token.",
    )


def _require_document_read(principal: dict[str, Any]) -> None:
    if principal.get("principal_type") == "employee":
        return
    scopes = set(principal.get("auth_scopes") or [])
    if principal.get("principal_type") == "service" and (
        "*" in scopes or scopes.intersection({"mcp:read", "mcp:write"})
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Document reads require mcp:read or mcp:write.",
    )


def _document_status(processing_state: str) -> str:
    return document_status(processing_state)


@app.post("/ingest/documents", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    payload: DocumentIngestPayload,
    principal: CurrentPrincipal,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    _require_document_write(principal)
    if idempotency_key is not None:
        try:
            idempotency_key = normalize_source_ref(idempotency_key)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
    org_ctx: OrgCtx = principal["resolved_org_ctx"]
    try:
        return await persist_portal_document(
            pool=org_ctx.pool,
            org_id=str(principal.get("org_id") or org_ctx.org_id),
            employee_id=(
                str(principal["id"])
                if principal.get("principal_type") == "employee"
                else None
            ),
            repo=payload.repo,
            doc_type=payload.doc_type,
            title=payload.title,
            content=payload.content,
            source_ref=(
                payload.source_ref
                or payload.idempotency_key
                or idempotency_key
            ),
        )
    except DocumentConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="source_ref is already used by another document",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _log.exception("Document ingest failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is temporarily unavailable.",
        ) from exc


@app.get("/documents")
@app.get("/ingest/documents")
async def list_ingested_documents(
    principal: CurrentPrincipal,
    repo: str | None = Query(default=None),
) -> dict[str, Any]:
    _require_document_read(principal)
    repository = None
    if repo is not None:
        try:
            repository, _ = normalize_repo(repo)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
    org_ctx: OrgCtx = principal["resolved_org_ctx"]
    rows = await org_ctx.pool.fetch(
        """
        SELECT id, repo, doc_type, title, source_ref, processing_state,
               current_version, extraction_error, created_at, updated_at
        FROM documents d
        WHERE d.org_id = $1
          AND d.source_provider = 'portal'
          AND ($2::text IS NULL OR d.repo IS NULL OR d.repo = $2)
        ORDER BY d.updated_at DESC, d.id
        LIMIT 200
        """,
        str(principal.get("org_id") or org_ctx.org_id),
        repository,
    )
    documents = []
    for row in rows:
        item = dict(row)
        item["version"] = item.pop("current_version")
        item["status"] = _document_status(str(item["processing_state"]))
        item["error"] = item.pop("extraction_error")
        documents.append(item)
    return {"documents": documents}


@app.post("/documents/{document_id}/retry")
@app.post("/ingest/documents/{document_id}/retry")
async def retry_ingested_document(
    document_id: str,
    principal: CurrentPrincipal,
) -> dict[str, Any]:
    _require_document_write(principal)
    org_ctx: OrgCtx = principal["resolved_org_ctx"]
    try:
        return await retry_portal_document(
            pool=org_ctx.pool,
            org_id=str(principal.get("org_id") or org_ctx.org_id),
            document_id=document_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        ) from exc
    except DocumentNotRetryableError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        _log.exception("Document retry failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is temporarily unavailable.",
        ) from exc


class IngestPayload(BaseModel):
    project: str | None = None
    entries: list[dict]


@app.post("/ingest")
async def ingest(
    payload: IngestPayload,
    employee: CurrentEmployee,
) -> JSONResponse:
    org_ctx: OrgCtx = employee["resolved_org_ctx"]
    org_id = employee.get("org_id") or org_ctx.org_id
    project = org_id.removeprefix("org:") if org_id else (payload.project or "")
    await _embed_nodes(payload.entries)
    result = await asyncio.to_thread(
        ingest_entries, org_ctx.neo4j, project, payload.entries,
        employee.get("canonical_name"),
    )
    return JSONResponse({"ok": True, **result})


class TraceMention(BaseModel):
    mention_text: str
    mention_type: Literal[
        "person",
        "service",
        "org",
        "repo",
        "tool",
        "object",
        "location",
        "event",
        "role_reference",
        "pronoun",
    ]


class TraceEntry(BaseModel):
    trace_id: str
    ts: str
    session_id: str | None = None
    project: str | None = None
    repo: str | None = None
    node: dict
    keywords: list[str] = []
    mentions: list[TraceMention] = []

    @field_validator("repo")
    @classmethod
    def validate_optional_repo(cls, value: str | None) -> str | None:
        if value is None:
            return None
        repository, _ = normalize_repo(value)
        return repository


class TracePayload(BaseModel):
    project: str | None = None
    traces: list[TraceEntry]


@app.post("/ingest/traces")
async def ingest_traces(
    payload: TracePayload,
    principal: CurrentPrincipal,
) -> JSONResponse:
    _require_document_write(principal)
    org_ctx: OrgCtx = principal["resolved_org_ctx"]
    org_id = principal.get("org_id") or org_ctx.org_id
    project = org_id.removeprefix("org:") if org_id else (payload.project or "")

    author_name: str = principal.get("canonical_name") or ""
    author_employee_id: str = str(principal.get("id") or "")

    traces_dicts = normalize_trace_entries(
        [trace_entry.model_dump() for trace_entry in payload.traces],
        project=project,
        org_id=org_id,
        author_name=author_name,
        author_employee_id=author_employee_id,
    )

    employee_id: str = str(principal.get("id") or "unknown")
    try:
        result = await store_traces_s3(
            project=project,
            org_id=org_id,
            traces=traces_dicts,
            employee_id=employee_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    if result["failed"] > 0:
        return JSONResponse(
            status_code=207,
            content={
                "ok": False,
                "accepted": result["stored"],
                "failed": result["failed"],
                "detail": f"{result['failed']} trace(s) failed to store",
            },
        )

    return JSONResponse({"ok": True, "accepted": result["stored"]})


@app.get("/mcp/sse")
async def mcp_sse(employee: McpCaller) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        yield "event: endpoint\ndata: /mcp/message\n\n"
        while True:
            await asyncio.sleep(15)
            yield ": ping\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/mcp/sse")
async def mcp_sse_post(
    request: Request,
    employee: McpCaller,
) -> JSONResponse:
    body = await request.body()
    return await _handle_mcp(body, employee)


@app.post("/mcp/message")
async def mcp_message(
    request: Request,
    employee: McpCaller,
) -> JSONResponse:
    body = await request.body()
    return await _handle_mcp(body, employee)


async def _handle_mcp(body: bytes, employee: dict) -> JSONResponse:
    try:
        req = json.loads(body)
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    method = req.get("method", "")
    req_id = req.get("id")
    org_ctx: OrgCtx = employee["resolved_org_ctx"]

    try:
        handler_ctx = McpContext(
            client=org_ctx.neo4j,
            employee=employee,
            db_pool=org_ctx.pool,
            org_id=org_ctx.org_id,
            inference_client_getter=_get_inference_client,
        )
        handler = AgentMcpHandler(handler_ctx)
        result = await handler.dispatch(method, req.get("params", {}))
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": result})
    except ResourceNotFound as exc:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32002, "message": str(exc), "data": {"uri": exc.uri}},
        })
    except Exception as exc:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": str(exc)},
        })


async def _get_api_org_ctx(
    authorization: Annotated[str | None, Header()] = None,
    x_eidetix_org: Annotated[str | None, Header(alias="X-Eidetix-Org")] = None,
    central_db: DbConn = Depends(get_db),
) -> OrgCtx:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    if not hmac.compare_digest(authorization[7:], _API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return await resolve_org_from_header(central_db, x_eidetix_org, DEFAULT_ORG_ID)


ApiOrgCtx = Annotated[OrgCtx, Depends(_get_api_org_ctx)]


@app.get("/api/projects", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP tools")
async def api_projects(org_ctx: ApiOrgCtx):
    return await asyncio.to_thread(q.list_projects, org_ctx.neo4j)


@app.get("/api/topics", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP tools")
async def api_topics(org_ctx: ApiOrgCtx):
    return await asyncio.to_thread(q.list_topics, org_ctx.neo4j)


@app.get("/api/search", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP search or recall")
async def api_search(
    org_ctx: ApiOrgCtx,
    q_param: str = Query(alias="q"),
):
    return await asyncio.to_thread(q.search_memory, org_ctx.neo4j, q_param)


@app.get("/api/entities/search", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP resolve_entities")
async def api_entities_search(
    org_ctx: ApiOrgCtx,
    q_param: str = Query(alias="q"),
):
    return await search_entities_pg(org_ctx.pool, org_ctx.org_id, q_param)


@app.get("/api/entities/by-channel", deprecated=True)
@deprecated_api(reason="legacy REST endpoint; no active callers")
async def api_entities_by_channel(
    org_ctx: ApiOrgCtx,
    channel_id: str = Query(),
):
    rows = await org_ctx.pool.fetch(
        """
        SELECT
            cm.employee_id,
            cm.external_user_id,
            cm.source_provider,
            er.id            AS entity_id,
            er.canonical_name,
            COALESCE(
                (
                    SELECT json_agg(ea.alias_raw ORDER BY ea.created_at)
                    FROM entity_aliases ea
                    WHERE ea.entity_id = er.id
                      AND ea.org_id    = er.org_id
                ),
                '[]'::json
            ) AS aliases
        FROM channel_members cm
        JOIN employees emp ON emp.id = cm.employee_id
        LEFT JOIN entity_registry er
               ON er.employee_id = cm.employee_id
              AND er.org_id      = emp.org_id
        WHERE cm.channel_id = $1
          AND emp.org_id    = $2
        """,
        channel_id,
        org_ctx.org_id,
    )
    return {"entities": [dict(r) for r in rows]}


@app.get("/api/traverse", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP get_graph")
async def api_traverse(
    org_ctx: ApiOrgCtx,
    node_id: str = Query(),
    rel: str | None = Query(default=None),
    direction: str = Query(default="both"),
    limit: int = Query(default=20, ge=1, le=50),
):
    return await asyncio.to_thread(
        q.traverse,
        client=org_ctx.neo4j,
        node_id=node_id,
        rel=rel,
        direction=direction,
        limit=limit,
    )


@app.get("/api/node", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP get_graph_expanded")
async def api_node(
    org_ctx: ApiOrgCtx,
    node_id: str = Query(),
):
    result = await asyncio.to_thread(q.get_node, org_ctx.neo4j, node_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return result


@app.get("/api/context", deprecated=True)
@deprecated_api(reason="legacy REST endpoint", replacement="MCP search or recall")
async def api_context(
    org_ctx: ApiOrgCtx,
    query: str = Query(default=""),
    limit: int = Query(default=8, ge=1, le=20),
):
    if not query:
        return {"nodes": [], "total_matched": 0, "returned": 0}
    embedding = (await _embed_small([query]))[0]
    return await asyncio.to_thread(
        q.query_context_vector,
        client=org_ctx.neo4j,
        query_embedding=embedding,
        limit=limit,
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True}
