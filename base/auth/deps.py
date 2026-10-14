"""
auth/deps.py — FastAPI dependency for session-token authentication.

Usage in a route::

    from base.auth.deps import CurrentEmployee

    async def my_route(employee: CurrentEmployee):
        return {"hello": employee["canonical_name"]}

Per-org routing
---------------
get_current_employee now:
  1. Resolves which env the token belongs to via token_org (central).
  2. Validates the session in THAT env's Postgres pool.
The employee dict gains an extra key "resolved_org_ctx" (OrgCtx) so routes
can access the already-resolved pool + neo4j without a second lookup.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, status

from base.db.client import DbConn, get_db
from base.routing import OrgCtx, resolve_org_from_header, resolve_org_from_token


def _default_org_id() -> str:
    raw = os.environ.get("ORGANISATION_ID", "").strip()
    return raw if raw.startswith("org:") else f"org:{raw}" if raw else ""


# SQL used in both deps — query text is unchanged; only the pool changes.
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
    SELECT
        id,
        org_id,
        name,
        scopes,
        expires_at
    FROM service_tokens
    WHERE token_hash = $1
      AND revoked_at IS NULL
      AND expires_at > now()
"""


def _extract_bearer_token(authorization: str | None) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header. Use Bearer <API_KEY>.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[len("Bearer "):].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def get_current_principal(
    authorization: Annotated[str | None, Header()] = None,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    """
    Validate either an employee session token or a scoped service token.
    """
    raw_token = _extract_bearer_token(authorization)
    api_key = os.environ.get("API_KEY", "")
    default_org = _default_org_id()
    if api_key and hmac.compare_digest(raw_token, api_key):
        org_ctx = await resolve_org_from_header(central_db, None, default_org)
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

    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    org_ctx: OrgCtx = await resolve_org_from_token(central_db, raw_token, default_org)

    async with org_ctx.pool.acquire() as env_conn:
        session_row = await env_conn.fetchrow(_SESSION_QUERY, token_hash)
        if session_row is not None:
            result = dict(session_row)
            result["principal_type"] = "employee"
            result["resolved_org_ctx"] = org_ctx
            result["auth_scopes"] = []
            return result

        service_row = await env_conn.fetchrow(_SERVICE_TOKEN_QUERY, token_hash)
        if service_row is not None:
            await env_conn.execute(
                "UPDATE service_tokens SET last_used_at = now() WHERE id = $1",
                service_row["id"],
            )
            result = dict(service_row)
            result["principal_type"] = "service"
            result["resolved_org_ctx"] = org_ctx
            result["auth_scopes"] = list(service_row["scopes"] or [])
            return result

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Use Bearer <API_KEY>.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_employee(
    authorization: Annotated[str | None, Header()] = None,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    """
    Validate a session bearer token and return the employee row.

    Token lookup:
      1. Extract raw token from  Authorization: Bearer <token>
      2. Compute SHA-256 hex digest
      3. Resolve org from central token_org (miss -> default org)
      4. SELECT sessions + employees in the org's env Postgres
      5. Return employee dict (with "resolved_org_ctx" key) or raise 401
    """
    principal = await get_current_principal(authorization=authorization, central_db=central_db)
    if principal.get("principal_type") != "employee":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Employee session required for this endpoint.",
        )
    return principal


# Convenient type alias for route signatures.
CurrentEmployee = Annotated[dict[str, Any], Depends(get_current_employee)]
CurrentPrincipal = Annotated[dict[str, Any], Depends(get_current_principal)]


def require_admin_employee(employee: CurrentEmployee) -> dict[str, Any]:
    if employee.get("role") not in {"owner", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return employee


AdminEmployee = Annotated[dict[str, Any], Depends(require_admin_employee)]
