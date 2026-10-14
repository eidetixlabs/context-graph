"""
auth/routes.py — Login and logout endpoints.

POST /auth/login   — validate email + password, issue bearer token
POST /auth/logout  — revoke the current session

Per-org routing
---------------
- login: resolves org via X-Eidetix-Org header if present, otherwise uses the
  default org for this webserver.  Dual-writes token_org to central after the
  session is created so subsequent requests can route via bearer token.
- logout: revokes the session in the env pool AND marks token_org revoked in central.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

logger = logging.getLogger(__name__)
from pydantic import BaseModel, EmailStr

from base.auth.deps import AdminEmployee, CurrentEmployee
from base.auth.password import verify_password, needs_rehash, hash_password
from base.db.client import DbConn, get_db
from base.routing import (
    OrgCtx,
    dual_write_token_org,
    revoke_token_org,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Desktop-app sessions last 30 days.
_DESKTOP_TOKEN_TTL = timedelta(days=30)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

_VALID_CLIENT_TYPES = {"desktop-app", "cursor", "claude-desktop", "api", "agent"}
_AGENT_TOKEN_EXPIRES_AT = datetime(9999, 12, 31, tzinfo=timezone.utc)
_SERVICE_TOKEN_PREFIX = "est_"
_SERVICE_TOKEN_DEFAULT_TTL = timedelta(days=90)
_SERVICE_TOKEN_MAX_TTL = timedelta(days=365)
_SERVICE_TOKEN_SCOPE_ALLOWLIST = {
    "mcp:read",
    "mcp:write",
    "workflow:create",
    "workflow:read",
    "workflow:approve",
}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_info: str | None = None
    # Identifies the calling MCP client; recorded on the session row.
    # Defaults to "desktop-app" for backwards compatibility.
    client_type: str = "desktop-app"


class EmployeeInfo(BaseModel):
    id: str
    email: str
    name: str
    role: str


class LoginResponse(BaseModel):
    token: str
    expires_at: datetime
    employee: EmployeeInfo


class ServiceTokenRequest(BaseModel):
    name: str
    scopes: list[str]
    expires_in_days: int = 90


class ServiceTokenRotateRequest(BaseModel):
    expires_in_days: int = 90


class ServiceTokenInfo(BaseModel):
    id: str
    name: str
    scopes: list[str]
    expires_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime


class ServiceTokenIssueResponse(BaseModel):
    token: str
    service_token: ServiceTokenInfo


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_org_id() -> str:
    raw = os.environ.get("ORGANISATION_ID", "").strip()
    return raw if raw.startswith("org:") else f"org:{raw}" if raw else ""


def _normalize_scopes(scopes: list[str]) -> list[str]:
    normalized = sorted({scope.strip().lower() for scope in scopes if scope and scope.strip()})
    invalid = [scope for scope in normalized if scope not in _SERVICE_TOKEN_SCOPE_ALLOWLIST]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scope(s): {', '.join(invalid)}",
        )
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one scope is required.",
        )
    return normalized


def _service_token_expiry(expires_in_days: int) -> datetime:
    if expires_in_days <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="expires_in_days must be > 0")
    requested = timedelta(days=expires_in_days)
    ttl = requested if requested <= _SERVICE_TOKEN_MAX_TTL else _SERVICE_TOKEN_MAX_TTL
    return datetime.now(timezone.utc) + (ttl or _SERVICE_TOKEN_DEFAULT_TTL)


async def _issue_service_token(
    env_conn,
    *,
    org_id: str,
    name: str,
    scopes: list[str],
    created_by_employee_id: str,
    expires_at: datetime,
) -> tuple[dict, str]:
    token_id = uuid.uuid4()
    raw_token = f"{_SERVICE_TOKEN_PREFIX}{secrets.token_urlsafe(36)}"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    row = await env_conn.fetchrow(
        """
        INSERT INTO service_tokens (
            id,
            org_id,
            name,
            token_hash,
            scopes,
            created_by_employee_id,
            expires_at
        )
        VALUES ($1, $2, $3, $4, $5::text[], $6, $7)
        RETURNING id, name, scopes, expires_at, revoked_at, last_used_at, created_at
        """,
        token_id,
        org_id,
        name,
        token_hash,
        scopes,
        created_by_employee_id,
        expires_at,
    )
    if row is None:
        raise RuntimeError("Failed to issue service token")
    return dict(row) | {"token_hash": token_hash}, raw_token


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Email + password login",
    description=(
        "Authenticate with email and password. "
        "Returns a bearer token valid for 30 days. "
        "Include the token in subsequent requests as: Authorization: Bearer <token>. "
        "Optional X-Eidetix-Org header (plain slug or 'org:<slug>') routes the lookup "
        "to a specific org's env postgres. Absent → uses the server's default org."
    ),
)
async def login(
    body: LoginRequest,
    request: Request,
    central_db: DbConn = Depends(get_db),
    x_eidetix_org: str | None = Header(None, alias="X-Eidetix-Org"),
) -> LoginResponse:
    logger.info(
        "login attempt email=%s client_type=%s org_header=%s",
        body.email.lower(), body.client_type, x_eidetix_org,
    )

    # ── Resolve org ────────────────────────────────────────────────────────
    # When X-Eidetix-Org is present, route to that org's env postgres.
    # Absent → fall back to the env this webserver is deployed for (backward
    # compatible: single-org deployments never set the header).
    from base.routing import (  # local import to avoid circular
        resolve_org_from_header,
        resolve_org_id_from_email_domain,
        resolve_org_ctx,
    )
    default_org = _default_org_id()
    if not default_org:
        logger.error("login rejected — ORGANISATION_ID is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server organization is not configured.",
        )

    if x_eidetix_org:
        org_ctx = await resolve_org_from_header(central_db, x_eidetix_org, default_org)
    else:
        domain_org = await resolve_org_id_from_email_domain(central_db, body.email)
        org_ctx = await resolve_org_ctx(central_db, domain_org or default_org, default_org)

    async with org_ctx.pool.acquire() as env_conn:
        # ── 1. Look up employee by email in the resolved org ───────────────
        emp = await env_conn.fetchrow(
            """
            SELECT id, org_id, email, canonical_name, role, status, auth_provider
            FROM employees
            WHERE email = $1 AND org_id = $2
            """,
            body.email.lower(),
            org_ctx.org_id,
        )

        if emp is None or emp["status"] != "active":
            logger.warning("login rejected — employee not found or inactive email=%s", body.email.lower())
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        from base.db.identity import is_identity_stub

        if is_identity_stub(
            employee_id=emp["id"],
            email=emp.get("email"),
            auth_provider=emp.get("auth_provider"),
        ):
            logger.warning("login rejected — connector identity stub email=%s", body.email.lower())
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        logger.debug("step 1 complete — employee found id=%s status=%s", emp["id"], emp["status"])

        # ── 2. Only local-password accounts are handled here ──────────────
        if emp["auth_provider"] != "local":
            logger.warning(
                "login rejected — non-local auth_provider=%s employee_id=%s",
                emp["auth_provider"], emp["id"],
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"This account uses '{emp['auth_provider']}' authentication. "
                    "Password login is not available."
                ),
            )
        logger.debug("step 2 complete — auth_provider=local employee_id=%s", emp["id"])

        # ── 3. Fetch password hash ─────────────────────────────────────────
        cred = await env_conn.fetchrow(
            "SELECT password_hash FROM local_credentials WHERE employee_id = $1",
            emp["id"],
        )

        if cred is None:
            logger.warning("login rejected — no local_credentials row employee_id=%s", emp["id"])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        logger.debug("step 3 complete — credential row fetched employee_id=%s", emp["id"])

        # ── 4. Verify password ─────────────────────────────────────────────
        if not verify_password(body.password, cred["password_hash"]):
            logger.warning("login rejected — password mismatch employee_id=%s", emp["id"])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        logger.debug("step 4 complete — password verified employee_id=%s", emp["id"])

        # ── 5. Opportunistic rehash ────────────────────────────────────────
        if needs_rehash(cred["password_hash"]):
            logger.info("step 5 — rehashing password for employee_id=%s", emp["id"])
            new_hash = hash_password(body.password)
            await env_conn.execute(
                "UPDATE local_credentials SET password_hash = $1 WHERE employee_id = $2",
                new_hash,
                emp["id"],
            )
            logger.debug("step 5 complete — password rehashed employee_id=%s", emp["id"])

        # ── 6. Issue bearer token ──────────────────────────────────────────
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        client_ip = request.client.host if request.client else None
        client_type = body.client_type if body.client_type in _VALID_CLIENT_TYPES else "desktop-app"
        if client_type == "agent":
            expires_at = _AGENT_TOKEN_EXPIRES_AT
        else:
            expires_at = datetime.now(timezone.utc) + _DESKTOP_TOKEN_TTL

        await env_conn.execute(
            """
            INSERT INTO sessions
                (employee_id, token_hash, client_type, device_info, ip_address, expires_at)
            VALUES ($1, $2, $3, $4, $5::inet, $6)
            """,
            emp["id"],
            token_hash,
            client_type,
            body.device_info,
            client_ip,
            expires_at,
        )
        logger.info(
            "step 6 complete — session created employee_id=%s client_type=%s ip=%s expires_at=%s",
            emp["id"], client_type, client_ip, expires_at,
        )

        # ── 7. Update last_login_at ────────────────────────────────────────
        await env_conn.execute(
            "UPDATE employees SET last_login_at = now() WHERE id = $1",
            emp["id"],
        )
        logger.debug("step 7 complete — last_login_at updated employee_id=%s", emp["id"])

    # ── 8. Dual-write token_org to central ────────────────────────────────
    await dual_write_token_org(central_db, token_hash, org_ctx.org_id, expires_at)
    logger.info("login success employee_id=%s email=%s org=%s", emp["id"], emp["email"], org_ctx.org_id)

    return LoginResponse(
        token=raw_token,
        expires_at=expires_at,
        employee=EmployeeInfo(
            id=emp["id"],
            email=emp["email"],
            name=emp["canonical_name"],
            role=emp["role"],
        ),
    )


# ---------------------------------------------------------------------------
# POST /auth/logout
# ---------------------------------------------------------------------------

@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke the current session",
)
async def logout(
    request: Request,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> None:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    raw_token = auth_header[len("Bearer "):]
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

    # ── Revoke in the org's env Postgres ──────────────────────────────────
    org_ctx: OrgCtx = employee.get("resolved_org_ctx")
    if org_ctx is not None:
        async with org_ctx.pool.acquire() as env_conn:
            await env_conn.execute(
                "UPDATE sessions SET revoked_at = now() WHERE token_hash = $1 AND revoked_at IS NULL",
                token_hash,
            )
    else:
        # Fallback: use the central pool (default org path)
        await central_db.execute(
            "UPDATE sessions SET revoked_at = now() WHERE token_hash = $1 AND revoked_at IS NULL",
            token_hash,
        )

    # ── Mark revoked in central token_org ────────────────────────────────
    await revoke_token_org(central_db, token_hash)


@router.get("/service-tokens", response_model=list[ServiceTokenInfo], summary="List service tokens")
async def list_service_tokens(employee: AdminEmployee) -> list[ServiceTokenInfo]:
    org_ctx: OrgCtx = employee["resolved_org_ctx"]
    async with org_ctx.pool.acquire() as env_conn:
        rows = await env_conn.fetch(
            """
            SELECT id, name, scopes, expires_at, revoked_at, last_used_at, created_at
            FROM service_tokens
            WHERE org_id = $1
            ORDER BY created_at DESC
            """,
            employee["org_id"],
        )
    return [ServiceTokenInfo(**dict(row)) for row in rows]


@router.post(
    "/service-tokens",
    response_model=ServiceTokenIssueResponse,
    summary="Issue a scoped service token",
)
async def create_service_token(
    body: ServiceTokenRequest,
    employee: AdminEmployee,
    central_db: DbConn = Depends(get_db),
) -> ServiceTokenIssueResponse:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token name is required.")
    scopes = _normalize_scopes(body.scopes)
    expires_at = _service_token_expiry(body.expires_in_days)

    org_ctx: OrgCtx = employee["resolved_org_ctx"]
    async with org_ctx.pool.acquire() as env_conn:
        token_row, raw_token = await _issue_service_token(
            env_conn,
            org_id=employee["org_id"],
            name=name,
            scopes=scopes,
            created_by_employee_id=employee["id"],
            expires_at=expires_at,
        )

    await dual_write_token_org(
        central_db,
        token_row["token_hash"],
        employee["org_id"],
        token_row["expires_at"],
    )
    return ServiceTokenIssueResponse(
        token=raw_token,
        service_token=ServiceTokenInfo(**{k: token_row[k] for k in ServiceTokenInfo.model_fields}),
    )


@router.post(
    "/service-tokens/{token_id}/rotate",
    response_model=ServiceTokenIssueResponse,
    summary="Rotate an existing service token",
)
async def rotate_service_token(
    token_id: str,
    body: ServiceTokenRotateRequest,
    employee: AdminEmployee,
    central_db: DbConn = Depends(get_db),
) -> ServiceTokenIssueResponse:
    expires_at = _service_token_expiry(body.expires_in_days)
    org_ctx: OrgCtx = employee["resolved_org_ctx"]
    async with org_ctx.pool.acquire() as env_conn:
        existing = await env_conn.fetchrow(
            """
            SELECT id, name, scopes, token_hash, revoked_at
            FROM service_tokens
            WHERE id = $1::uuid AND org_id = $2
            """,
            token_id,
            employee["org_id"],
        )
        if existing is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Service token not found.")
        if existing["revoked_at"] is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Service token already revoked.")

        await env_conn.execute(
            "UPDATE service_tokens SET revoked_at = now() WHERE id = $1::uuid",
            token_id,
        )
        token_row, raw_token = await _issue_service_token(
            env_conn,
            org_id=employee["org_id"],
            name=str(existing["name"]),
            scopes=list(existing["scopes"] or []),
            created_by_employee_id=employee["id"],
            expires_at=expires_at,
        )

    await revoke_token_org(central_db, existing["token_hash"])
    await dual_write_token_org(
        central_db,
        token_row["token_hash"],
        employee["org_id"],
        token_row["expires_at"],
    )
    return ServiceTokenIssueResponse(
        token=raw_token,
        service_token=ServiceTokenInfo(**{k: token_row[k] for k in ServiceTokenInfo.model_fields}),
    )


@router.post(
    "/service-tokens/{token_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a service token",
)
async def revoke_service_token(
    token_id: str,
    employee: AdminEmployee,
    central_db: DbConn = Depends(get_db),
) -> None:
    org_ctx: OrgCtx = employee["resolved_org_ctx"]
    async with org_ctx.pool.acquire() as env_conn:
        row = await env_conn.fetchrow(
            """
            SELECT token_hash, revoked_at
            FROM service_tokens
            WHERE id = $1::uuid AND org_id = $2
            """,
            token_id,
            employee["org_id"],
        )
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Service token not found.")
        if row["revoked_at"] is not None:
            return
        await env_conn.execute(
            "UPDATE service_tokens SET revoked_at = now() WHERE id = $1::uuid",
            token_id,
        )

    await revoke_token_org(central_db, row["token_hash"])
