"""
auth/google.py — Google OAuth 2.0 browser-based login flow.

Flow:
  1. Desktop → POST /auth/google/start
         Server creates an oauth_attempts row (state, expiry).
         Returns {attempt_id, browser_url}.

  2. Desktop opens browser_url:
         https://app.eidetix.net/login/google?attempt_id=<id>

  3. GET /login/google?attempt_id=<id>
         Looks up attempt, builds Google authorization URL with stored state.
         Returns HTTP 302 → Google OAuth consent screen.

  4. Google → GET /oauth/callback?code=<code>&state=<state>
         Server verifies state, exchanges code for tokens, verifies hd claim,
         finds or provisions the employee, creates a session, stores a
         short-lived bootstrap_token in the attempt row.
         Returns a success HTML page.

  5. Desktop polls GET /auth/google/status?attempt_id=<id>
         Returns {status, bootstrap_token} when completed.

  6. Desktop → POST /auth/google/exchange  {bootstrap_token: "<raw>"}
         Server validates bootstrap_token (one-time, 5 min TTL), issues a
         full 30-day bearer session, returns {token, expires_at, employee}.

Security notes:
  • state is verified on callback — CSRF protection (Google recommends this).
  • hd claim verified server-side — restricts to Google Workspace domains.
  • Google `sub` used as primary identity — stable across email changes.
  • bootstrap_token is single-use; cleared from DB after exchange.
  • Client credentials never leave the server.

Gmail connector consent (purpose='gmail')
------------------------------------------
A second, parallel flow reuses steps 2-4 above (same /login/google and
/oauth/callback routes) but is initiated only by an already-authenticated
employee, and never issues a new session:

  1. Desktop → POST /auth/gmail/connect/start   (Authorization: Bearer <session>)
         Server creates an oauth_attempts row with purpose='gmail' and
         employee_id/org_id already populated (the employee is known).
         Returns {attempt_id, browser_url}.

  2. Desktop opens browser_url (same /login/google route). Because the
     attempt's purpose is 'gmail', the Google consent screen requests
     `gmail.readonly` with access_type=offline&prompt=consent so a refresh
     token is always issued.

  3. Google → GET /oauth/callback?code=<code>&state=<state>
         Because the attempt's purpose is 'gmail', the callback exchanges
         the code, encrypts the refresh/access tokens, and upserts a
         gmail_connections row for that employee. No session is created.

  4. Desktop polls GET /auth/gmail/connect/status?attempt_id=<id>
         (Authorization: Bearer <session>, same employee as the attempt.)
         Returns {status, gmail_address, error}.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from base.auth.deps import CurrentEmployee
from base.crypto import CredentialEncryptionError, encrypt_secret
from base.db.client import DbConn, get_db
from base.routing import OrgCtx, dual_write_token_org

router = APIRouter(tags=["auth"])


def _default_org_id() -> str:
    raw = os.environ.get("ORGANISATION_ID", "").strip()
    return raw if raw.startswith("org:") else f"org:{raw}" if raw else ""

# ---------------------------------------------------------------------------
# Config — injected via environment (populated from SSM on EC2)
# ---------------------------------------------------------------------------

_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
_APP_DOMAIN = os.environ.get("APP_DOMAIN", "app.eidetix.net")

_ATTEMPT_TTL = timedelta(minutes=10)
_BOOTSTRAP_TTL = timedelta(minutes=5)
_SESSION_TTL = timedelta(days=30)

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"

# Gmail connector consent — always requests offline access + forces the
# consent screen so a refresh token is issued on every connect/reconnect.
_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _public_base() -> str:
    """Public origin for OAuth redirect + browser URLs.

    APP_DOMAIN may be:
      - host only:           app.eidetix.net  → https://app.eidetix.net
      - host:port:           localhost:8080   → http://localhost:8080
      - full origin:         http://localhost:8080 (used as-is)

    Localhost / 127.0.0.1 default to http so docker/dev works without TLS.
    """
    raw = (_APP_DOMAIN or "").strip().rstrip("/")
    if not raw:
        return "https://app.eidetix.net"
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    host = raw.split("/", 1)[0]
    hostname = host.split(":", 1)[0].lower()
    scheme = "http" if hostname in ("localhost", "127.0.0.1") else "https"
    return f"{scheme}://{raw}"


def _redirect_uri() -> str:
    return f"{_public_base()}/oauth/callback"


def _is_configured() -> bool:
    return bool(_CLIENT_ID and _CLIENT_SECRET)


async def _resolve_org_for_google_hd(central_db: DbConn, hd_domain: str) -> str | None:
    """
    Resolve org_id for Google Workspace domain from central org_env.

    Resolution order:
      1) Exact email_domain match (preferred and explicit).
      2) Legacy slug fallback from first domain label.
    """
    domain = hd_domain.strip().lower()
    if not domain:
        return None

    row = await central_db.fetchrow(
        "SELECT org_id FROM org_env WHERE email_domain = $1 AND status = 'active'",
        domain,
    )
    if row:
        return row["org_id"]

    legacy_slug = domain.split(".", 1)[0]
    row = await central_db.fetchrow(
        "SELECT org_id FROM org_env WHERE slug = $1 AND status = 'active'",
        legacy_slug,
    )
    return row["org_id"] if row else None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartResponse(BaseModel):
    attempt_id: str
    browser_url: str


class StatusResponse(BaseModel):
    status: str                    # pending | completed | failed | expired
    bootstrap_token: str | None = None
    error: str | None = None


class ExchangeRequest(BaseModel):
    bootstrap_token: str


class EmployeeInfo(BaseModel):
    id: str
    email: str
    name: str
    role: str


class ExchangeResponse(BaseModel):
    token: str
    expires_at: datetime
    employee: EmployeeInfo


class GmailConnectStatusResponse(BaseModel):
    status: str                    # pending | completed | failed | expired
    gmail_address: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# POST /auth/google/start
# ---------------------------------------------------------------------------

@router.post(
    "/auth/google/start",
    response_model=StartResponse,
    summary="Begin Google OAuth flow",
    description=(
        "Creates a server-side OAuth attempt and returns the attempt_id and a "
        "browser_url that the desktop app should open in the system browser."
    ),
)
async def google_start(
    request: Request,
    db: DbConn = Depends(get_db),
) -> StartResponse:
    if not _is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )

    attempt_id = secrets.token_urlsafe(24)
    state = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + _ATTEMPT_TTL
    client_ip = request.client.host if request.client else None

    await db.execute(
        """
        INSERT INTO oauth_attempts (id, state, ip_address, expires_at)
        VALUES ($1, $2, $3::inet, $4)
        """,
        attempt_id,
        state,
        client_ip,
        expires_at,
    )

    browser_url = f"{_public_base()}/login/google?attempt_id={attempt_id}"
    return StartResponse(attempt_id=attempt_id, browser_url=browser_url)


# ---------------------------------------------------------------------------
# POST /auth/gmail/connect/start
# ---------------------------------------------------------------------------

@router.post(
    "/auth/gmail/connect/start",
    response_model=StartResponse,
    summary="Begin Gmail connector consent (authenticated employees only)",
    description=(
        "Creates a server-side OAuth attempt scoped to the calling employee "
        "and returns the attempt_id and a browser_url that the desktop app "
        "should open in the system browser. Unlike /auth/google/start, this "
        "requires a valid session and never issues a new one — it only "
        "records Gmail consent for the caller."
    ),
)
async def gmail_connect_start(
    employee: CurrentEmployee,
    db: DbConn = Depends(get_db),
) -> StartResponse:
    if not _is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured on this server.",
        )

    org_ctx: OrgCtx = employee["resolved_org_ctx"]

    attempt_id = secrets.token_urlsafe(24)
    state = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + _ATTEMPT_TTL

    await db.execute(
        """
        INSERT INTO oauth_attempts (id, state, purpose, employee_id, org_id, expires_at)
        VALUES ($1, $2, 'gmail', $3, $4, $5)
        """,
        attempt_id,
        state,
        employee["id"],
        org_ctx.org_id,
        expires_at,
    )

    browser_url = f"{_public_base()}/login/google?attempt_id={attempt_id}"
    return StartResponse(attempt_id=attempt_id, browser_url=browser_url)


# ---------------------------------------------------------------------------
# GET /login/google?attempt_id=<id>
# ---------------------------------------------------------------------------

@router.get(
    "/login/google",
    summary="Redirect browser to Google OAuth consent screen",
    response_class=RedirectResponse,
)
async def google_login(
    attempt_id: str = Query(...),
    db: DbConn = Depends(get_db),
) -> Response:
    if not _is_configured():
        return HTMLResponse(
            _error_page("Google OAuth is not configured on this server."),
            status_code=503,
        )

    attempt = await db.fetchrow(
        """
        SELECT state, purpose, expires_at FROM oauth_attempts
        WHERE id = $1 AND status = 'pending'
        """,
        attempt_id,
    )

    if attempt is None:
        return HTMLResponse(_error_page("Login attempt not found or already used."), status_code=400)

    if attempt["expires_at"] <= datetime.now(timezone.utc):
        await db.execute(
            "UPDATE oauth_attempts SET status = 'expired' WHERE id = $1",
            attempt_id,
        )
        return HTMLResponse(_error_page("Login attempt has expired. Please try again."), status_code=400)

    if attempt["purpose"] == "gmail":
        # Gmail connector consent — offline access, forced consent screen so
        # a refresh token is always returned (even on reconnect).
        params = {
            "client_id": _CLIENT_ID,
            "redirect_uri": _redirect_uri(),
            "response_type": "code",
            "scope": f"openid email {_GMAIL_SCOPE}",
            "state": attempt["state"],
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
        }
    else:
        params = {
            "client_id": _CLIENT_ID,
            "redirect_uri": _redirect_uri(),
            "response_type": "code",
            "scope": "openid email profile",
            "state": attempt["state"],
            # hd parameter hints to Google to show only Workspace accounts,
            # but we verify hd server-side regardless.
            "hd": "*",
            "access_type": "online",
            "prompt": "select_account",
        }

    google_url = f"{_GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=google_url, status_code=302)


# ---------------------------------------------------------------------------
# GET /oauth/callback?code=<code>&state=<state>
# ---------------------------------------------------------------------------

@router.get(
    "/oauth/callback",
    summary="Google OAuth callback — verifies state, exchanges code, creates session",
    response_class=HTMLResponse,
)
async def google_callback(
    request: Request,
    db: DbConn = Depends(get_db),
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> HTMLResponse:
    # ── Google reported an error (user denied consent, etc.) ───────────────
    if error:
        await _fail_attempt_by_state(db, state, f"Google error: {error}")
        return HTMLResponse(_error_page(f"Google sign-in was cancelled or denied: {error}"))

    if not code or not state:
        return HTMLResponse(_error_page("Missing code or state parameter."), status_code=400)

    # ── Look up attempt by state ────────────────────────────────────────────
    attempt = await db.fetchrow(
        """
        SELECT id, purpose, employee_id, org_id, expires_at FROM oauth_attempts
        WHERE state = $1 AND status = 'pending'
        """,
        state,
    )

    if attempt is None:
        return HTMLResponse(
            _error_page("Invalid or already-used login attempt."),
            status_code=400,
        )

    attempt_id = attempt["id"]

    if attempt["expires_at"] <= datetime.now(timezone.utc):
        await db.execute(
            "UPDATE oauth_attempts SET status = 'expired' WHERE id = $1",
            attempt_id,
        )
        return HTMLResponse(_error_page("Login attempt has expired. Please try again."))

    if attempt["purpose"] == "gmail":
        return await _handle_gmail_callback(db, attempt_id, attempt, code)

    # ── Exchange authorization code for tokens ──────────────────────────────
    try:
        id_info = await _exchange_and_verify_code(code)
    except ValueError as exc:
        await _fail_attempt_by_id(db, attempt_id, str(exc))
        return HTMLResponse(_error_page(str(exc)))

    # ── Verify hd claim — only registered-org Workspace accounts allowed ────
    hd = id_info.get("hd", "")
    if not hd:
        msg = "Only Google Workspace accounts are permitted. Personal Gmail accounts are not allowed."
        await _fail_attempt_by_id(db, attempt_id, msg)
        return HTMLResponse(_error_page(msg))

    # ── Resolve org from central org_env by Workspace domain ─────────────
    org_id = await _resolve_org_for_google_hd(db, hd)
    if not org_id:
        msg = f"Your organization ({hd}) is not registered with Eidetix."
        await _fail_attempt_by_id(db, attempt_id, msg)
        return HTMLResponse(_error_page(msg))

    # ── Fetch the org record from the env's Postgres ──────────────────────
    from base.routing import _resolve_to_ctx  # avoid circular at module level
    default_org = _default_org_id()
    org_ctx: OrgCtx = await _resolve_to_ctx(db, org_id, default_org)

    async with org_ctx.pool.acquire() as env_conn:
        org = await env_conn.fetchrow(
            "SELECT id, name, slug FROM organizations WHERE id = $1",
            org_id,
        )
        if not org:
            msg = f"Your organization ({hd}) is not registered with Eidetix."
            await _fail_attempt_by_id(db, attempt_id, msg)
            return HTMLResponse(_error_page(msg))

        # ── Find or provision employee ────────────────────────────────────
        google_sub = id_info["sub"]
        email = id_info.get("email", "").lower()
        name = id_info.get("name") or id_info.get("email", email)
        avatar_url = id_info.get("picture")

        try:
            employee = await _find_or_provision_employee(
                env_conn, org, google_sub, email, name, avatar_url
            )
        except HTTPException as exc:
            await _fail_attempt_by_id(db, attempt_id, exc.detail)
            return HTMLResponse(_error_page(exc.detail))

        # Update last_login_at
        await env_conn.execute(
            "UPDATE employees SET last_login_at = now() WHERE id = $1",
            employee["id"],
        )

    # ── Generate a short-lived bootstrap_token for the desktop to claim ─────
    # The browser itself does not get a session — it shows a success page and
    # closes. The desktop app exchanges the bootstrap_token for a real session
    # via POST /auth/google/exchange.
    bootstrap_token = secrets.token_urlsafe(32)
    bootstrap_expires_at = datetime.now(timezone.utc) + _BOOTSTRAP_TTL

    # Store org_id in the attempt row so exchange() knows which env to use.
    await db.execute(
        """
        UPDATE oauth_attempts
        SET status = 'completed',
            employee_id = $1,
            bootstrap_token = $2,
            completed_at = now(),
            expires_at = $3,
            org_id = $4
        WHERE id = $5
        """,
        employee["id"],
        bootstrap_token,
        bootstrap_expires_at,
        org_id,
        attempt_id,
    )

    return HTMLResponse(_success_page(name))


# ---------------------------------------------------------------------------
# GET /auth/google/status?attempt_id=<id>
# ---------------------------------------------------------------------------

@router.get(
    "/auth/google/status",
    response_model=StatusResponse,
    summary="Poll for Google OAuth completion",
    description=(
        "Called by the desktop app to check whether the OAuth flow has completed. "
        "Returns bootstrap_token once when status=completed. "
        "The bootstrap_token must be exchanged immediately via POST /auth/google/exchange."
    ),
)
async def google_status(
    attempt_id: str = Query(...),
    db: DbConn = Depends(get_db),
) -> StatusResponse:
    attempt = await db.fetchrow(
        """
        SELECT status, bootstrap_token, error_message, expires_at
        FROM oauth_attempts
        WHERE id = $1
          AND purpose = 'login'
        """,
        attempt_id,
    )

    if attempt is None:
        raise HTTPException(status_code=404, detail="Attempt not found.")

    now = datetime.now(timezone.utc)

    # Expire pending attempts that have exceeded their TTL
    if attempt["status"] == "pending" and attempt["expires_at"] <= now:
        await db.execute(
            "UPDATE oauth_attempts SET status = 'expired' WHERE id = $1",
            attempt_id,
        )
        return StatusResponse(status="expired")

    return StatusResponse(
        status=attempt["status"],
        bootstrap_token=attempt["bootstrap_token"] if attempt["status"] == "completed" else None,
        error=attempt["error_message"] if attempt["status"] == "failed" else None,
    )


# ---------------------------------------------------------------------------
# GET /auth/gmail/connect/status?attempt_id=<id>
# ---------------------------------------------------------------------------

@router.get(
    "/auth/gmail/connect/status",
    response_model=GmailConnectStatusResponse,
    summary="Poll for Gmail connector consent completion",
    description=(
        "Called by the desktop app to check whether Gmail consent has "
        "completed. Only the employee who started the attempt may poll it."
    ),
)
async def gmail_connect_status(
    employee: CurrentEmployee,
    attempt_id: str = Query(...),
    db: DbConn = Depends(get_db),
) -> GmailConnectStatusResponse:
    attempt = await db.fetchrow(
        """
        SELECT status, purpose, employee_id, error_message, expires_at
        FROM oauth_attempts
        WHERE id = $1
        """,
        attempt_id,
    )

    if attempt is None or attempt["purpose"] != "gmail":
        raise HTTPException(status_code=404, detail="Attempt not found.")

    if attempt["employee_id"] != employee["id"]:
        raise HTTPException(status_code=403, detail="This attempt belongs to a different account.")

    now = datetime.now(timezone.utc)

    if attempt["status"] == "pending" and attempt["expires_at"] <= now:
        await db.execute(
            "UPDATE oauth_attempts SET status = 'expired' WHERE id = $1",
            attempt_id,
        )
        return GmailConnectStatusResponse(status="expired")

    gmail_address: str | None = None
    if attempt["status"] == "completed":
        org_ctx: OrgCtx = employee["resolved_org_ctx"]
        async with org_ctx.pool.acquire() as env_conn:
            row = await env_conn.fetchrow(
                "SELECT gmail_address FROM gmail_connections WHERE org_id = $1 AND employee_id = $2",
                org_ctx.org_id,
                employee["id"],
            )
        gmail_address = row["gmail_address"] if row else None

    return GmailConnectStatusResponse(
        status=attempt["status"],
        gmail_address=gmail_address,
        error=attempt["error_message"] if attempt["status"] == "failed" else None,
    )


# ---------------------------------------------------------------------------
# POST /auth/google/exchange
# ---------------------------------------------------------------------------

@router.post(
    "/auth/google/exchange",
    response_model=ExchangeResponse,
    summary="Exchange bootstrap token for a desktop session",
    description=(
        "Consumes the one-time bootstrap_token returned by /auth/google/status "
        "and issues a full 30-day bearer session for the desktop app."
    ),
)
async def google_exchange(
    body: ExchangeRequest,
    request: Request,
    db: DbConn = Depends(get_db),
) -> ExchangeResponse:
    now = datetime.now(timezone.utc)

    # Find completed attempt with this bootstrap_token that hasn't been claimed yet
    attempt = await db.fetchrow(
        """
        SELECT id, employee_id, expires_at, org_id
        FROM oauth_attempts
        WHERE bootstrap_token = $1
          AND status = 'completed'
        """,
        body.bootstrap_token,
    )

    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or already-used bootstrap token.",
        )

    if attempt["expires_at"] <= now:
        await db.execute(
            "UPDATE oauth_attempts SET bootstrap_token = NULL WHERE id = $1",
            attempt["id"],
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bootstrap token has expired. Please sign in again.",
        )

    # Consume the bootstrap_token — one-time use
    await db.execute(
        "UPDATE oauth_attempts SET bootstrap_token = NULL WHERE id = $1",
        attempt["id"],
    )

    # Resolve the org's env (org_id stored in attempt row by google_callback)
    org_id = attempt["org_id"] or _default_org_id()
    from base.routing import _resolve_to_ctx
    default_org = _default_org_id()
    org_ctx: OrgCtx = await _resolve_to_ctx(db, org_id, default_org)

    async with org_ctx.pool.acquire() as env_conn:
        employee = await env_conn.fetchrow(
            "SELECT id, email, canonical_name, role, status FROM employees WHERE id = $1",
            attempt["employee_id"],
        )
        if employee is None or employee["status"] != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not active.",
            )

        # Issue a desktop-app session (30 days) in the env Postgres
        client_ip = request.client.host if request.client else None
        raw_token, expires_at = await _create_session(
            env_conn, employee["id"], "desktop-app", None, client_ip
        )

    # Dual-write token_org to central
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    await dual_write_token_org(db, token_hash, org_ctx.org_id, expires_at)

    return ExchangeResponse(
        token=raw_token,
        expires_at=expires_at,
        employee=EmployeeInfo(
            id=employee["id"],
            email=employee["email"],
            name=employee["canonical_name"],
            role=employee["role"],
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _exchange_code(code: str) -> tuple[dict, dict]:
    """Exchange an authorization code for tokens and verify the id_token.

    Returns (id_info, token_data) on success, where token_data includes
    access_token, refresh_token (only present when access_type=offline was
    requested), and expires_in.
    Raises ValueError with a user-friendly message on failure.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: exchange code for tokens at Google's token endpoint
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": _CLIENT_ID,
                "client_secret": _CLIENT_SECRET,
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
        )

    if token_resp.status_code != 200:
        raise ValueError("Failed to exchange authorization code with Google.")

    token_data = token_resp.json()
    id_token = token_data.get("id_token")
    if not id_token:
        raise ValueError("Google did not return an id_token.")

    # Step 2: verify id_token via Google's tokeninfo endpoint.
    # For V1 scale this is acceptable; switch to local JWT verification
    # (google-auth library) if token-endpoint latency becomes an issue.
    async with httpx.AsyncClient(timeout=10.0) as client:
        info_resp = await client.get(
            _GOOGLE_TOKENINFO_URL,
            params={"id_token": id_token},
        )

    if info_resp.status_code != 200:
        raise ValueError("Could not verify Google identity token.")

    id_info = info_resp.json()

    # Verify the token is intended for our client
    if id_info.get("aud") != _CLIENT_ID:
        raise ValueError("Identity token audience mismatch — possible replay attack.")

    if not id_info.get("email_verified"):
        raise ValueError("Google account email is not verified.")

    return id_info, token_data


async def _exchange_and_verify_code(code: str) -> dict:
    """Backwards-compatible wrapper — returns just the id_token claims."""
    id_info, _token_data = await _exchange_code(code)
    return id_info


async def _handle_gmail_callback(
    db: DbConn,
    attempt_id: str,
    attempt: Any,
    code: str,
) -> HTMLResponse:
    """Handle /oauth/callback for a purpose='gmail' attempt.

    Unlike the login flow, employee_id/org_id are already known (set at
    /auth/gmail/connect/start time by the authenticated caller). This
    exchanges the code, encrypts the refresh/access tokens, and upserts
    gmail_connections. No session is created or modified.
    """
    employee_id = attempt["employee_id"]
    org_id = attempt["org_id"]
    if not employee_id or not org_id:
        msg = "Invalid Gmail connect attempt. Please retry from the desktop app."
        await _fail_attempt_by_id(db, attempt_id, msg)
        return HTMLResponse(_error_page(msg))

    try:
        id_info, token_data = await _exchange_code(code)
    except ValueError as exc:
        await _fail_attempt_by_id(db, attempt_id, str(exc))
        return HTMLResponse(_error_page(str(exc)))

    refresh_token = token_data.get("refresh_token")
    access_token = token_data.get("access_token")
    expires_in = token_data.get("expires_in")

    if not refresh_token:
        # Google omits refresh_token if the user already granted consent
        # without prompt=consent forcing a fresh one. We always request
        # prompt=consent for the gmail purpose, so this should not happen —
        # but guard defensively so a stale/cached authorization can't
        # silently overwrite a good refresh token with nothing.
        msg = (
            "Google did not return a refresh token. Please try connecting "
            "Gmail again and approve all requested permissions."
        )
        await _fail_attempt_by_id(db, attempt_id, msg)
        return HTMLResponse(_error_page(msg))

    gmail_address = (id_info.get("email") or "").lower()
    if not gmail_address:
        msg = "Google did not return an email address for this account."
        await _fail_attempt_by_id(db, attempt_id, msg)
        return HTMLResponse(_error_page(msg))

    from base.routing import _resolve_to_ctx  # avoid circular at module level
    default_org = _default_org_id()
    org_ctx: OrgCtx = await _resolve_to_ctx(db, org_id, default_org)

    try:
        encrypted_refresh_token = encrypt_secret(refresh_token)
        encrypted_access_token = encrypt_secret(access_token) if access_token else None
    except CredentialEncryptionError as exc:
        await _fail_attempt_by_id(db, attempt_id, str(exc))
        return HTMLResponse(_error_page(str(exc)))
    access_token_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
        if expires_in is not None
        else None
    )

    async with org_ctx.pool.acquire() as env_conn:
        mailbox_conflict = False
        async with env_conn.transaction():
            # Serialize ownership claims, including when no connection row
            # exists yet. The partial unique index is the database backstop.
            await env_conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"{org_id}\x00{gmail_address}",
            )
            employee = await env_conn.fetchrow(
                "SELECT id, canonical_name, status FROM employees WHERE id = $1",
                employee_id,
            )
            if employee is None or employee["status"] != "active":
                msg = "Your account is not active. Please contact your administrator."
                await _fail_attempt_by_id(db, attempt_id, msg)
                return HTMLResponse(_error_page(msg))

            existing_owner = await env_conn.fetchval(
                """
                SELECT employee_id
                FROM gmail_connections
                WHERE org_id = $1
                  AND lower(gmail_address) = $2
                  AND status = 'active'
                  AND employee_id <> $3
                LIMIT 1
                """,
                org_id,
                gmail_address,
                employee_id,
            )
            mailbox_conflict = existing_owner is not None
            if not mailbox_conflict:
                connection_id = f"gmailconn:{employee_id}"
                await env_conn.execute(
                    """
                    INSERT INTO gmail_connections
                        (id, org_id, employee_id, gmail_address,
                         encrypted_refresh_token, encrypted_access_token,
                         access_token_expires_at, token_stored_at, status, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, now(), 'active', NULL)
                    ON CONFLICT (org_id, employee_id) DO UPDATE
                        SET gmail_address             = EXCLUDED.gmail_address,
                            encrypted_refresh_token    = EXCLUDED.encrypted_refresh_token,
                            encrypted_access_token     = EXCLUDED.encrypted_access_token,
                            access_token_expires_at    = EXCLUDED.access_token_expires_at,
                            crawl_checkpoint          = CASE
                                WHEN gmail_connections.gmail_address IS DISTINCT FROM EXCLUDED.gmail_address
                                THEN NULL
                                ELSE gmail_connections.crawl_checkpoint
                            END,
                            ingestion_checkpoint      = CASE
                                WHEN gmail_connections.gmail_address IS DISTINCT FROM EXCLUDED.gmail_address
                                THEN NULL
                                ELSE gmail_connections.ingestion_checkpoint
                            END,
                            token_stored_at            = now(),
                            status                     = 'active',
                            error_message              = NULL
                    """,
                    connection_id,
                    org_id,
                    employee_id,
                    gmail_address,
                    encrypted_refresh_token,
                    encrypted_access_token,
                    access_token_expires_at,
                )

        if mailbox_conflict:
            msg = "This Gmail account is already connected by another employee."
            await _fail_attempt_by_id(db, attempt_id, msg)
            return HTMLResponse(_error_page(msg))

    await db.execute(
        """
        UPDATE oauth_attempts
        SET status = 'completed', completed_at = now()
        WHERE id = $1
        """,
        attempt_id,
    )

    return HTMLResponse(_gmail_success_page(gmail_address))


async def _find_or_provision_employee(
    db: DbConn,
    org: dict,
    google_sub: str,
    email: str,
    name: str,
    avatar_url: str | None,
) -> dict:
    """Find an existing employee by Google sub or email, or auto-provision a new one.

    - Preferred lookup: auth_provider='google', auth_provider_id=google_sub
    - Fallback lookup: by email (links a local account to Google)
    - No match: auto-provision a new employee in the org

    Returns the employee asyncpg Record.
    Raises HTTPException on errors (suspended, deactivated, org mismatch).
    """
    # 1. Look up by Google sub (most reliable — stable across email changes)
    emp = await db.fetchrow(
        "SELECT * FROM employees WHERE auth_provider = 'google' AND auth_provider_id = $1",
        google_sub,
    )
    if emp:
        if emp["status"] == "deactivated":
            raise HTTPException(status_code=403, detail="This account has been deactivated.")
        if emp["status"] == "suspended":
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        # Update avatar_url if it changed
        if avatar_url and emp["avatar_url"] != avatar_url:
            await db.execute(
                "UPDATE employees SET avatar_url = $1 WHERE id = $2",
                avatar_url,
                emp["id"],
            )
        return emp

    # 2. Look up by email — may be an existing local-password account
    emp = await db.fetchrow(
        "SELECT * FROM employees WHERE email = $1 AND org_id = $2",
        email,
        org["id"],
    )
    if emp:
        from base.db.identity import is_identity_stub

        if is_identity_stub(
            employee_id=emp["id"],
            email=emp.get("email"),
            auth_provider=emp.get("auth_provider"),
        ):
            raise HTTPException(
                status_code=403,
                detail="This email is not an organization account. Ask an admin for an invite.",
            )
        if emp["status"] == "deactivated":
            raise HTTPException(status_code=403, detail="This account has been deactivated.")
        if emp["status"] == "suspended":
            raise HTTPException(status_code=403, detail="This account has been suspended.")
        # Link this existing account to Google (e.g., was previously 'local')
        await db.execute(
            """
            UPDATE employees
            SET auth_provider = 'google',
                auth_provider_id = $1,
                avatar_url = COALESCE($2, avatar_url)
            WHERE id = $3
            """,
            google_sub,
            avatar_url,
            emp["id"],
        )
        return emp

    # 3. Auto-provision — first Google sign-in for this user in this org.
    # ON CONFLICT DO NOTHING guards against a race where two concurrent
    # requests attempt to create the same employee (e.g., double-click).
    emp_id = f"emp:{email.split('@')[0]}-{secrets.token_hex(4)}"
    await db.execute(
        """
        INSERT INTO employees
            (id, org_id, email, canonical_name, role, status,
             auth_provider, auth_provider_id, avatar_url)
        VALUES ($1, $2, $3, $4, 'member', 'active', 'google', $5, $6)
        ON CONFLICT (org_id, email) DO NOTHING
        """,
        emp_id,
        org["id"],
        email,
        name,
        google_sub,
        avatar_url,
    )

    emp = await db.fetchrow(
        "SELECT * FROM employees WHERE email = $1 AND org_id = $2",
        email,
        org["id"],
    )
    if emp is None:
        raise HTTPException(status_code=500, detail="Failed to provision employee account.")
    from base.db.identity import is_identity_stub

    if is_identity_stub(
        employee_id=emp["id"],
        email=emp.get("email"),
        auth_provider=emp.get("auth_provider"),
    ):
        raise HTTPException(
            status_code=403,
            detail="This email is not an organization account. Ask an admin for an invite.",
        )
    return emp


async def _create_session(
    db: DbConn,
    employee_id: str,
    client_type: str,
    device_info: str | None,
    ip_address: str | None,
) -> tuple[str, datetime]:
    """Insert a new session row and return (raw_token, expires_at)."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    expires_at = datetime.now(timezone.utc) + _SESSION_TTL

    await db.execute(
        """
        INSERT INTO sessions
            (employee_id, token_hash, client_type, device_info, ip_address, expires_at)
        VALUES ($1, $2, $3, $4, $5::inet, $6)
        """,
        employee_id,
        token_hash,
        client_type,
        device_info,
        ip_address,
        expires_at,
    )

    return raw_token, expires_at


async def _fail_attempt_by_state(db: DbConn, state: str | None, message: str) -> None:
    if not state:
        return
    await db.execute(
        "UPDATE oauth_attempts SET status = 'failed', error_message = $1 WHERE state = $2",
        message,
        state,
    )


async def _fail_attempt_by_id(db: DbConn, attempt_id: str, message: str) -> None:
    await db.execute(
        "UPDATE oauth_attempts SET status = 'failed', error_message = $1 WHERE id = $2",
        message,
        attempt_id,
    )


# ---------------------------------------------------------------------------
# HTML page templates
# ---------------------------------------------------------------------------

def _success_page(name: str) -> str:
    safe_name = name.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Signed in — Eidetix</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .card {{
      background: #1a1d27;
      border: 1px solid #2d3148;
      border-radius: 12px;
      padding: 48px 40px;
      text-align: center;
      max-width: 420px;
      width: 100%;
    }}
    .icon {{ font-size: 48px; margin-bottom: 20px; }}
    h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 10px; }}
    p {{ color: #94a3b8; font-size: 15px; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">✓</div>
    <h1>Signed in as {safe_name}</h1>
    <p>You can close this window and return to the Eidetix desktop app.</p>
  </div>
</body>
</html>"""


def _gmail_success_page(gmail_address: str) -> str:
    safe_address = gmail_address.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gmail connected — Eidetix</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .card {{
      background: #1a1d27;
      border: 1px solid #2d3148;
      border-radius: 12px;
      padding: 48px 40px;
      text-align: center;
      max-width: 420px;
      width: 100%;
    }}
    .icon {{ font-size: 48px; margin-bottom: 20px; }}
    h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 10px; }}
    p {{ color: #94a3b8; font-size: 15px; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">✓</div>
    <h1>Gmail connected</h1>
    <p>{safe_address} is now connected to Eidetix.<br>You can close this window and return to the Eidetix desktop app.</p>
  </div>
</body>
</html>"""


def _error_page(message: str) -> str:
    safe_msg = message.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign-in error — Eidetix</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .card {{
      background: #1a1d27;
      border: 1px solid #3d2020;
      border-radius: 12px;
      padding: 48px 40px;
      text-align: center;
      max-width: 420px;
      width: 100%;
    }}
    .icon {{ font-size: 48px; margin-bottom: 20px; }}
    h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 10px; color: #f87171; }}
    p {{ color: #94a3b8; font-size: 15px; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">✗</div>
    <h1>Sign-in failed</h1>
    <p>{safe_msg}</p>
  </div>
</body>
</html>"""
