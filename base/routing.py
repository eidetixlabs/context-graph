"""
base/routing.py — Per-org request routing for the Eidetix webserver.

Resolves which org owns a request, then hands back the correct
Postgres pool + Neo4j client for that org from the cached registries.

Resolution order
----------------
1. Bearer session token  → look up token_org in the central Postgres.
   Miss (no row or revoked) → fall back to the default org.
2. API_KEY bearer         → read X-Eidetix-Org header.
   Missing header          → fall back to the default org.

The default org always short-circuits to the process-global db_pool and
the global Neo4jClient, so no extra connections are opened for single-org
deployments (i.e. today's on-prem default behaviour is unchanged).

Public surface
--------------
OrgCtx                  — dataclass returned by get_org_ctx
get_org_ctx             — FastAPI dependency (requires Authorization header)
get_org_ctx_from_token  — lower-level helper used by auth deps
dual_write_token_org    — write token_hash + org_id to central token_org
revoke_token_org        — mark token revoked in central token_org
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Annotated, Any

import asyncpg
try:
    from fastapi import Depends, Header
except ImportError:  # pragma: no cover - used only in lightweight unit test environments
    def Depends(value):  # type: ignore[override]
        return value

    def Header(default=None, alias=None):  # type: ignore[override]
        return default

from base.db.client import DbConn, db_pool, get_db, org_pool_registry
from base.neo4j_client import Neo4jClient, org_neo4j_registry

_log = logging.getLogger(__name__)

# In-process TTL for the token_hash → org_id cache (seconds).
# Revocations propagate within this window.
_TOKEN_ORG_CACHE_TTL_S = 60

# {token_hash: (org_id | None, monotonic_expiry)}
_token_org_cache: dict[str, tuple[str | None, float]] = {}


def _allow_default_org_fallback() -> bool:
    raw = os.environ.get("EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# OrgCtx — the resolved context passed to every route
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OrgCtx:
    org_id: str
    pool: asyncpg.Pool
    neo4j: Neo4jClient


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _lookup_token_org(central_db: DbConn, token_hash: str) -> str | None:
    """Return org_id from token_org, or None if not found / revoked / expired.

    Results are cached in-process for _TOKEN_ORG_CACHE_TTL_S seconds so
    repeated requests with the same token skip the central-Postgres round-trip.
    """
    cached = _token_org_cache.get(token_hash)
    if cached is not None:
        org_id, expiry = cached
        if time.monotonic() < expiry:
            return org_id
        del _token_org_cache[token_hash]

    row = await central_db.fetchrow(
        """
        SELECT org_id FROM token_org
        WHERE token_hash = $1
          AND revoked_at IS NULL
          AND expires_at > now()
        """,
        token_hash,
    )
    org_id = row["org_id"] if row else None
    _token_org_cache[token_hash] = (org_id, time.monotonic() + _TOKEN_ORG_CACHE_TTL_S)
    return org_id


async def _fetch_org_env(central_db: DbConn, org_id: str) -> dict | None:
    """Fetch the org_env row for org_id from the central Postgres."""
    row = await central_db.fetchrow(
        """
        SELECT org_id, pg_dsn, neo4j_uri, neo4j_user, neo4j_password, neo4j_database
        FROM org_env
        WHERE org_id = $1 AND status = 'active'
        """,
        org_id,
    )
    return dict(row) if row else None


async def _resolve_to_ctx(central_db: DbConn, org_id: str, default_org_id: str) -> OrgCtx:
    """
    Given a resolved org_id, return the OrgCtx (pool + neo4j client).

    For the default org this is a cheap short-circuit to process globals.
    For other orgs we fetch org_env coords and ask the registries (cached).
    """
    if org_id == default_org_id:
        default_client = org_neo4j_registry.get_default()
        if default_client is None:
            raise RuntimeError("Default Neo4j client not initialised yet")
        return OrgCtx(org_id=org_id, pool=db_pool.get(), neo4j=default_client)

    env = await _fetch_org_env(central_db, org_id)
    if env is None:
        if not _allow_default_org_fallback():
            raise RuntimeError(
                f"org_env row missing or inactive for org={org_id}; "
                "set EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK=1 to opt into fallback"
            )
        _log.warning("org_env row missing or inactive for org=%s — falling back to default", org_id)
        default_client = org_neo4j_registry.get_default()
        if default_client is None:
            raise RuntimeError("Default Neo4j client not initialised yet")
        return OrgCtx(org_id=default_org_id, pool=db_pool.get(), neo4j=default_client)

    pool = await org_pool_registry.get(org_id, env["pg_dsn"])
    client = await org_neo4j_registry.get(
        org_id,
        neo4j_uri=env["neo4j_uri"],
        neo4j_user=env["neo4j_user"],
        neo4j_password=env["neo4j_password"],
        neo4j_database=env["neo4j_database"],
    )
    return OrgCtx(org_id=org_id, pool=pool, neo4j=client)


async def resolve_org_ctx(
    central_db: DbConn,
    org_id: str,
    default_org_id: str,
) -> OrgCtx:
    """Resolve a known org_id to its env Postgres/Neo4j context."""
    return await _resolve_to_ctx(central_db, org_id, default_org_id)


# ---------------------------------------------------------------------------
# Public: token-based org resolution (used by auth deps)
# ---------------------------------------------------------------------------

async def resolve_org_from_token(
    central_db: DbConn,
    raw_token: str,
    default_org_id: str,
) -> OrgCtx:
    """
    Resolve org from a bearer session token.

    1. Hash token -> look up token_org in central.
    2. Miss -> default org.
    3. Resolve coords from org_env, return OrgCtx.
    """
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    org_id = await _lookup_token_org(central_db, token_hash)
    if org_id is None:
        if not _allow_default_org_fallback():
            raise RuntimeError(
                "token_org row missing/revoked/expired; "
                "set EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK=1 to opt into fallback"
            )
        org_id = default_org_id
    return await _resolve_to_ctx(central_db, org_id, default_org_id)


async def resolve_org_id_from_email_domain(central_db: DbConn, email: str) -> str | None:
    """Return the active org that has claimed this email domain, if any."""
    if "@" not in email:
        return None
    domain = email.strip().lower().rsplit("@", 1)[-1]
    if not domain:
        return None
    row = await central_db.fetchrow(
        """
        SELECT org_id
        FROM org_env
        WHERE email_domain = $1 AND status = 'active'
        """,
        domain,
    )
    if row:
        return str(row["org_id"])

    legacy_slug = domain.split(".", 1)[0]
    row = await central_db.fetchrow(
        """
        SELECT org_id
        FROM org_env
        WHERE slug = $1 AND status = 'active'
        """,
        legacy_slug,
    )
    return str(row["org_id"]) if row else None


async def resolve_org_from_header(
    central_db: DbConn,
    org_header: str | None,
    default_org_id: str,
) -> OrgCtx:
    """
    Resolve org from the X-Eidetix-Org header (used by API_KEY / service callers).

    Missing or empty header -> default org.
    Unknown org -> default org (with a warning).
    """
    if not org_header:
        if not _allow_default_org_fallback():
            raise RuntimeError(
                "X-Eidetix-Org header missing; "
                "set EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK=1 to opt into fallback"
            )
        return await _resolve_to_ctx(central_db, default_org_id, default_org_id)
    # Normalise: allow plain slug "acme" or "org:acme"
    org_id = org_header if org_header.startswith("org:") else f"org:{org_header}"
    return await _resolve_to_ctx(central_db, org_id, default_org_id)


# ---------------------------------------------------------------------------
# Public: dual-write helpers (called by auth routes)
# ---------------------------------------------------------------------------

async def dual_write_token_org(
    central_db: DbConn,
    token_hash: str,
    org_id: str,
    expires_at: Any,
) -> None:
    """Insert (or upsert) a token_org row in the central Postgres."""
    await central_db.execute(
        """
        INSERT INTO token_org (token_hash, org_id, expires_at)
        VALUES ($1, $2, $3)
        ON CONFLICT (token_hash) DO UPDATE
            SET org_id     = EXCLUDED.org_id,
                expires_at = EXCLUDED.expires_at,
                revoked_at = NULL
        """,
        token_hash,
        org_id,
        expires_at,
    )
    _token_org_cache[token_hash] = (org_id, time.monotonic() + _TOKEN_ORG_CACHE_TTL_S)


async def revoke_token_org(central_db: DbConn, token_hash: str) -> None:
    """Mark a token as revoked in the central token_org table."""
    _token_org_cache.pop(token_hash, None)
    await central_db.execute(
        "UPDATE token_org SET revoked_at = now() WHERE token_hash = $1",
        token_hash,
    )
