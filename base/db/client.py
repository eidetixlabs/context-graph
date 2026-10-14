"""
db/client.py — asyncpg connection pool for the Eidetix cloud server.

Usage
-----
At application startup (FastAPI lifespan):
    await db_pool.open()

In route handlers / dependencies:
    async def my_route(db: DbConn = Depends(get_db)):
        row = await db.fetchrow("SELECT ...")

At shutdown:
    await db_pool.close()

In one-off workers:
    async with db_pool:
        ...

Per-org routing
---------------
OrgPoolRegistry provides lazy per-org asyncpg pools keyed by org_id.
The central pool (db_pool / DATABASE_URL) is also the default org's pool;
the registry short-circuits to it for the default org so no extra connection
is opened.

    pool = await org_pool_registry.get(org_id, pg_dsn)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import AsyncGenerator
from urllib.parse import parse_qsl, urlparse

import asyncpg
from asyncpg.pool import PoolConnectionProxy

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSL helper (unchanged)
# ---------------------------------------------------------------------------


def _pool_ssl_kwarg(dsn: str) -> dict:
    """Return {"ssl": ...} only when we need to override asyncpg defaults."""

    flag = os.environ.get("DATABASE_SSL", "").strip().lower()
    if flag in ("1", "true", "yes", "require"):
        return {"ssl": True}
    if flag in ("0", "false", "no", "disable", "off"):
        return {"ssl": False}
    if flag:
        _log.warning("Unrecognised DATABASE_SSL value %r — deferring to asyncpg default (prefer)", flag)
        return {}

    if os.environ.get("PGSSLMODE"):
        return {}

    query_keys = {k.lower() for k, _ in parse_qsl(urlparse(dsn).query)}
    if "sslmode" in query_keys:
        return {}

    return {"ssl": False}


# ---------------------------------------------------------------------------
# Pool singleton (central / default org)
# ---------------------------------------------------------------------------

class _DbPool:
    """Thin wrapper around asyncpg.Pool that supports lazy initialisation."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._cm_depth: int = 0
        self._cm_open_count: int = 0

    async def open(self) -> None:
        if self._pool is not None:
            return
        dsn = os.environ.get("DATABASE_URL", "")
        if not dsn:
            raise RuntimeError(
                "DATABASE_URL is not set. "
                "Add it to the .env file (or SSM) and restart the server."
            )
        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
            server_settings={"timezone": "UTC"},
            **_pool_ssl_kwarg(dsn),
        )

    async def __aenter__(self) -> "_DbPool":
        opened_here = self._pool is None
        await self.open()
        self._cm_depth += 1
        if opened_here:
            self._cm_open_count += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._cm_depth -= 1
        if self._cm_open_count > 0 and self._cm_depth == 0:
            self._cm_open_count -= 1
            await self.close()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def get(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "DB pool is not initialised. "
                "Call await db_pool.open() first or use 'async with db_pool'."
            )
        return self._pool


db_pool = _DbPool()


# ---------------------------------------------------------------------------
# FastAPI dependency (central / default org)
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[PoolConnectionProxy, None]:
    """
    FastAPI dependency that acquires a connection from the pool for the
    duration of a single request and releases it afterwards.

    Example::

        async def my_route(db: DbConn = Depends(get_db)):
            row = await db.fetchrow("SELECT id FROM employees WHERE email = $1", email)
    """
    async with db_pool.get().acquire() as conn:
        yield conn


# Type alias used in route function signatures (pool.acquire() yields a proxy).
DbConn = PoolConnectionProxy


# ---------------------------------------------------------------------------
# Per-org pool registry
# ---------------------------------------------------------------------------

_ORG_POOL_MAX_SIZE = 5        # connections per org pool
_ORG_POOL_MIN_SIZE = 1
_ORG_POOL_COMMAND_TIMEOUT = 30
_ORG_REGISTRY_LRU_CAP = 50   # max distinct org pools kept open simultaneously


class OrgPoolRegistry:
    """
    Lazy per-org asyncpg pool registry.

    The central pool (db_pool) is ALSO the default org's pool.  Callers pass
    default_org_id so the registry can short-circuit to db_pool for that org.

    All non-default org pools are created on first request, cached, and
    evicted via an LRU cap when the number of open pools exceeds
    _ORG_REGISTRY_LRU_CAP.

    Thread/async safety: all mutations are protected by a single asyncio.Lock.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Ordered so we can evict the least-recently-used entry
        self._pools: OrderedDict[str, asyncpg.Pool] = OrderedDict()
        self._default_org_id: str = ""

    def set_default_org(self, org_id: str) -> None:
        self._default_org_id = org_id

    async def get(self, org_id: str, pg_dsn: str) -> asyncpg.Pool:
        """Return the pool for org_id, creating it from pg_dsn if needed."""
        if org_id == self._default_org_id:
            return db_pool.get()

        async with self._lock:
            if org_id in self._pools:
                self._pools.move_to_end(org_id)
                return self._pools[org_id]

            pool = await asyncpg.create_pool(
                pg_dsn,
                min_size=_ORG_POOL_MIN_SIZE,
                max_size=_ORG_POOL_MAX_SIZE,
                command_timeout=_ORG_POOL_COMMAND_TIMEOUT,
                server_settings={"timezone": "UTC"},
                **_pool_ssl_kwarg(pg_dsn),
            )
            self._pools[org_id] = pool
            self._pools.move_to_end(org_id)

            # Evict LRU if over cap
            while len(self._pools) > _ORG_REGISTRY_LRU_CAP:
                evicted_org, evicted_pool = self._pools.popitem(last=False)
                _log.info("OrgPoolRegistry: evicting pool for org=%s (LRU cap reached)", evicted_org)
                try:
                    await evicted_pool.close()
                except Exception as exc:
                    _log.warning("OrgPoolRegistry: error closing evicted pool org=%s: %s", evicted_org, exc)

            return pool

    async def remove(self, org_id: str) -> None:
        """Close and remove the pool for an org (called by edicli env remove)."""
        async with self._lock:
            pool = self._pools.pop(org_id, None)
        if pool is not None:
            try:
                await pool.close()
            except Exception as exc:
                _log.warning("OrgPoolRegistry: error closing pool for org=%s: %s", org_id, exc)

    async def close_all(self) -> None:
        """Close all non-default org pools (called on webserver shutdown)."""
        async with self._lock:
            pools = list(self._pools.items())
            self._pools.clear()
        for org_id, pool in pools:
            try:
                await pool.close()
            except Exception as exc:
                _log.warning("OrgPoolRegistry: error closing pool org=%s on shutdown: %s", org_id, exc)


org_pool_registry = OrgPoolRegistry()
