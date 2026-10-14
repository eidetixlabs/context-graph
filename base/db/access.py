"""Scope resolution for RBAC access control.

Resolves the set of channel_ids an employee is allowed to access,
with a TTL cache to avoid hammering the DB on every query.
"""

import asyncio
import threading
from typing import Optional

import asyncpg
from cachetools import TTLCache

_CACHE_TTL = 300  # 5 minutes
_CACHE_SIZE = 1000

_cache: TTLCache = TTLCache(maxsize=_CACHE_SIZE, ttl=_CACHE_TTL)
_cache_lock = threading.Lock()


def trace_personal_scope(employee_id: str) -> str:
    """Return the RBAC scope id that owns one employee's private traces."""
    return f"scope:trace:{employee_id}"


async def get_allowed_scopes(
    pool: asyncpg.Pool, employee_id: Optional[str]
) -> Optional[list[str]]:
    """Return the channel_ids this employee is allowed to access.

    Returns None for service accounts (employee_id=None), which means
    bypass all scope filtering. Returns a list of channel_id strings for
    human employees (may be empty if they have no channel memberships).
    """
    if employee_id is None:
        return None

    with _cache_lock:
        cached = _cache.get(employee_id)
        if cached is not None:
            return cached

    rows = await pool.fetch(
        "SELECT channel_id FROM channel_members WHERE employee_id = $1",
        employee_id,
    )
    scopes = [r["channel_id"] for r in rows]
    personal_scope = trace_personal_scope(employee_id)
    if personal_scope not in scopes:
        scopes.append(personal_scope)

    with _cache_lock:
        _cache[employee_id] = scopes

    return scopes


def invalidate_scope_cache(employee_id: str) -> None:
    """Remove a single employee's cached scopes (e.g. after membership change)."""
    with _cache_lock:
        _cache.pop(employee_id, None)
