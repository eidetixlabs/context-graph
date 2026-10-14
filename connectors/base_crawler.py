"""
connectors/base_crawler.py — Crawler base class and shared connector primitives.

Exceptions, RunContext, connector instance loading, checkpoints, Postgres helpers
used by crawlers, and BaseCrawler live here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from asyncpg.pool import PoolConnectionProxy

from base.db.client import db_pool

# Connections from pool.acquire() are typed as PoolConnectionProxy, not Connection.
AsyncpgConn = asyncpg.Connection | PoolConnectionProxy

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Generic connector failure."""


class ConnectorAuthError(ConnectorError):
    """401/403 from the external API. Triggers backoff + failure tracking."""


class ConnectorRateLimitError(ConnectorError):
    """429 from the external API. Triggers backoff using Retry-After if present."""

    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ConnectorParseError(ConnectorError):
    """Downloaded content exists but could not be parsed/structured."""


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

_BACKOFF_BASE = 60.0
_BACKOFF_CAP = 600.0
_MAX_AUTH_FAILURES = 5


async def backoff_sleep(attempt: int, retry_after: float | None = None) -> None:
    """Exponential backoff with jitter, starting at 60s, capped at 10 min."""
    if retry_after is not None:
        delay = min(retry_after, _BACKOFF_CAP)
    else:
        delay = min(_BACKOFF_BASE * (2**attempt), _BACKOFF_CAP)
    jitter = random.uniform(0, delay * 0.1)
    total = delay + jitter
    _log.info("Backoff: sleeping %.1f s (attempt %d)", total, attempt)
    await asyncio.sleep(total)


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    connector_id: str
    run_type: str  # 'crawl' | 'ingest'
    items_processed: int = 0
    items_failed: int = 0

    def ok(self) -> None:
        self.items_processed += 1

    def fail(self) -> None:
        self.items_failed += 1


# ---------------------------------------------------------------------------
# ConnectorInstance
# ---------------------------------------------------------------------------


@dataclass
class ConnectorInstance:
    id: str
    org_id: str
    provider: str
    config: dict
    token: str
    status: str


def _parse_connector_config(raw: object) -> dict:
    """Normalize connector_instances.config from JSONB (dict or legacy double-encoded str)."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def ensure_connector_registered(
    connector_id: str,
    org_id: str,
    provider: str,
    config: dict | None = None,
) -> None:
    """Idempotent UPSERT of connector_instances. Called at worker startup.

    Preserves revoked/paused status. Config is updated on each call so Helm/env changes take effect on next run.
    """
    pool = db_pool.get()
    await pool.execute(
        """
        INSERT INTO connector_instances (id, org_id, provider, config, status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, 'active', now(), now())
        ON CONFLICT (id) DO UPDATE
            SET config        = EXCLUDED.config,
                status        = CASE
                                    WHEN connector_instances.status IN ('revoked', 'paused')
                                    THEN connector_instances.status
                                    ELSE 'active'
                                END,
                updated_at    = now()
        """,
        connector_id,
        org_id,
        provider,
        json.dumps(config or {}),
    )
    _log.info("Connector registered: %s (org=%s provider=%s)", connector_id, org_id, provider)


_HAS_ENCRYPTED_TOKEN_COLUMN: bool | None = None


async def _connector_row_has_encrypted_token(pool: asyncpg.Pool) -> bool:
    global _HAS_ENCRYPTED_TOKEN_COLUMN
    if _HAS_ENCRYPTED_TOKEN_COLUMN is not None:
        return _HAS_ENCRYPTED_TOKEN_COLUMN
    found = await pool.fetchval(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'connector_instances'
          AND column_name = 'encrypted_token'
        """
    )
    _HAS_ENCRYPTED_TOKEN_COLUMN = found is not None
    return _HAS_ENCRYPTED_TOKEN_COLUMN


async def load_connector_instance(connector_id: str) -> ConnectorInstance:
    pool = db_pool.get()
    columns = "id, org_id, provider, config, status"
    if await _connector_row_has_encrypted_token(pool):
        columns = f"{columns}, encrypted_token"
    row = await pool.fetchrow(
        f"""
        SELECT {columns}
        FROM connector_instances
        WHERE id = $1
        """,
        connector_id,
    )
    if row is None:
        raise ConnectorError(f"Connector not found: {connector_id}")
    if row["status"] != "active":
        raise ConnectorError(
            f"Connector {connector_id} is not active (status={row['status']})"
        )
    token = os.environ.get("CONNECTOR_TOKEN", "")
    if not token and row.get("encrypted_token"):
        from base.crypto import decrypt_secret

        token = decrypt_secret(row["encrypted_token"])
    if not token and row["provider"] != "gmail":
        raise ConnectorError("CONNECTOR_TOKEN env var is not set")
    return ConnectorInstance(
        id=row["id"],
        org_id=row["org_id"],
        provider=row["provider"],
        config=_parse_connector_config(row["config"]),
        token=token,
        status=row["status"],
    )


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


async def read_checkpoint(connector_id: str, run_type: str) -> datetime | None:
    pool = db_pool.get()
    row = await pool.fetchrow(
        """
        SELECT last_cursor FROM connector_checkpoints
        WHERE connector_id = $1 AND run_type = $2
        """,
        connector_id,
        run_type,
    )
    return row["last_cursor"] if row else None


##TODO : need to do this in transaction. 
async def write_checkpoint(connector_id: str, run_type: str, cursor) -> None:
    pool = db_pool.get()
    await pool.execute(
        """
        INSERT INTO connector_checkpoints (connector_id, run_type, last_cursor, updated_at)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (connector_id, run_type)
        DO UPDATE SET last_cursor = EXCLUDED.last_cursor, updated_at = now()
        """,
        connector_id,
        run_type,
        cursor,
    )


# ---------------------------------------------------------------------------
# Postgres helpers (crawler + shared)
# ---------------------------------------------------------------------------


async def upsert_access_scope(
    conn: AsyncpgConn,
    scope_id: str,
    org_id: str,
    provider: str,
    source_ref: str,
    source_label: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO access_scopes (id, org_id, source_provider, source_ref, source_label)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (org_id, source_provider, source_ref) DO NOTHING
        """,
        scope_id,
        org_id,
        provider,
        source_ref,
        source_label,
    )


async def sync_channel_members(
    conn: AsyncpgConn,
    channel_id: str,
    provider: str,
    members: list[dict],
) -> None:
    external_ids = [m["external_user_id"] for m in members]

    async with conn.transaction():
        await conn.execute(
            """
            DELETE FROM channel_members
            WHERE channel_id = $1
              AND external_user_id != ALL($2::text[])
            """,
            channel_id,
            external_ids,
        )

        for m in members:
            await conn.execute(
                """
                INSERT INTO channel_members
                    (channel_id, employee_id, external_user_id, source_provider, crawled_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (channel_id, external_user_id)
                DO UPDATE SET
                    employee_id = EXCLUDED.employee_id,
                    crawled_at  = now()
                """,
                channel_id,
                m.get("employee_id"),
                m["external_user_id"],
                provider,
            )


# ---------------------------------------------------------------------------
# Mark connector error
# ---------------------------------------------------------------------------


async def mark_connector_error(connector_id: str, message: str) -> None:
    pool = db_pool.get()
    await pool.execute(
        """
        UPDATE connector_instances
        SET status = 'error', error_message = $2, updated_at = now()
        WHERE id = $1
        """,
        connector_id,
        message,
    )
    _log.error("Connector %s marked as error: %s", connector_id, message)


# ---------------------------------------------------------------------------
# BaseCrawler
# ---------------------------------------------------------------------------


class BaseCrawler(ABC):
    """
    Base class for all crawler workers.

    Subclasses implement crawl() which is called on each scheduled run.
    """

    RUN_TYPE = "crawl"

    @abstractmethod
    async def crawl(
        self,
        instance: ConnectorInstance,
        run_ctx: RunContext,
        pool: asyncpg.Pool,
    ) -> None:
        """Provider-specific crawl logic."""

    async def run(self, connector_id: str, interval_seconds: float = 3600.0) -> None:
        _log.info("Crawler starting: connector=%s interval=%ss", connector_id, interval_seconds)
        consecutive_auth_failures = 0

        while True:
            try:
                instance = await load_connector_instance(connector_id)
                run_ctx = RunContext(connector_id=connector_id, run_type=self.RUN_TYPE)

                _log.info("Crawl run starting: %s", connector_id)
                await self.crawl(instance, run_ctx, db_pool.get())
                consecutive_auth_failures = 0
                _log.info(
                    "Crawl run complete: %s processed=%d failed=%d",
                    connector_id,
                    run_ctx.items_processed,
                    run_ctx.items_failed,
                )

            except ConnectorAuthError as exc:
                consecutive_auth_failures += 1
                _log.warning(
                    "Auth failure #%d on connector %s: %s",
                    consecutive_auth_failures,
                    connector_id,
                    exc,
                )
                if consecutive_auth_failures >= _MAX_AUTH_FAILURES:
                    await mark_connector_error(
                        connector_id,
                        f"Auth failed {consecutive_auth_failures} consecutive times: {exc}",
                    )
                    _log.error(
                        "Connector %s disabled after %d auth failures. Fix the token and re-enable.",
                        connector_id,
                        consecutive_auth_failures,
                    )
                    return
                await backoff_sleep(consecutive_auth_failures)
                continue

            except ConnectorRateLimitError as exc:
                _log.warning("Rate limited on connector %s: %s", connector_id, exc)
                await backoff_sleep(0, retry_after=exc.retry_after)
                continue

            except ConnectorError as exc:
                _log.error("Connector error on %s: %s", connector_id, exc)

            except Exception:
                _log.error(
                    "Unexpected error in crawler %s:\n%s",
                    connector_id,
                    traceback.format_exc(),
                )

            await asyncio.sleep(interval_seconds)

    async def run_once(self, connector_id: str) -> None:
        """Run a single crawl cycle and exit. Used by K8s CronJob pods."""
        _log.info("Crawler run_once starting: connector=%s", connector_id)
        try:
            instance = await load_connector_instance(connector_id)
            run_ctx = RunContext(connector_id=connector_id, run_type=self.RUN_TYPE)
            await self.crawl(instance, run_ctx, db_pool.get())
            _log.info(
                "Crawler run_once complete: %s processed=%d failed=%d",
                connector_id,
                run_ctx.items_processed,
                run_ctx.items_failed,
            )
        except ConnectorAuthError as exc:
            await mark_connector_error(connector_id, f"Auth failed: {exc}")
            _log.error("Crawler run_once auth failure on %s: %s", connector_id, exc)
            raise SystemExit(1)
        except ConnectorError as exc:
            _log.error("Crawler run_once connector error on %s: %s", connector_id, exc)
            raise SystemExit(1)
        except Exception:
            _log.error(
                "Crawler run_once unexpected error on %s:\n%s",
                connector_id,
                traceback.format_exc(),
            )
            raise SystemExit(1)
