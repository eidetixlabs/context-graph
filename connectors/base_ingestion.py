"""
connectors/base_ingestion.py — Ingestion worker base class, S3 helpers, document PG helpers.

Depends on shared primitives from connectors.base_crawler.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import asyncpg
import boto3

from connectors.base_crawler import (
    AsyncpgConn,
    ConnectorAuthError,
    ConnectorError,
    ConnectorInstance,
    ConnectorRateLimitError,
    RunContext,
    _MAX_AUTH_FAILURES,
    backoff_sleep,
    load_connector_instance,
    mark_connector_error,
    read_checkpoint,
    write_checkpoint,
)
from base.db.client import db_pool

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S3 client — lazy singleton (tests patch via connectors.base_test.patch_s3_client)
# ---------------------------------------------------------------------------

_s3: Any = None

def _s3_client():
    global _s3
    if _s3 is None:
        kwargs: dict = {"region_name": os.environ.get("AWS_REGION", "us-east-1")}
        endpoint = os.environ.get("S3_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT")
        if endpoint:
            from botocore.config import Config

            kwargs["endpoint_url"] = endpoint
            kwargs["config"] = Config(s3={"addressing_style": "path"})
        _s3 = boto3.client("s3", **kwargs)
    return _s3


def _s3_bucket() -> str:
    bucket = os.environ.get("AWS_S3_BUCKET", "")
    if not bucket:
        raise ConnectorError("AWS_S3_BUCKET env var is not set")
    return bucket


def _s3_key_raw(org_id: str, provider: str, source_ref: str, version: int) -> str:
    return f"{org_id}/{provider}/raw/{source_ref}/v{version}.json"


def _s3_key_failed(org_id: str, provider: str, source_ref: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{org_id}/{provider}/failed/{source_ref}/{ts}.json"


def write_raw_s3(org_id: str, provider: str, source_ref: str, version: int, content: bytes) -> str:
    """Write raw document bytes to S3. Returns the S3 key."""
    key = _s3_key_raw(org_id, provider, source_ref, version)
    _s3_client().put_object(
        Bucket=_s3_bucket(),
        Key=key,
        Body=content,
        ContentType="application/json",
    )
    _log.debug("S3 write raw: %s", key)
    return key


def write_failed_s3(org_id: str, provider: str, source_ref: str, content: bytes) -> str:
    """Write raw downloaded content to failed/ when parsing fails. Returns S3 key."""
    key = _s3_key_failed(org_id, provider, source_ref)
    _s3_client().put_object(
        Bucket=_s3_bucket(),
        Key=key,
        Body=content,
        ContentType="application/json",
    )
    _log.debug("S3 write failed: %s", key)
    return key


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def check_content_hash(
    conn: AsyncpgConn,
    org_id: str,
    provider: str,
    source_ref: str,
    new_hash: str,
) -> tuple[str | None, int, bool]:
    row = await conn.fetchrow(
        """
        SELECT id, current_version, content_hash
        FROM documents
        WHERE org_id = $1 AND source_provider = $2 AND source_ref = $3
        """,
        org_id,
        provider,
        source_ref,
    )
    if row is None:
        return None, 0, False
    is_unchanged = row["content_hash"] == new_hash
    return row["id"], row["current_version"], is_unchanged


async def upsert_document(
    conn: AsyncpgConn,
    *,
    doc_id: str,
    org_id: str,
    provider: str,
    source_scope: str,
    source_ref: str,
    title: str | None,
    content_hash: str,
    char_count: int | None,
    s3_key: str,
    version: int,
    source_created_at: datetime | None,
    source_updated_at: datetime | None,
    author_employee_id: str | None = None,
    author_entity_id: str | None = None,
    parent_ref: str | None = None,
) -> None:
    async with conn.transaction():
        existing = await conn.fetchrow(
            "SELECT id, org_id FROM documents WHERE id = $1", doc_id
        )
        existing_org = existing.get("org_id") if existing is not None else None
        if existing is not None and existing_org is not None and str(existing_org) != org_id:
            raise ValueError("document id is already owned by another organization")

        if existing is None:
            await conn.execute(
                """
                INSERT INTO documents (
                    id, org_id, source_provider, source_scope, source_ref,
                    title, author_employee_id, author_entity_id, content_hash, char_count, s3_key,
                    current_version, processing_state, processing_version,
                    source_created_at, source_updated_at,
                    ingested_at, created_at, updated_at, parent_ref
                ) VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10, $11,
                    $12, 'raw', 1,
                    $13, $14,
                    now(), now(), now(), $15
                )
                """,
                doc_id,
                org_id,
                provider,
                source_scope,
                source_ref,
                title,
                author_employee_id,
                author_entity_id,
                content_hash,
                char_count,
                s3_key,
                version,
                source_created_at,
                source_updated_at,
                parent_ref,
            )
        else:
            await conn.execute(
                """
                UPDATE documents SET
                    title               = $2,
                    author_employee_id  = COALESCE($3, author_employee_id),
                    author_entity_id    = COALESCE($4, author_entity_id),
                    content_hash        = $5,
                    char_count          = $6,
                    s3_key              = $7,
                    current_version     = $8,
                    processing_state    = 'raw',
                    source_updated_at   = $9,
                    parent_ref          = $10,
                    updated_at          = now()
                WHERE id = $1 AND org_id = $11
                """,
                doc_id,
                title,
                author_employee_id,
                author_entity_id,
                content_hash,
                char_count,
                s3_key,
                version,
                source_updated_at,
                parent_ref,
                org_id,
            )

        await conn.execute(
            """
            INSERT INTO document_versions (doc_id, version, content_hash, s3_key, char_count, ingested_at)
            VALUES ($1, $2, $3, $4, $5, now())
            ON CONFLICT (doc_id, version) DO NOTHING
            """,
            doc_id,
            version,
            content_hash,
            s3_key,
            char_count,
        )


# ---------------------------------------------------------------------------
# BaseIngestionWorker
# ---------------------------------------------------------------------------


class BaseIngestionWorker(ABC):
    """Base class for all ingestion workers."""

    RUN_TYPE = "ingest"

    @abstractmethod
    async def ingest(
        self,
        instance: ConnectorInstance,
        run_ctx: RunContext,
        pool: asyncpg.Pool,
        last_cursor: datetime | None,
    ) -> None:
        """Provider-specific ingestion logic."""

    async def advance_checkpoint(self, connector_id: str, cursor: datetime) -> None:
        await write_checkpoint(connector_id, self.RUN_TYPE, cursor)

    def handle_parse_failure(
        self,
        org_id: str,
        provider: str,
        source_ref: str,
        raw_bytes: bytes,
        exc: Exception,
    ) -> str:
        _log.error(
            "Parse failure for %s/%s/%s: %s\n%s",
            org_id,
            provider,
            source_ref,
            exc,
            traceback.format_exc(),
        )
        return write_failed_s3(org_id, provider, source_ref, raw_bytes)

    def write_raw(
        self, org_id: str, provider: str, source_ref: str, version: int, content: bytes
    ) -> str:
        return write_raw_s3(org_id, provider, source_ref, version, content)

    async def upsert_document(self, conn: AsyncpgConn, **kwargs: Any) -> None:
        await upsert_document(conn, **kwargs)

    async def check_content_hash(
        self,
        conn: AsyncpgConn,
        org_id: str,
        provider: str,
        source_ref: str,
        new_hash: str,
    ) -> tuple[str | None, int, bool]:
        return await check_content_hash(conn, org_id, provider, source_ref, new_hash)

    async def run(self, connector_id: str, poll_interval_seconds: float = 300.0) -> None:
        _log.info(
            "Ingestion worker starting: connector=%s poll=%ss",
            connector_id,
            poll_interval_seconds,
        )
        consecutive_auth_failures = 0

        while True:
            try:
                instance = await load_connector_instance(connector_id)
                last_cursor = await read_checkpoint(connector_id, self.RUN_TYPE)
                run_ctx = RunContext(connector_id=connector_id, run_type=self.RUN_TYPE)

                _log.info(
                    "Ingest run starting: %s last_cursor=%s",
                    connector_id,
                    last_cursor,
                )
                await self.ingest(instance, run_ctx, db_pool.get(), last_cursor)
                consecutive_auth_failures = 0
                _log.info(
                    "Ingest run complete: %s processed=%d failed=%d",
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
                        "Connector %s disabled after %d auth failures.",
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
                    "Unexpected error in ingestion worker %s:\n%s",
                    connector_id,
                    traceback.format_exc(),
                )

            await asyncio.sleep(poll_interval_seconds)

    async def run_once(self, connector_id: str) -> None:
        """Run a single ingest cycle and exit. Used by K8s CronJob pods."""
        _log.info("Ingestion run_once starting: connector=%s", connector_id)
        try:
            instance = await load_connector_instance(connector_id)
            last_cursor = await read_checkpoint(connector_id, self.RUN_TYPE)
            run_ctx = RunContext(connector_id=connector_id, run_type=self.RUN_TYPE)
            _log.info("Ingest run_once: %s last_cursor=%s", connector_id, last_cursor)
            await self.ingest(instance, run_ctx, db_pool.get(), last_cursor)
            _log.info(
                "Ingest run_once complete: %s processed=%d failed=%d",
                connector_id,
                run_ctx.items_processed,
                run_ctx.items_failed,
            )
        except ConnectorAuthError as exc:
            await mark_connector_error(connector_id, f"Auth failed: {exc}")
            _log.error("Ingest run_once auth failure on %s: %s", connector_id, exc)
            raise SystemExit(1)
        except ConnectorError as exc:
            _log.error("Ingest run_once connector error on %s: %s", connector_id, exc)
            raise SystemExit(1)
        except Exception:
            _log.error(
                "Ingest run_once unexpected error on %s:\n%s",
                connector_id,
                traceback.format_exc(),
            )
            raise SystemExit(1)
