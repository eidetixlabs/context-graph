"""Idempotent worker for queued GitHub repository profile synthesis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import asyncpg

from base.db.client import db_pool
from processor.agents.repository_profile import synthesize_repository_profile
from processor.config import config
from processor.schemas.repository_profile import RepoProfileV1
from processor.storage.s3 import read_json_object, write_json_object
from processor.tools.eidetix import close_shared_client

_log = logging.getLogger(__name__)

_PROFILE_PROVIDER = "github-profile"


@dataclass(frozen=True)
class RepositoryProfileQueueRow:
    id: str
    org_id: str
    connector_id: str
    source_scope: str
    repository: str
    ref: str
    commit_sha: str
    evidence_hash: str
    evidence_s3_key: str
    trigger_reason: str
    release_metadata: dict[str, Any]
    claimed_at: datetime

    @classmethod
    def from_record(cls, row: Any) -> RepositoryProfileQueueRow:
        release_metadata = row["release_metadata"]
        if isinstance(release_metadata, str):
            try:
                release_metadata = json.loads(release_metadata)
            except ValueError:
                release_metadata = {}
        return cls(
            id=str(row["id"]),
            org_id=str(row["org_id"]),
            connector_id=str(row["connector_id"]),
            source_scope=str(row["source_scope"]),
            repository=str(row["repository"]),
            ref=str(row["ref"]),
            commit_sha=str(row["commit_sha"]),
            evidence_hash=str(row["evidence_hash"]),
            evidence_s3_key=str(row["evidence_s3_key"]),
            trigger_reason=str(row["trigger_reason"]),
            release_metadata=(
                dict(release_metadata) if isinstance(release_metadata, dict) else {}
            ),
            claimed_at=row["updated_at"],
        )


async def claim_repository_profiles(
    pool: asyncpg.Pool,
    *,
    batch_size: int = 10,
) -> list[RepositoryProfileQueueRow]:
    """Atomically claim pending/retry/stale rows across worker replicas."""
    async with pool.acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            WITH candidates AS (
                SELECT id
                FROM github_repository_profiles
                WHERE generation_state IN ('pending', 'failed')
                   OR (
                       generation_state = 'in_progress'
                       AND updated_at < now() - interval '30 minutes'
                   )
                ORDER BY updated_at, id
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE github_repository_profiles AS profile
            SET generation_state = 'in_progress',
                generation_error = NULL,
                updated_at = now()
            FROM candidates
            WHERE profile.id = candidates.id
            RETURNING
                profile.id,
                profile.org_id,
                profile.connector_id,
                profile.source_scope,
                profile.repository,
                profile.ref,
                profile.commit_sha,
                profile.evidence_hash,
                profile.evidence_s3_key,
                profile.trigger_reason,
                profile.release_metadata,
                profile.updated_at
            """,
            batch_size,
        )
    return [RepositoryProfileQueueRow.from_record(row) for row in rows]


async def load_repository_context_notes(
    conn: Any,
    row: RepositoryProfileQueueRow,
) -> list[dict[str, Any]]:
    """Load active attributed notes when migration 018 is available."""
    table_name = await conn.fetchval(
        "SELECT to_regclass('public.repository_context_notes')"
    )
    if table_name is None:
        return []
    records = await conn.fetch(
        """
        SELECT
            id,
            category,
            body,
            author_employee_id,
            created_at,
            updated_at
        FROM repository_context_notes
        WHERE org_id = $1
          AND source_scope = $2
          AND lower(repository) = lower($3)
          AND status = 'active'
        ORDER BY created_at, id
        """,
        row.org_id,
        row.source_scope,
        row.repository,
    )
    return [
        {
            "id": str(record["id"]),
            "category": str(record["category"]),
            "body": str(record["body"]),
            "author_employee_id": str(record["author_employee_id"]),
            "created_at": _isoformat(record["created_at"]),
            "updated_at": _isoformat(record["updated_at"]),
        }
        for record in records
    ]


async def process_repository_profile(
    pool: asyncpg.Pool,
    row: RepositoryProfileQueueRow,
) -> bool:
    """Generate and persist one claimed row, guarding terminal state by hash."""
    try:
        evidence = await read_json_object(row.evidence_s3_key)
        if str(evidence.get("evidence_hash") or "") != row.evidence_hash:
            raise ValueError(
                "queued evidence hash does not match the stored evidence bundle"
            )
        if (
            str(evidence.get("repository") or "") != row.repository
            or str(evidence.get("ref") or "") != row.ref
            or str(evidence.get("commit_sha") or "") != row.commit_sha
        ):
            raise ValueError("queued repository identity does not match evidence")

        async with pool.acquire() as conn:
            notes = await load_repository_context_notes(conn, row)
        profile = await synthesize_repository_profile(evidence, notes)
        profile_doc_id = await persist_repository_profile(pool, row, profile)
        if profile_doc_id is None:
            _log.info(
                "Repository profile synthesis cancelled after queue refresh: %s",
                row.id,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - queue rows must reach a retry state
        _log.error(
            "Repository profile failed for %s@%s:\n%s",
            row.repository,
            row.ref,
            traceback.format_exc(),
        )
        error = str(exc)[:2048]
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE github_repository_profiles
                SET generation_state = 'failed',
                    generation_error = $2,
                    updated_at = now()
                WHERE id = $1
                  AND generation_state = 'in_progress'
                  AND evidence_hash = $3
                  AND commit_sha = $4
                  AND evidence_s3_key = $5
                  AND updated_at = $6
                """,
                row.id,
                error,
                row.evidence_hash,
                row.commit_sha,
                row.evidence_s3_key,
                row.claimed_at,
            )
        return False


async def persist_repository_profile(
    pool: asyncpg.Pool,
    row: RepositoryProfileQueueRow,
    profile: RepoProfileV1,
) -> str | None:
    """Atomically publish a profile and complete its unchanged queue claim."""
    payload = profile.model_dump(mode="json")
    payload_bytes = _canonical_json_bytes(payload)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    source_ref = f"{row.repository}/_profile/{row.ref}"
    doc_id = f"doc:{_PROFILE_PROVIDER}:{source_ref}"

    source_updated_at = _parse_timestamp(profile.freshness.repository_updated_at)
    s3_key = f"{row.org_id}/{_PROFILE_PROVIDER}/raw/{payload_hash}.json"

    async with pool.acquire() as conn, conn.transaction():
        queue_claim = await conn.fetchrow(
            """
            SELECT id, generation_state, evidence_hash, commit_sha, evidence_s3_key,
                   updated_at
            FROM github_repository_profiles
            WHERE id = $1
            FOR UPDATE
            """,
            row.id,
        )
        if not _queue_claim_matches(queue_claim, row):
            return None

        # Serializes both the existing-row and first-insert cases without a new
        # schema column. The lock lives only for this transaction.
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            doc_id,
        )
        existing = await conn.fetchrow(
            """
            SELECT id, current_version, content_hash, s3_key
            FROM documents
            WHERE org_id = $1
              AND source_provider = $2
              AND source_ref = $3
            FOR UPDATE
            """,
            row.org_id,
            _PROFILE_PROVIDER,
            source_ref,
        )
        if existing is not None and str(existing["content_hash"]) == payload_hash:
            existing_doc_id = str(existing["id"])
            await _complete_queue_claim(conn, row, existing_doc_id)
            return existing_doc_id

        version = (
            int(existing["current_version"]) if existing is not None else 0
        ) + 1
        # The object name is immutable and independent of version allocation;
        # a transaction rollback can leave only a harmless content-addressed object.
        await write_json_object(s3_key, payload_bytes)
        await conn.execute(
            """
            INSERT INTO documents (
                id,
                org_id,
                source_provider,
                source_scope,
                source_ref,
                title,
                content_hash,
                char_count,
                s3_key,
                current_version,
                processing_state,
                processing_version,
                source_updated_at,
                ingested_at,
                created_at,
                updated_at,
                parent_ref
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                'raw', 1, $11, now(), now(), now(), $12
            )
            ON CONFLICT (id) DO UPDATE
            SET source_scope = EXCLUDED.source_scope,
                title = EXCLUDED.title,
                content_hash = EXCLUDED.content_hash,
                char_count = EXCLUDED.char_count,
                s3_key = EXCLUDED.s3_key,
                current_version = EXCLUDED.current_version,
                processing_state = 'raw',
                source_updated_at = EXCLUDED.source_updated_at,
                ingested_at = now(),
                parent_ref = EXCLUDED.parent_ref,
                updated_at = now()
            """,
            doc_id,
            row.org_id,
            _PROFILE_PROVIDER,
            row.source_scope,
            source_ref,
            f"{row.repository} repository profile ({row.ref})",
            payload_hash,
            len(payload_bytes),
            s3_key,
            version,
            source_updated_at,
            row.repository,
        )
        await conn.execute(
            """
            INSERT INTO document_versions (
                doc_id,
                version,
                content_hash,
                s3_key,
                char_count,
                ingested_at
            )
            VALUES ($1, $2, $3, $4, $5, now())
            """,
            doc_id,
            version,
            payload_hash,
            s3_key,
            len(payload_bytes),
        )
        await _complete_queue_claim(conn, row, doc_id)
        return doc_id


def _queue_claim_matches(claim: Any, row: RepositoryProfileQueueRow) -> bool:
    return bool(
        claim is not None
        and str(claim["generation_state"]) == "in_progress"
        and str(claim["evidence_hash"]) == row.evidence_hash
        and str(claim["commit_sha"]) == row.commit_sha
        and str(claim["evidence_s3_key"]) == row.evidence_s3_key
        and claim["updated_at"] == row.claimed_at
    )


async def _complete_queue_claim(
    conn: Any,
    row: RepositoryProfileQueueRow,
    profile_doc_id: str,
) -> None:
    status = await conn.execute(
        """
        UPDATE github_repository_profiles
        SET generation_state = 'completed',
            generation_error = NULL,
            profile_doc_id = $2,
            last_generated_at = now(),
            updated_at = now()
        WHERE id = $1
          AND generation_state = 'in_progress'
          AND evidence_hash = $3
          AND commit_sha = $4
          AND evidence_s3_key = $5
          AND updated_at = $6
        """,
        row.id,
        profile_doc_id,
        row.evidence_hash,
        row.commit_sha,
        row.evidence_s3_key,
        row.claimed_at,
    )
    if status != "UPDATE 1":
        raise RuntimeError("repository profile queue claim changed during publication")


class RepositoryProfileWorker:
    """Run profile synthesis followed by profile document indexing."""

    def __init__(self, index_worker=None) -> None:
        if index_worker is None:
            from processor.providers.github_profile import (  # noqa: PLC0415
                GithubProfileProcessorWorker,
            )

            index_worker = GithubProfileProcessorWorker()
        self.index_worker = index_worker

    async def _run_cycle(self) -> tuple[int, int]:
        pool = db_pool.get()
        rows = await claim_repository_profiles(
            pool,
            batch_size=config.extraction_batch_size,
        )
        completed = 0
        for row in rows:
            if await process_repository_profile(pool, row):
                completed += 1
        index_stats = await self.index_worker._run_cycle()
        _log.info(
            "Repository profile cycle complete: claimed=%d completed=%d failed=%d "
            "index_stats=%s",
            len(rows),
            completed,
            len(rows) - completed,
            index_stats,
        )
        return len(rows), completed

    async def run(
        self,
        poll_interval: float | None = None,
        run_once: bool = False,
    ) -> None:
        interval = poll_interval or config.poll_interval_seconds
        stop_event = asyncio.Event()

        def _handle_signal() -> None:
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle_signal)
            except (NotImplementedError, RuntimeError):
                pass

        await db_pool.open()
        try:
            while not stop_event.is_set():
                try:
                    await self._run_cycle()
                except Exception:  # noqa: BLE001 - keep the polling worker alive
                    _log.error(
                        "Unexpected repository profile cycle error:\n%s",
                        traceback.format_exc(),
                    )
                if run_once or stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass
        finally:
            await close_shared_client()
            await db_pool.close()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _isoformat(raw: Any) -> str:
    if isinstance(raw, datetime):
        return raw.astimezone(timezone.utc).isoformat()
    return str(raw)
