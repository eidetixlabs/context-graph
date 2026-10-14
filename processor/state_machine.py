"""
processor/state_machine.py — Document processing state machine.

Wraps all Postgres state transitions for the `documents` table.
Uses SELECT FOR UPDATE SKIP LOCKED so multiple worker replicas can run
concurrently without double-processing documents.

State flow:
    raw  ──[claim_for_scoring]──►  scoring  ──[scored/skipped]──►  scored | skipped
    scored  ──[claim_for_extraction]──►  extracting
    extracting ──[durable payload]──► indexing ──[both sinks complete]──► extracted
    any  ──[failed]──►  failed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import asyncpg

from base.db.client import db_pool

_log = logging.getLogger(__name__)


class StaleDocumentWork(RuntimeError):
    """Raised when work no longer matches the document version/state it claimed."""


def _require_updated(status: str, doc_id: str, document_version: int) -> None:
    if status != "UPDATE 1":
        raise StaleDocumentWork(
            f"stale document work cancelled: {doc_id}@v{document_version}"
        )


# ---------------------------------------------------------------------------
# Document record (subset of the documents table)
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    id: str
    org_id: str
    source_provider: str
    source_scope: str
    source_ref: str
    title: str | None
    s3_key: str
    processing_state: str
    signal_score: float | None
    char_count: int | None
    current_version: int
    author_employee_id: str | None = None
    source_created_at: datetime | None = None
    parent_ref: str | None = None
    repo: str | None = None
    doc_type: str | None = None


# ---------------------------------------------------------------------------
# Claim documents (SELECT FOR UPDATE SKIP LOCKED)
# ---------------------------------------------------------------------------


async def claim_documents(
    source_provider: str,
    state: str,
    batch_size: int = 50,
    org_id: str | None = None,
) -> list[DocumentRecord]:
    """
    Atomically claim a batch of documents in `state` for processing.

    Uses SKIP LOCKED so parallel worker replicas don't compete.
    Documents are immediately transitioned to the corresponding in-progress
    state (raw→scoring, scored→extracting) to prevent re-claiming.

    Returns the claimed DocumentRecords.
    """
    pool = db_pool.get()

    next_state = _in_progress_state(state)

    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                """
                SELECT
                    id, org_id, source_provider, source_scope, source_ref,
                    title, s3_key, processing_state, signal_score, char_count,
                    current_version, author_employee_id, source_created_at, parent_ref,
                    repo, doc_type
                FROM documents
                WHERE source_provider = $1
                  AND processing_state = $2
                  AND ($3::text IS NULL OR org_id = $3)
                ORDER BY ingested_at
                LIMIT $4
                FOR UPDATE SKIP LOCKED
                """,
                source_provider,
                state,
                org_id,
                batch_size,
            )

            if not rows:
                return []

            doc_ids = [r["id"] for r in rows]

            await conn.execute(
                """
                UPDATE documents
                SET processing_state = $1, updated_at = now()
                WHERE id = ANY($2::text[])
                """,
                next_state,
                doc_ids,
            )

            _log.info(
                "Claimed %d documents for %s: %s → %s",
                len(doc_ids),
                source_provider,
                state,
                next_state,
            )

    return [_row_to_record(r) for r in rows]


def _in_progress_state(state: str) -> str:
    mapping = {
        "raw": "scoring",
        "scored": "extracting",
        "indexing": "indexing",
    }
    if state not in mapping:
        raise ValueError(f"Cannot claim documents in state '{state}'. Valid: {list(mapping)}")
    return mapping[state]


def _row_to_record(row: asyncpg.Record) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        org_id=row["org_id"],
        source_provider=row["source_provider"],
        source_scope=row["source_scope"],
        source_ref=row["source_ref"],
        title=row["title"],
        s3_key=row["s3_key"],
        processing_state=row["processing_state"],
        signal_score=row["signal_score"],
        char_count=row["char_count"],
        current_version=row["current_version"],
        author_employee_id=row["author_employee_id"],
        source_created_at=row["source_created_at"],
        parent_ref=row["parent_ref"],
        repo=row["repo"],
        doc_type=row["doc_type"],
    )


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def transition_scored(
    doc_id: str,
    document_version: int,
    score: float,
    threshold: float,
) -> str:
    """
    scoring → scored (score >= threshold) or skipped (score < threshold).
    Writes the signal_score either way.
    Returns the new state.
    """
    new_state = "scored" if score >= threshold else "skipped"
    pool = db_pool.get()
    status = await pool.execute(
        """
        UPDATE documents
        SET processing_state = $1,
            signal_score      = $2,
            updated_at        = now()
        WHERE id = $3
          AND current_version = $4
          AND processing_state = 'scoring'
        """,
        new_state,
        score,
        doc_id,
        document_version,
    )
    _require_updated(status, doc_id, document_version)
    _log.debug("Document %s: scored %.3f → %s", doc_id, score, new_state)
    return new_state


async def transition_extracted(
    doc_id: str,
    document_version: int,
    model: str,
) -> None:
    """indexing → extracted for the claimed document version."""
    pool = db_pool.get()
    status = await pool.execute(
        """
        UPDATE documents
        SET processing_state  = 'extracted',
            extraction_model  = $1,
            extraction_error  = NULL,
            updated_at        = now()
        WHERE id = $2
          AND current_version = $3
          AND processing_state = 'indexing'
        """,
        model,
        doc_id,
        document_version,
    )
    _require_updated(status, doc_id, document_version)
    _log.debug("Document %s: extracted with model=%s", doc_id, model)


async def transition_indexing(doc_id: str, document_version: int) -> None:
    """extracting → indexing after the canonical index payload is durable."""
    pool = db_pool.get()
    status = await pool.execute(
        """
        UPDATE documents
        SET processing_state = 'indexing',
            extraction_error = NULL,
            updated_at = now()
        WHERE id = $1
          AND current_version = $2
          AND processing_state = 'extracting'
        """,
        doc_id,
        document_version,
    )
    _require_updated(status, doc_id, document_version)
    _log.debug("Document %s: durable payload ready for indexing", doc_id)


async def transition_failed(
    doc_id: str,
    document_version: int,
    expected_state: str,
    error: str,
) -> None:
    """any → failed. Truncates error to 1 KB for storage."""
    pool = db_pool.get()
    truncated = error[:1024] if len(error) > 1024 else error
    status = await pool.execute(
        """
        UPDATE documents
        SET processing_state = 'failed',
            extraction_error = $1,
            updated_at       = now()
        WHERE id = $2
          AND current_version = $3
          AND processing_state = $4
        """,
        truncated,
        doc_id,
        document_version,
        expected_state,
    )
    _require_updated(status, doc_id, document_version)
    _log.warning("Document %s: failed — %s", doc_id, truncated)


async def reset_stale_in_progress(
    source_provider: str,
    stale_minutes: int = 30,
) -> int:
    """
    Reset documents stuck in 'scoring' or 'extracting' back to their previous
    state. This handles worker crashes — documents claimed but never completed
    will be re-queued on the next poll.

    Returns number of documents reset.
    """
    pool = db_pool.get()
    rows = await pool.fetch(
        """
        UPDATE documents
        SET processing_state = CASE
            WHEN processing_state = 'scoring'    THEN 'raw'
            WHEN processing_state = 'extracting' THEN 'scored'
        END,
        updated_at = now()
        WHERE source_provider = $1
          AND processing_state IN ('scoring', 'extracting')
          AND updated_at < now() - ($2 || ' minutes')::interval
        RETURNING id
        """,
        source_provider,
        str(stale_minutes),
    )
    count = len(rows)
    if count:
        _log.warning(
            "Reset %d stale in-progress documents for provider=%s (stale_minutes=%d)",
            count,
            source_provider,
            stale_minutes,
        )
    return count
