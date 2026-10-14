"""Portal document ingestion using the existing connector storage primitives."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from base.tools.repository_context_notes import normalize_repository
from connectors.base_ingestion import sha256_hex, upsert_document, write_raw_s3


_SOURCE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


class DocumentConflictError(ValueError):
    """Raised when a stable source reference is already owned elsewhere."""


class DocumentNotFoundError(LookupError):
    """Raised when a portal document is missing for the caller's org."""


class DocumentNotRetryableError(ValueError):
    """Raised when a portal document is not in a terminal retryable state."""

    def __init__(self, processing_state: str) -> None:
        self.processing_state = processing_state
        super().__init__(
            f"document is {document_status(processing_state)} and cannot be retried"
        )


RETRYABLE_PROCESSING_STATES = frozenset({"failed", "skipped"})
_DOCUMENT_ID_RE = re.compile(r"^doc:[A-Za-z0-9][A-Za-z0-9:._-]{0,298}$")


def portal_document_id(org_id: str, source_ref: str) -> str:
    return f"doc:portal:{org_id}:{source_ref}"


def portal_source_scope(org_id: str, repo: str | None) -> tuple[str | None, str]:
    if not repo:
        return None, f"scope:portal:{org_id}"
    repository, source_scope = normalize_repo(repo)
    return repository, source_scope


def normalize_repo(repo: str) -> tuple[str, str]:
    if not isinstance(repo, str):
        raise ValueError("repo must be a string in owner/name form")
    parts = repo.strip().split("/")
    if len(parts) != 2:
        raise ValueError("repo must be in owner/name form")
    _, _, repository, source_scope = normalize_repository(parts[0], parts[1])
    return repository, source_scope


def normalize_document_id(document_id: str) -> str:
    value = (document_id or "").strip()
    if not _DOCUMENT_ID_RE.fullmatch(value):
        raise ValueError("document_id is invalid")
    return value


def normalize_source_ref(source_ref: str | None) -> str:
    if source_ref is None:
        return str(uuid.uuid4())
    normalized = source_ref.strip()
    if not _SOURCE_REF_RE.fullmatch(normalized):
        raise ValueError(
            "source_ref must be 1-200 characters using letters, numbers, '.', '_', ':', or '-'"
        )
    return normalized


def build_portal_payload(
    *,
    repo: str,
    doc_type: str,
    title: str,
    content: str,
    source_ref: str,
) -> bytes:
    return json.dumps(
        {
            "schema_version": "portal_document.v1",
            "repo": repo,
            "doc_type": doc_type,
            "title": title,
            "content": content,
            "source_ref": source_ref,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def document_status(processing_state: str) -> str:
    if processing_state == "extracted":
        return "ready"
    if processing_state in {"skipped", "failed"}:
        return processing_state
    return "processing"


async def persist_portal_document(
    *,
    pool: Any,
    org_id: str,
    employee_id: str | None,
    repo: str | None,
    doc_type: str,
    title: str,
    content: str,
    source_ref: str | None,
) -> dict[str, Any]:
    """Version and enqueue a repo-scoped document through connector primitives."""
    repository, source_scope = portal_source_scope(org_id, repo)
    stable_ref = normalize_source_ref(source_ref)
    doc_id = portal_document_id(org_id, stable_ref)
    raw_bytes = build_portal_payload(
        repo=repository or "",
        doc_type=doc_type,
        title=title,
        content=content,
        source_ref=stable_ref,
    )
    content_hash = sha256_hex(raw_bytes)
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn, conn.transaction():
        existing, unchanged = await _locked_portal_document(
            conn,
            org_id=org_id,
            stable_ref=stable_ref,
            repository=repository,
            content_hash=content_hash,
        )
        if unchanged and existing is not None:
            return _unchanged_result(
                existing,
                repository=repository,
                doc_type=doc_type,
                title=title,
                stable_ref=stable_ref,
            )
        version = int(existing["current_version"]) + 1 if existing is not None else 1

    s3_key = await asyncio.to_thread(
        write_raw_s3,
        org_id,
        "portal",
        stable_ref,
        version,
        raw_bytes,
    )

    async with pool.acquire() as conn, conn.transaction():
        existing, unchanged = await _locked_portal_document(
            conn,
            org_id=org_id,
            stable_ref=stable_ref,
            repository=repository,
            content_hash=content_hash,
        )
        if unchanged and existing is not None:
            return _unchanged_result(
                existing,
                repository=repository,
                doc_type=doc_type,
                title=title,
                stable_ref=stable_ref,
            )
        version = int(existing["current_version"]) + 1 if existing is not None else 1
        await _write_portal_document(
            conn,
            org_id=org_id,
            employee_id=employee_id,
            doc_id=doc_id,
            repository=repository,
            source_scope=source_scope,
            stable_ref=stable_ref,
            title=title,
            content=content,
            content_hash=content_hash,
            s3_key=s3_key,
            version=version,
            doc_type=doc_type,
            now=now,
            created=existing is None,
        )

    return {
        "id": doc_id,
        "repo": repository,
        "doc_type": doc_type,
        "title": title,
        "source_ref": stable_ref,
        "version": version,
        "processing_state": "raw",
        "status": "processing",
        "unchanged": False,
    }


async def retry_portal_document(
    *,
    pool: Any,
    org_id: str,
    document_id: str,
) -> dict[str, Any]:
    """Re-queue a failed or skipped portal document from its stored raw payload."""
    doc_id = normalize_document_id(document_id)
    retryable = list(sorted(RETRYABLE_PROCESSING_STATES))
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE documents
            SET processing_state = 'raw',
                extraction_error = NULL,
                signal_score = NULL,
                updated_at = now()
            WHERE id = $1
              AND org_id = $2
              AND source_provider = 'portal'
              AND processing_state = ANY($3::text[])
            RETURNING id, repo, doc_type, title, source_ref,
                      current_version, processing_state
            """,
            doc_id,
            org_id,
            retryable,
        )
        if row is not None:
            return {
                "id": str(row["id"]),
                "repo": row["repo"],
                "doc_type": row["doc_type"],
                "title": row["title"],
                "source_ref": row["source_ref"],
                "version": int(row["current_version"]),
                "processing_state": str(row["processing_state"]),
                "status": document_status(str(row["processing_state"])),
                "unchanged": False,
            }
        existing = await conn.fetchrow(
            """
            SELECT processing_state
            FROM documents
            WHERE id = $1
              AND org_id = $2
              AND source_provider = 'portal'
            """,
            doc_id,
            org_id,
        )
    if existing is None:
        raise DocumentNotFoundError("document not found")
    raise DocumentNotRetryableError(str(existing["processing_state"]))


def _unchanged_result(
    existing: Any,
    *,
    repository: str | None,
    doc_type: str,
    title: str,
    stable_ref: str,
) -> dict[str, Any]:
    state = str(existing["processing_state"])
    return {
        "id": str(existing["id"]),
        "repo": repository,
        "doc_type": doc_type,
        "title": title,
        "source_ref": stable_ref,
        "version": int(existing["current_version"]),
        "processing_state": state,
        "status": document_status(state),
        "unchanged": True,
    }


async def _locked_portal_document(
    conn: Any,
    *,
    org_id: str,
    stable_ref: str,
    repository: str | None,
    content_hash: str,
) -> tuple[Any, bool]:
    await conn.fetchval(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"portal-document:{org_id}:{stable_ref}",
    )
    existing = await conn.fetchrow(
        """
        SELECT id, org_id, current_version, content_hash, processing_state, repo
        FROM documents
        WHERE org_id = $1
          AND source_provider = 'portal'
          AND source_ref = $2
        FOR UPDATE
        """,
        org_id,
        stable_ref,
    )
    existing_repo = None
    if existing is not None:
        try:
            existing_repo = existing["repo"]
        except KeyError:
            existing_repo = None
    if (
        existing_repo is not None
        and repository is not None
        and str(existing_repo) != repository
    ):
        raise DocumentConflictError(
            "source_ref is already used by a different repository"
        )
    unchanged = (
        existing is not None and str(existing["content_hash"]) == content_hash
    )
    return existing, unchanged


async def _write_portal_document(
    conn: Any,
    *,
    org_id: str,
    employee_id: str | None,
    doc_id: str,
    repository: str | None,
    source_scope: str,
    stable_ref: str,
    title: str,
    content: str,
    content_hash: str,
    s3_key: str,
    version: int,
    doc_type: str,
    now: datetime,
    created: bool,
) -> None:
    if repository:
        await conn.execute(
            """
            INSERT INTO access_scopes
                (id, org_id, source_provider, source_ref, source_label)
            VALUES ($1, $2, 'github', $3, $3)
            ON CONFLICT (org_id, source_provider, source_ref) DO NOTHING
            """,
            source_scope,
            org_id,
            repository,
        )
    try:
        await upsert_document(
            conn,
            doc_id=doc_id,
            org_id=org_id,
            provider="portal",
            source_scope=source_scope,
            source_ref=stable_ref,
            title=title,
            content_hash=content_hash,
            char_count=len(content),
            s3_key=s3_key,
            version=version,
            source_created_at=now if created else None,
            source_updated_at=now,
            author_employee_id=employee_id,
        )
    except ValueError as exc:
        if "another organization" in str(exc):
            raise DocumentConflictError(str(exc)) from exc
        raise
    await conn.execute(
        """
        UPDATE documents
        SET repo = $2,
            doc_type = $3,
            source_scope = $4,
            updated_at = now()
        WHERE id = $1
          AND org_id = $5
        """,
        doc_id,
        repository,
        doc_type,
        source_scope,
        org_id,
    )
