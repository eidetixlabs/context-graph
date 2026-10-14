"""Attributed repository context note writes with repository-scoped RBAC."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import uuid4

import asyncpg

from base.db.access import get_allowed_scopes

NOTE_CATEGORIES = frozenset(
    {
        "purpose",
        "architecture",
        "integration",
        "testing",
        "operations",
        "contributing",
        "insight",
    }
)
NOTE_TITLE_MAX_LENGTH = 64
NOTE_BODY_MAX_LENGTH = 10_000

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RepositoryNoteAccessError(PermissionError):
    """The authenticated employee cannot write to the repository scope."""


class RepositoryNoteConflictError(ValueError):
    """A requested supersession is invalid or no longer possible."""


def normalize_repository(owner: str, repo: str) -> tuple[str, str, str, str]:
    """Validate and canonicalize a case-insensitive GitHub repository identity."""
    if not isinstance(owner, str):
        raise ValueError("owner must be a string")
    if not isinstance(repo, str):
        raise ValueError("repo must be a string")

    normalized_owner = owner.strip().lower()
    normalized_repo = repo.strip().lower()
    if not normalized_owner or not _OWNER_RE.fullmatch(normalized_owner):
        raise ValueError("owner must be a valid GitHub owner")
    if not normalized_repo or not _REPO_RE.fullmatch(normalized_repo):
        raise ValueError("repo must be a valid GitHub repository name")

    repository = f"{normalized_owner}/{normalized_repo}"
    return (
        normalized_owner,
        normalized_repo,
        repository,
        f"scope:github:{repository}",
    )


def _normalize_note(category: str, body: str) -> tuple[str, str]:
    if not isinstance(category, str):
        raise ValueError("category must be a string")
    normalized_category = category.strip().lower()
    if len(normalized_category) > NOTE_TITLE_MAX_LENGTH:
        raise ValueError(
            f"category title must be at most {NOTE_TITLE_MAX_LENGTH} characters"
        )
    if normalized_category not in NOTE_CATEGORIES:
        allowed = ", ".join(sorted(NOTE_CATEGORIES))
        raise ValueError(f"category must be one of: {allowed}")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("body must be a non-empty string")
    normalized_body = body.strip()
    if len(normalized_body) > NOTE_BODY_MAX_LENGTH:
        raise ValueError(f"body must be at most {NOTE_BODY_MAX_LENGTH} characters")
    return normalized_category, normalized_body


def _updated_count(command_status: str) -> int:
    try:
        return int(command_status.rsplit(" ", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _serialize_created_at(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def _require_repository_access(
    pool: asyncpg.Pool,
    *,
    employee_id: str,
    repository: str,
    source_scope: str,
) -> str:
    if not employee_id:
        raise RepositoryNoteAccessError("an authenticated employee is required")
    allowed_scopes = await get_allowed_scopes(pool, employee_id)
    matching_scope = next(
        (
            scope
            for scope in (allowed_scopes or set())
            if str(scope).lower() == source_scope.lower()
        ),
        None,
    )
    if matching_scope is None:
        raise RepositoryNoteAccessError(
            f"employee does not have access to repository {repository}"
        )
    return str(matching_scope)


async def add_repository_context_note(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    employee_id: str,
    owner: str,
    repo: str,
    category: str,
    body: str,
) -> dict[str, Any]:
    """Append an attributed note and enqueue matching profiles atomically."""
    return await _write_repository_context_note(
        pool,
        org_id=org_id,
        employee_id=employee_id,
        owner=owner,
        repo=repo,
        category=category,
        body=body,
        supersedes_id=None,
    )


async def supersede_repository_context_note(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    employee_id: str,
    owner: str,
    repo: str,
    supersedes_id: str,
    category: str,
    body: str,
) -> dict[str, Any]:
    """Replace one active note in the same repository and retain its history."""
    if not isinstance(supersedes_id, str) or not supersedes_id.strip():
        raise ValueError("supersedes_id must be a non-empty string")
    return await _write_repository_context_note(
        pool,
        org_id=org_id,
        employee_id=employee_id,
        owner=owner,
        repo=repo,
        category=category,
        body=body,
        supersedes_id=supersedes_id.strip(),
    )


async def _write_repository_context_note(
    pool: asyncpg.Pool,
    *,
    org_id: str,
    employee_id: str,
    owner: str,
    repo: str,
    category: str,
    body: str,
    supersedes_id: str | None,
) -> dict[str, Any]:
    if not isinstance(org_id, str) or not org_id.strip():
        raise ValueError("org_id must be a non-empty string")

    _, _, repository, requested_scope = normalize_repository(owner, repo)
    normalized_category, normalized_body = _normalize_note(category, body)
    source_scope = await _require_repository_access(
        pool,
        employee_id=employee_id,
        repository=repository,
        source_scope=requested_scope,
    )

    note_id = f"repository-note:{uuid4()}"
    async with pool.acquire() as conn:
        async with conn.transaction():
            if supersedes_id is not None:
                prior = await conn.fetchrow(
                    """
                    SELECT id, org_id, source_scope, repository, status
                    FROM repository_context_notes
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    supersedes_id,
                )
                if prior is None:
                    raise RepositoryNoteConflictError(
                        "the note to supersede does not exist"
                    )
                same_repository = (
                    str(prior["org_id"]) == org_id
                    and str(prior["source_scope"]) == source_scope
                    and str(prior["repository"]) == repository
                )
                if not same_repository:
                    raise RepositoryNoteConflictError(
                        "a note can only supersede an active note in the same repository"
                    )
                if str(prior["status"]) != "active":
                    raise RepositoryNoteConflictError(
                        "the note to supersede is not active"
                    )
                superseded = await conn.execute(
                    """
                    UPDATE repository_context_notes
                    SET status = 'superseded'
                    WHERE id = $1
                      AND org_id = $2
                      AND source_scope = $3
                      AND repository = $4
                      AND status = 'active'
                    """,
                    supersedes_id,
                    org_id,
                    source_scope,
                    repository,
                )
                if _updated_count(superseded) != 1:
                    raise RepositoryNoteConflictError(
                        "the note to supersede is no longer active"
                    )

            note = await conn.fetchrow(
                """
                INSERT INTO repository_context_notes (
                    id,
                    org_id,
                    source_scope,
                    repository,
                    author_employee_id,
                    category,
                    body,
                    supersedes_id
                )
                SELECT $1, $2, $3, $4, $5, $6, $7, $8
                WHERE length($6) <= $9
                  AND length($7) <= $10
                RETURNING
                    id,
                    repository,
                    category,
                    body,
                    status,
                    supersedes_id,
                    author_employee_id,
                    created_at
                """,
                note_id,
                org_id,
                source_scope,
                repository,
                employee_id,
                normalized_category,
                normalized_body,
                supersedes_id,
                NOTE_TITLE_MAX_LENGTH,
                NOTE_BODY_MAX_LENGTH,
            )
            if note is None:
                raise ValueError("note exceeds server-side persistence limits")
            enqueued = await conn.execute(
                """
                UPDATE github_repository_profiles
                SET generation_state = 'pending',
                    trigger_reason = 'context_note',
                    generation_error = NULL
                WHERE org_id = $1
                  AND source_scope = $2
                  AND lower(repository) = lower($3)
                """,
                org_id,
                source_scope,
                repository,
            )

    return {
        "id": str(note["id"]),
        "repository": str(note["repository"]),
        "category": str(note["category"]),
        "body": str(note["body"]),
        "status": str(note["status"]),
        "supersedes_id": (
            str(note["supersedes_id"])
            if note["supersedes_id"] is not None
            else None
        ),
        "author_employee_id": str(note["author_employee_id"]),
        "created_at": _serialize_created_at(note["created_at"]),
        "enqueued_profiles": _updated_count(enqueued),
    }
