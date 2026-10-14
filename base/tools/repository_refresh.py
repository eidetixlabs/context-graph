"""Durable GitHub repository evidence refresh requests."""

from __future__ import annotations

import json
from typing import Any

from base.tools.repository_context_notes import normalize_repository


def _updated_count(command_status: str) -> int:
    try:
        return int(command_status.rsplit(" ", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _connector_config(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def connector_config_includes_repository(
    config: dict[str, Any],
    repository: str,
) -> bool:
    """Confirm a canonical repository belongs to the connector selection."""
    owner, _ = repository.split("/", 1)
    configured_repo = str(config.get("repo") or "").strip().lower()
    if configured_repo:
        return configured_repo == repository
    configured_owner = str(config.get("owner") or "").strip().lower()
    return bool(configured_owner) and configured_owner == owner


def normalize_repository_name(repository: str) -> tuple[str, str]:
    if not isinstance(repository, str):
        raise ValueError("repository must be a string in owner/repo form")
    parts = repository.strip().split("/")
    if len(parts) != 2:
        raise ValueError("repository must be in owner/repo form")
    _, _, normalized, scope = normalize_repository(parts[0], parts[1])
    return normalized, scope


async def request_repository_refresh(
    *,
    pool: Any,
    org_id: str,
    repository: str,
    reason: str,
    connector_id: str | None = None,
    allowed_scopes: list[str] | None = None,
    event_id: str | None = None,
    force_generation: bool = False,
) -> dict[str, Any]:
    """Queue evidence recollection for an accessible configured repository."""
    normalized_repository, requested_scope = normalize_repository_name(repository)
    normalized_reason = str(reason or "").strip()[:128]
    if not normalized_reason:
        raise ValueError("refresh reason must be non-empty")
    if connector_id is not None:
        if not isinstance(connector_id, str) or not connector_id.strip():
            raise ValueError("connector_id must be a non-empty string when supplied")
        connector_id = connector_id.strip()
    normalized_event_id = str(event_id).strip()[:255] if event_id else None

    if allowed_scopes is not None and not any(
        str(scope).lower() == requested_scope.lower() for scope in allowed_scopes
    ):
        raise PermissionError(
            f"caller does not have access to repository {normalized_repository}"
        )

    async with pool.acquire() as conn:
        connectors = await conn.fetch(
            """
            SELECT id, org_id, config
            FROM connector_instances
            WHERE org_id = $1
              AND provider = 'github'
              AND status = 'active'
              AND ($2::text IS NULL OR id = $2)
            ORDER BY id
            """,
            org_id,
            connector_id,
        )
        if not connectors:
            raise LookupError("no active GitHub connector found")

        configured_connector_ids: list[str] = []
        candidates: list[tuple[str, list[Any]]] = []
        for connector in connectors:
            candidate_id = str(connector["id"])
            config = _connector_config(connector["config"])
            if not connector_config_includes_repository(
                config,
                normalized_repository,
            ):
                continue
            configured_connector_ids.append(candidate_id)
            profiles = await conn.fetch(
                """
                SELECT id, ref, source_scope
                FROM github_repository_profiles
                WHERE connector_id = $1
                  AND lower(repository) = lower($2)
                  AND org_id = $3
                ORDER BY ref, id
                """,
                candidate_id,
                normalized_repository,
                org_id,
            )
            if not profiles:
                continue
            if any(
                str(profile["source_scope"]).lower() != requested_scope.lower()
                for profile in profiles
            ):
                raise PermissionError(
                    "repository profile scope does not match its identity"
                )
            candidates.append((candidate_id, list(profiles)))

        if not candidates:
            if connector_id and not configured_connector_ids:
                raise LookupError(
                    f"repository {normalized_repository} is not configured for "
                    f"connector {connector_id}"
                )
            raise LookupError(
                f"no repository profile found for {normalized_repository}"
            )
        if len(candidates) > 1:
            matching_ids = ", ".join(candidate_id for candidate_id, _ in candidates)
            raise LookupError(
                f"repository {normalized_repository} matches multiple active GitHub "
                f"connectors ({matching_ids}); specify connector_id"
            )

        selected_connector_id, profiles = candidates[0]

        profile_ids = [str(profile["id"]) for profile in profiles]
        status = await conn.execute(
            """
            UPDATE github_repository_profiles
            SET refresh_requested_at = now(),
                refresh_reason = $3,
                refresh_event_id = COALESCE($4, refresh_event_id),
                refresh_force_generation = (
                    refresh_force_generation OR $5
                ),
                updated_at = now()
            WHERE id = ANY($1::text[])
              AND org_id = $2
              AND ($4::text IS NULL OR refresh_event_id IS DISTINCT FROM $4)
            """,
            profile_ids,
            org_id,
            normalized_reason,
            normalized_event_id,
            bool(force_generation),
        )

    return {
        "accepted": True,
        "connector_id": selected_connector_id,
        "repository": normalized_repository,
        "reason": normalized_reason,
        "requested_profiles": _updated_count(status),
    }
