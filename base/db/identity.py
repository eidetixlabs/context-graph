"""
base/db/identity.py — Shared identity resolution helper.

resolve_or_create_identity() looks up or creates a placeholder employee +
entity_registry row for a given external provider user ID, then ensures
baseline entity_aliases rows exist.

All operations run inside a single transaction on the supplied asyncpg
connection.
"""

from __future__ import annotations

import hashlib
import logging
import re

_log = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 0.85
PLACEHOLDER_ID_PREFIX = "emp:placeholder:"
PLACEHOLDER_EMAIL_SUFFIX = "@placeholder.internal"


def is_identity_stub(
    *,
    employee_id: str | None = None,
    email: str | None = None,
    auth_provider: str | None = None,
) -> bool:
    """Return True for connector-ingested people who must not be org accounts.

    Claimed stubs (invite accepted or an auth provider set) are real members.
    """
    if auth_provider:
        return False
    emp_id = (employee_id or "").strip()
    emp_email = (email or "").strip().lower()
    if emp_id.startswith(PLACEHOLDER_ID_PREFIX):
        return True
    return emp_email.endswith(PLACEHOLDER_EMAIL_SUFFIX)


def _slug(text: str) -> str:
    """Deterministic slug: lowercase, non-alphanum → '-', collapse runs, strip edges."""
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "unknown"


def _employee_entity_id(org_id: str, employee_id: str) -> str:
    """Return an employee-stable person entity ID that cannot collide by name."""
    digest = hashlib.sha256(employee_id.encode("utf-8")).hexdigest()[:12]
    return f"entity:person:{_slug(org_id)}:employee:{_slug(employee_id)}-{digest}"


def _placeholder_employee_id(provider: str, org_id: str, external_user_id: str) -> str:
    """Return collision-resistant placeholder ID for an external identity."""
    identity_key = "\x00".join((provider, org_id, external_user_id))
    digest = hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:12]
    return (
        f"emp:placeholder:{provider}:{_slug(org_id)}:"
        f"{_slug(external_user_id)}-{digest}"
    )


async def fuzzy_find_entity(
    conn,
    org_id: str,
    entity_type: str,
    name: str,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> tuple[str, float] | None:
    """
    Find an existing entity by fuzzy matching against aliases.

    Type-aware: only compares against entities of the same type. Uses
    rapidfuzz token_sort_ratio for robust matching of names with different
    word order (e.g. "John Smith" vs "Smith, John").

    Args:
        conn:        asyncpg connection
        org_id:      Organization scope
        entity_type: Entity type to match (person, service, etc.)
        name:        Name to search for
        threshold:   Minimum similarity ratio (0.0-1.0), default 0.85

    Returns:
        (entity_id, score) if a match is found above threshold, else None.
        Auto-merges (returns existing entity) — no SAME_AS queue.
    """
    if not name or not name.strip():
        return None

    name_lower = name.lower().strip()

    from rapidfuzz import fuzz  # lazy — connectors import this module but never call fuzzy_find_entity

    rows = await conn.fetch(
        """
        SELECT ea.entity_id, ea.alias_normalized, er.canonical_name
        FROM entity_aliases ea
        JOIN entity_registry er ON er.id = ea.entity_id AND er.org_id = ea.org_id
        WHERE ea.org_id = $1 AND er.entity_type = $2
        """,
        org_id,
        entity_type,
    )

    if not rows:
        return None

    best_match: tuple[str, float] | None = None

    for row in rows:
        alias = row["alias_normalized"] or ""
        canonical = (row["canonical_name"] or "").lower()

        score_alias = fuzz.token_sort_ratio(name_lower, alias) / 100.0
        score_canonical = fuzz.token_sort_ratio(name_lower, canonical) / 100.0
        score = max(score_alias, score_canonical)

        if score >= threshold:
            if best_match is None or score > best_match[1]:
                best_match = (row["entity_id"], score)

    if best_match:
        _log.info(
            "Fuzzy match: '%s' -> entity=%s (score=%.2f, type=%s, org=%s)",
            name,
            best_match[0],
            best_match[1],
            entity_type,
            org_id,
        )

    return best_match


async def search_entities_pg(
    pool,
    org_id: str,
    query: str,
    limit: int = 20,
) -> list[dict]:
    """
    Search entities by alias or canonical name using Postgres (no Neo4j).

    First tries prefix/ILIKE match on alias_normalized and canonical_name.
    Falls back to rapidfuzz token_sort_ratio if exact/prefix returns nothing.

    Args:
        pool:    asyncpg pool
        org_id:  Organization scope (empty string = match all orgs)
        query:   Search string
        limit:   Max results to return

    Returns:
        List of dicts with: id, canonical_name, entity_type, org_id, aliases
    """
    if not query or not query.strip():
        return []

    q_lower = query.lower().strip()
    q_prefix = q_lower + "%"
    org_filter = "AND er.org_id = $3" if org_id else ""
    params: list = [q_prefix, limit + 40]
    if org_id:
        params.append(org_id)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (er.id)
                er.id,
                er.canonical_name,
                er.entity_type,
                er.org_id,
                COALESCE(
                    (
                        SELECT json_agg(ea2.alias_raw ORDER BY ea2.created_at)
                        FROM entity_aliases ea2
                        WHERE ea2.entity_id = er.id AND ea2.org_id = er.org_id
                    ),
                    '[]'::json
                ) AS aliases
            FROM entity_registry er
            LEFT JOIN entity_aliases ea ON ea.entity_id = er.id AND ea.org_id = er.org_id
            WHERE (
                ea.alias_normalized ILIKE $1
                OR lower(er.canonical_name) ILIKE $1
            )
            {org_filter}
            AND er.status = 'active'
            ORDER BY er.id
            LIMIT $2
            """,
            *params,
        )

        if rows:
            results = [dict(r) for r in rows[:limit]]
            return results

        # Fallback: fuzzy match via rapidfuzz against all aliases in the org
        from rapidfuzz import fuzz
        fuzzy_params: list = [limit * 10]
        fuzzy_org_filter = "AND er.org_id = $2" if org_id else ""
        if org_id:
            fuzzy_params.append(org_id)

        all_rows = await conn.fetch(
            f"""
            SELECT
                er.id,
                er.canonical_name,
                er.entity_type,
                er.org_id,
                ea.alias_normalized,
                COALESCE(
                    (
                        SELECT json_agg(ea2.alias_raw ORDER BY ea2.created_at)
                        FROM entity_aliases ea2
                        WHERE ea2.entity_id = er.id AND ea2.org_id = er.org_id
                    ),
                    '[]'::json
                ) AS aliases
            FROM entity_registry er
            LEFT JOIN entity_aliases ea ON ea.entity_id = er.id AND ea.org_id = er.org_id
            WHERE er.status = 'active'
            {fuzzy_org_filter}
            LIMIT $1
            """,
            *fuzzy_params,
        )

        scored: list[tuple[float, dict]] = []
        seen_ids: set[str] = set()
        for row in all_rows:
            alias = row["alias_normalized"] or ""
            canonical = (row["canonical_name"] or "").lower()
            score = max(
                fuzz.token_sort_ratio(q_lower, alias) / 100.0,
                fuzz.token_sort_ratio(q_lower, canonical) / 100.0,
            )
            if score >= 0.6 and row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]


async def resolve_or_create_identity(
    conn,
    org_id: str,
    provider: str,
    external_user_id: str,
    name: str | None,
    email: str | None,
) -> tuple[str, str]:
    """Look up or create a placeholder employee + entity for an external user.

    Returns (employee_id, entity_id).
    All writes are idempotent and run inside a single transaction.

    Identity scoping: all lookups and placeholder IDs are scoped to org_id so
    two orgs using the same external provider cannot collide or share employees.
    """
    async with conn.transaction():
        org_slug = _slug(org_id)

        # 1. Try identity_mappings lookup — scoped to this org
        mapping_row = await conn.fetchrow(
            """
            SELECT employee_id FROM identity_mappings
            WHERE org_id = $1 AND provider = $2 AND external_id = $3
            """,
            org_id,
            provider,
            external_user_id,
        )

        alias_source = "onboarding"
        created_employee = False

        if mapping_row:
            employee_id = mapping_row["employee_id"]
            alias_source = "idp-sync"
        else:
            # 2. Prefer an existing employee with the same org-scoped email.
            # This avoids UNIQUE(org_id, email) collisions for connected Gmail
            # accounts that already have an employee row.
            existing_employee = None
            normalized_email = email.lower().strip() if email else None
            if normalized_email:
                existing_employee = await conn.fetchrow(
                    """
                    SELECT id
                    FROM employees
                    WHERE org_id = $1
                      AND lower(email) = $2
                    LIMIT 1
                    """,
                    org_id,
                    normalized_email,
                )

            if existing_employee:
                employee_id = existing_employee["id"]
                alias_source = "idp-sync"
            else:
                # 3. Create placeholder employee — ID includes org slug to
                #    prevent cross-tenant collision on the same external_user_id.
                ext_slug = _slug(external_user_id)
                employee_id = _placeholder_employee_id(provider, org_id, external_user_id)
                canonical_name = name or external_user_id
                # Placeholder email is org-scoped so UNIQUE(org_id, email) is
                # satisfied even when two orgs share external IDs.
                emp_email = (
                    normalized_email
                    or f"placeholder+{provider}+{org_slug}+{ext_slug}@placeholder.internal"
                )

                inserted = await conn.fetchrow(
                    """
                    INSERT INTO employees (
                        id, org_id, email, canonical_name, role, status,
                        deactivated_at, created_at, updated_at
                    )
                    VALUES (
                        $1, $2, $3, $4, 'member', 'deactivated',
                        now(), now(), now()
                    )
                    ON CONFLICT (org_id, email) DO UPDATE
                        SET updated_at = employees.updated_at
                    RETURNING id
                    """,
                    employee_id,
                    org_id,
                    emp_email,
                    canonical_name,
                )
                employee_id = inserted["id"]
                created_employee = True

        # 3. Claim the external mapping before entity creation. Re-read after
        # conflict so concurrent resolvers use the authoritative winner.
        if not mapping_row:
            await conn.execute(
                """
                INSERT INTO identity_mappings (org_id, employee_id, provider, external_id, created_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (org_id, provider, external_id) DO NOTHING
                """,
                org_id,
                employee_id,
                provider,
                external_user_id,
            )
            authoritative_mapping = await conn.fetchrow(
                """
                SELECT employee_id
                FROM identity_mappings
                WHERE org_id = $1 AND provider = $2 AND external_id = $3
                """,
                org_id,
                provider,
                external_user_id,
            )
            if authoritative_mapping:
                employee_id = authoritative_mapping["employee_id"]

        # 4. Look up entity_registry — scoped to this org+employee
        entity_row = await conn.fetchrow(
            """
            SELECT id FROM entity_registry
            WHERE employee_id = $1 AND org_id = $2
            """,
            employee_id,
            org_id,
        )

        if entity_row:
            entity_id = entity_row["id"]
        else:
            # 5. Create an employee-stable entity ID. Names are not unique:
            #    using their slug can bind different employees to one entity.
            canonical_name_for_entity = name or external_user_id
            entity_id = _employee_entity_id(org_id, employee_id)

            await conn.execute(
                """
                INSERT INTO entity_registry (id, org_id, entity_type, canonical_name, employee_id, status, created_at, updated_at)
                VALUES ($1, $2, 'person', $3, $4, 'active', now(), now())
                ON CONFLICT (id) DO UPDATE
                    SET employee_id = EXCLUDED.employee_id,
                        updated_at  = now()
                WHERE entity_registry.employee_id IS NULL
                """,
                entity_id,
                org_id,
                canonical_name_for_entity,
                employee_id,
            )
            # Re-fetch to tolerate concurrent creation for the same employee.
            entity_row = await conn.fetchrow(
                "SELECT id FROM entity_registry WHERE employee_id = $1 AND org_id = $2",
                employee_id,
                org_id,
            )
            if entity_row:
                entity_id = entity_row["id"]

        # 6. Upsert baseline aliases
        canonical_name_for_alias = name or external_user_id
        aliases: list[tuple[str, str]] = []

        # normalized canonical name
        norm_name = canonical_name_for_alias.lower().strip()
        if norm_name:
            aliases.append((norm_name, canonical_name_for_alias))

        # email local-part
        if email and "@" in email:
            local = email.split("@")[0].lower()
            if local and local != norm_name:
                aliases.append((local, email))

        # normalized external_user_id
        norm_ext = external_user_id.lower()
        if norm_ext and norm_ext not in {a for a, _ in aliases}:
            aliases.append((norm_ext, external_user_id))

        for alias_normalized, alias_raw in aliases:
            await conn.execute(
                """
                INSERT INTO entity_aliases (alias_normalized, entity_id, org_id, alias_raw, source, created_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (org_id, alias_normalized) DO NOTHING
                """,
                alias_normalized,
                entity_id,
                org_id,
                alias_raw,
                alias_source,
            )

        if created_employee:
            _log.debug(
                "Created placeholder employee %s for %s:%s in org %s",
                employee_id,
                provider,
                external_user_id,
                org_id,
            )

        return employee_id, entity_id
