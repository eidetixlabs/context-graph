"""
processor/resolution/persistence.py — Persist resolution data to Postgres.

Public functions:
  persist_entities_from_extraction — write every Entity node from the final
      LLM ExtractionResult to entity_registry/entity_aliases.
  persist_trace_new_entities_from_map — write is_new=True trace-resolved entities.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from processor.resolution.entity_ids import build_entity_id
from processor.resolution.models import ResolutionMap
from processor.schemas.extraction import ExtractionResult

_log = logging.getLogger(__name__)
_FUZZY_RECONCILE_THRESHOLD = 0.70


def _normalize_alias(alias: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", alias.lower())).strip()


def _surface_forms_for_entity(
    resolution_map: ResolutionMap | None,
    entity_id: str,
) -> list[str]:
    """Every surface form the resolver bound to ``entity_id``.

    Pulls the canonical name + variants the resolution map collected for the
    entity, plus every mention whose resolution entry points at it. This is how
    subsumed fragments ("Riya"), role-decorated forms ("Riya (Data)") and
    abbreviations ("Riya K.") reach entity_aliases — the LLM Entity node only
    carries a subset of them.
    """
    if resolution_map is None:
        return []
    forms: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        normalized = _normalize_alias(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            forms.append(value)

    for entity in resolution_map.resolved_entities:
        if entity.entity_id == entity_id:
            _add(entity.canonical_name)
            for variant in entity.variants:
                _add(variant)
    for mention, entry in resolution_map.entries.items():
        if entry.entity_id == entity_id:
            _add(mention)
    return forms


def _node_to_raw(node: Any) -> dict[str, Any]:
    if isinstance(node, dict):
        return node
    if hasattr(node, "model_dump"):
        dumped = node.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    try:
        coerced = dict(node)
        return coerced if isinstance(coerced, dict) else {}
    except Exception:
        return {}


def _apply_entity_id_rewrites_to_extraction(
    extraction_result: ExtractionResult,
    entity_id_rewrites: dict[str, str],
) -> None:
    if not entity_id_rewrites:
        return

    for node in extraction_result.nodes:
        if isinstance(node, dict):
            if node.get("type") != "Entity":
                continue
            old_id = str(node.get("id") or "").strip()
            new_id = entity_id_rewrites.get(old_id)
            if new_id and new_id != old_id:
                node["id"] = new_id
            continue

        if getattr(node, "type", None) != "Entity":
            continue
        old_id = str(getattr(node, "id", "") or "").strip()
        new_id = entity_id_rewrites.get(old_id)
        if new_id and new_id != old_id:
            setattr(node, "id", new_id)

    for edge in extraction_result.edges:
        if isinstance(edge, dict):
            from_id = str(edge.get("from_id") or "").strip()
            to_id = str(edge.get("to_id") or "").strip()
            new_from = entity_id_rewrites.get(from_id)
            new_to = entity_id_rewrites.get(to_id)
            if new_from and new_from != from_id:
                edge["from_id"] = new_from
            if new_to and new_to != to_id:
                edge["to_id"] = new_to
            continue

        from_id = str(getattr(edge, "from_id", "") or "").strip()
        to_id = str(getattr(edge, "to_id", "") or "").strip()
        new_from = entity_id_rewrites.get(from_id)
        new_to = entity_id_rewrites.get(to_id)
        if new_from and new_from != from_id:
            setattr(edge, "from_id", new_from)
        if new_to and new_to != to_id:
            setattr(edge, "to_id", new_to)


def _entity_aliases(raw: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    canonical_name = str(raw.get("canonical_name") or raw.get("label") or "").strip()
    aliases_raw = raw.get("aliases")

    if isinstance(aliases_raw, str) and aliases_raw.strip():
        try:
            parsed = json.loads(aliases_raw)
        except json.JSONDecodeError:
            parsed = [aliases_raw]
        if isinstance(parsed, list):
            aliases.extend(str(alias) for alias in parsed if str(alias).strip())
    elif isinstance(aliases_raw, list):
        aliases.extend(str(alias) for alias in aliases_raw if str(alias).strip())

    if canonical_name:
        aliases.insert(0, canonical_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias = alias.strip()
        alias_norm = _normalize_alias(alias)
        if not alias_norm or alias_norm in seen:
            continue
        seen.add(alias_norm)
        deduped.append(alias)
    return deduped


def _entity_from_resolution_map(
    resolution_map: ResolutionMap | None,
    *,
    raw_entity_id: str,
    canonical_name: str,
) -> str | None:
    if resolution_map is None:
        return None

    canonical_norm = _normalize_alias(canonical_name)
    for entity in resolution_map.resolved_entities:
        if raw_entity_id and entity.entity_id == raw_entity_id:
            return entity.entity_id
        if canonical_norm and _normalize_alias(entity.canonical_name) == canonical_norm:
            return entity.entity_id

    for mention, entry in resolution_map.entries.items():
        if not entry.entity_id:
            continue
        if raw_entity_id and entry.entity_id == raw_entity_id:
            return entry.entity_id
        if canonical_norm and _normalize_alias(mention) == canonical_norm:
            return entry.entity_id

    return None


async def _entity_from_alias_lookup(
    conn: Any,
    *,
    org_id: str,
    entity_type: str,
    aliases: list[str],
) -> str | None:
    alias_norms = [
        alias_norm
        for alias_norm in (_normalize_alias(alias) for alias in aliases)
        if alias_norm
    ]
    if not alias_norms:
        return None

    row = await conn.fetchrow(
        """
        SELECT er.id AS entity_id
        FROM entity_aliases ea
        JOIN entity_registry er
          ON er.id = ea.entity_id
         AND er.org_id = ea.org_id
         AND er.status = 'active'
        WHERE ea.org_id = $1
          AND er.entity_type = $2
          AND ea.alias_normalized = ANY($3::text[])
        LIMIT 1
        """,
        org_id,
        entity_type,
        alias_norms,
    )
    return str(row["entity_id"]) if row else None


async def _entity_from_fuzzy_lookup(
    conn: Any,
    *,
    org_id: str,
    entity_type: str,
    canonical_name: str,
) -> str | None:
    mention_norm = _normalize_alias(canonical_name)
    if not mention_norm:
        return None

    rows = await conn.fetch(
        """
        SELECT
            er.id,
            er.canonical_name,
            COALESCE(
                array_agg(ea.alias_normalized) FILTER (WHERE ea.alias_normalized IS NOT NULL),
                ARRAY[]::text[]
            ) AS aliases
        FROM entity_registry er
        LEFT JOIN entity_aliases ea
          ON ea.entity_id = er.id
         AND ea.org_id = er.org_id
        WHERE er.org_id = $1
          AND er.entity_type = $2
          AND er.status = 'active'
        GROUP BY er.id, er.canonical_name
        """,
        org_id,
        entity_type,
    )
    if not rows:
        return None

    from rapidfuzz import fuzz

    best_entity_id: str | None = None
    best_score = 0.0
    for row in rows:
        canonical = _normalize_alias(str(row["canonical_name"] or ""))
        score = fuzz.token_sort_ratio(mention_norm, canonical) / 100.0 if canonical else 0.0
        for alias in row["aliases"] or []:
            alias_norm = str(alias or "").strip()
            if not alias_norm:
                continue
            alias_score = fuzz.token_sort_ratio(mention_norm, alias_norm) / 100.0
            if alias_score > score:
                score = alias_score
        if score > best_score:
            best_score = score
            best_entity_id = str(row["id"])

    if best_entity_id and best_score >= _FUZZY_RECONCILE_THRESHOLD:
        return best_entity_id
    return None


async def persist_entities_from_extraction(
    doc: Any,
    extraction_result: ExtractionResult,
    conn: Any,
    resolution_map: ResolutionMap | None = None,
) -> None:
    """
    Persist every final LLM Entity node, independent of MENTIONS edges.

    The LLM ExtractionResult is the source of truth after Phase 3. Postgres
    registration is idempotent so existing entities are left intact and missing
    final entities are created before Neo4j relationship writes.
    """
    entities: list[dict[str, Any]] = []
    for node in extraction_result.nodes:
        raw = _node_to_raw(node)
        if raw.get("type") != "Entity":
            continue
        entity_type = str(raw.get("entity_type") or "").strip()
        canonical_name = str(raw.get("canonical_name") or raw.get("label") or "").strip()
        if not entity_type or not canonical_name:
            _log.warning("Skipping invalid Entity node for doc=%s: %s", doc.id, raw)
            continue
        entities.append(raw)

    if not entities:
        _log.debug("No final Entity nodes to persist for doc=%s", doc.id)
        return

    reconciled: dict[str, dict[str, Any]] = {}
    entity_id_rewrites: dict[str, str] = {}
    try:
        async with conn.transaction():
            for raw in entities:
                raw_entity_id = str(raw.get("id") or "").strip()
                entity_type = str(raw["entity_type"])
                canonical_name = str(raw.get("canonical_name") or raw.get("label"))
                aliases = _entity_aliases(raw)

                matched_entity_id = _entity_from_resolution_map(
                    resolution_map,
                    raw_entity_id=raw_entity_id,
                    canonical_name=canonical_name,
                )
                if matched_entity_id is None:
                    matched_entity_id = await _entity_from_alias_lookup(
                        conn,
                        org_id=doc.org_id,
                        entity_type=entity_type,
                        aliases=aliases,
                    )
                if matched_entity_id is None:
                    matched_entity_id = await _entity_from_fuzzy_lookup(
                        conn,
                        org_id=doc.org_id,
                        entity_type=entity_type,
                        canonical_name=canonical_name,
                    )

                entity_id = matched_entity_id or build_entity_id(
                    entity_type=entity_type,
                    org_id=doc.org_id,
                    canonical_name=canonical_name,
                )
                if raw_entity_id:
                    previous = entity_id_rewrites.get(raw_entity_id)
                    if previous and previous != entity_id:
                        _log.warning(
                            "Conflicting entity rewrite ignored: raw_id=%s old=%s new=%s doc=%s",
                            raw_entity_id,
                            previous,
                            entity_id,
                            doc.id,
                        )
                    else:
                        entity_id_rewrites[raw_entity_id] = entity_id

                payload = reconciled.get(entity_id)
                if payload is None:
                    payload = {
                        "entity_type": entity_type,
                        "canonical_name": canonical_name,
                        "aliases": [],
                        "_seen_aliases": set(),
                    }
                    reconciled[entity_id] = payload

                for alias_raw in aliases:
                    alias_norm = _normalize_alias(alias_raw)
                    if not alias_norm or alias_norm in payload["_seen_aliases"]:
                        continue
                    payload["_seen_aliases"].add(alias_norm)
                    payload["aliases"].append(alias_raw)

            # Fold in resolver-discovered surface forms (subsumed fragments,
            # role-decorated and abbreviated mentions) so every matched alias is
            # persisted, not just the ones the final LLM node happened to emit.
            for entity_id, payload in reconciled.items():
                for surface in _surface_forms_for_entity(resolution_map, entity_id):
                    alias_norm = _normalize_alias(surface)
                    if not alias_norm or alias_norm in payload["_seen_aliases"]:
                        continue
                    payload["_seen_aliases"].add(alias_norm)
                    payload["aliases"].append(surface)

            for entity_id, payload in reconciled.items():
                entity_type = str(payload["entity_type"])
                canonical_name = str(payload["canonical_name"])

                await conn.execute(
                    """
                    INSERT INTO entity_registry
                        (id, org_id, entity_type, canonical_name, status)
                    VALUES ($1, $2, $3, $4, 'active')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    entity_id,
                    doc.org_id,
                    entity_type,
                    canonical_name,
                )

                for alias_raw in payload["aliases"]:
                    alias_norm = _normalize_alias(alias_raw)
                    if not alias_norm:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO entity_aliases
                            (alias_normalized, entity_id, org_id, alias_raw, source)
                        VALUES ($1, $2, $3, $4, 'extraction')
                        ON CONFLICT (org_id, alias_normalized) DO NOTHING
                        """,
                        alias_norm,
                        entity_id,
                        doc.org_id,
                        alias_raw,
                    )

        _apply_entity_id_rewrites_to_extraction(extraction_result, entity_id_rewrites)
        _log.info(
            "Entity persistence complete from extraction_result: doc=%s persisted=%d",
            doc.id,
            len(reconciled),
        )
    except Exception as exc:
        _log.warning(
            "Entity persistence from extraction_result failed: doc=%s error=%s",
            doc.id,
            exc,
        )


async def persist_trace_new_entities_from_map(
    trace_id: str,
    org_id: str,
    resolution_map: ResolutionMap,
    conn: Any,
) -> None:
    """
    Persist trace-resolved entities where is_new=True into Postgres.

    Persists all newly created entities (including author entities).
    """
    new_entities_to_persist = [e for e in resolution_map.resolved_entities if e.is_new]
    if not new_entities_to_persist:
        _log.debug("No new trace entities to persist: trace_id=%s", trace_id)
        return

    persisted = 0
    try:
        async with conn.transaction():
            for entity in new_entities_to_persist:
                await conn.execute(
                    """
                    INSERT INTO entity_registry
                        (id, org_id, entity_type, canonical_name, status)
                    VALUES ($1, $2, $3, $4, 'active')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    entity.entity_id,
                    org_id,
                    entity.entity_type,
                    entity.canonical_name,
                )

                aliases_to_insert = list(entity.variants)
                if entity.canonical_name not in aliases_to_insert:
                    aliases_to_insert.insert(0, entity.canonical_name)

                for alias_raw in aliases_to_insert:
                    alias_norm = _normalize_alias(alias_raw)
                    if not alias_norm:
                        continue
                    await conn.execute(
                        """
                        INSERT INTO entity_aliases
                            (alias_normalized, entity_id, org_id, alias_raw, source)
                        VALUES ($1, $2, $3, $4, 'extraction')
                        ON CONFLICT (org_id, alias_normalized) DO NOTHING
                        """,
                        alias_norm,
                        entity.entity_id,
                        org_id,
                        alias_raw,
                    )

                entity.is_new = False
                if entity.entity_id not in resolution_map.newly_persisted_ids:
                    resolution_map.newly_persisted_ids.append(entity.entity_id)
                resolution_map.add_audit(
                    entity.canonical_name,
                    f"persisted trace entity {entity.entity_id}",
                )
                persisted += 1

        _update_counts(resolution_map)
        _log.info(
            "Trace entity persistence complete: trace_id=%s org_id=%s persisted=%d",
            trace_id,
            org_id,
            persisted,
        )
    except Exception as exc:
        _log.warning(
            "Trace entity persistence failed: trace_id=%s org_id=%s error=%s",
            trace_id,
            org_id,
            exc,
        )


def _update_counts(resolution_map: ResolutionMap) -> None:
    total = len(resolution_map.entries)
    resolved = sum(1 for e in resolution_map.entries.values() if e.entity_id is not None)
    new = sum(
        1 for entity in resolution_map.resolved_entities if entity.is_new
    )
    failed = total - resolved

    resolution_map.counts = {
        "total": total,
        "resolved": resolved,
        "new": new,
        "failed": max(0, failed),
    }
