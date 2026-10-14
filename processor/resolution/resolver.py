"""
processor/resolution/resolver.py — Deterministic entity resolver.

Priority chain per mention:
  1. Email exact match        → confidence 1.0
  2. Alias exact match        → confidence 0.95
  3. Role + org match         → confidence 0.85
  4. Fuzzy candidate shortlist
  5. LLM disambiguation       → whenever fuzzy shortlist is non-empty
  6. No match                 → is_new=True, entity deferred

All Postgres data is bulk-fetched once per resolve_mentions call and matched
in memory, avoiding per-mention round-trips.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from base.db.identity import FUZZY_MATCH_THRESHOLD
from processor.resolution.entity_ids import build_entity_id
from processor.resolution.entity_quality import (
    MIN_MENTION_CONFIDENCE,
    derive_person_alias_base,
    is_junk_mention,
    is_subsumed_fragment,
    passes_quality_gate,
)
from processor.resolution.models import (
    MentionCandidate,
    ResolvedEntity,
    ResolutionMap,
    ResolutionMapEntry,
)

_log = logging.getLogger(__name__)

_LLM_FUZZY_THRESHOLD = min(FUZZY_MATCH_THRESHOLD, 0.60)
_LLM_FUZZY_MAX_CANDIDATES = 20


@dataclass
class _Candidate:
    entity_id: str
    confidence: float
    resolution_method: str


@dataclass
class _OrgData:
    """Bulk data pre-fetched from Postgres once per resolve_mentions call."""

    # email_lower → entity_id  (only active entities)
    email_index: dict[str, str] = field(default_factory=dict)

    # (alias_normalized, entity_type) → list[entity_id]  (only active entities)
    alias_index: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    # entity_id → role_value_lower  (only active persons)
    role_index: dict[str, str] = field(default_factory=dict)

    # entity_type → list[(entity_id, aliases: tuple[str, ...], canonical_name_lower)]
    # One entry per entity (not per alias) — used for in-memory fuzzy matching.
    fuzzy_index: dict[str, list[tuple[str, tuple[str, ...], str]]] = field(default_factory=dict)

    # entity_id → canonical_name (display case) — used to format the known-entity
    # mapping block injected into the Phase 1 LLM prompt.
    canonical_index: dict[str, str] = field(default_factory=dict)


# TTL cache for _fetch_org_data — org entity data changes rarely within a processing batch.
# Key: org_id  Value: (_OrgData, fetched_at_monotonic)
_ORG_DATA_CACHE: dict[str, tuple[_OrgData, float]] = {}
_ORG_DATA_CACHE_TTL_SECONDS: float = 60.0


async def _fetch_org_data(conn: Any, org_id: str, *, consistent: bool = False) -> _OrgData:
    """
    Bulk-fetch all entity data for org_id needed by the resolution priority chain.

    Four queries replace the per-mention round-trips previously fired by
    _resolve_single (email lookup, alias lookup, role lookup, fuzzy_find_entity).

    Results are cached in-process for _ORG_DATA_CACHE_TTL_SECONDS (default 60 s)
    so repeated calls within a batch (same org, multiple docs) pay only one round-trip.
    """
    now = time.monotonic()
    if not consistent:
        cached = _ORG_DATA_CACHE.get(org_id)
        if cached is not None:
            org_data, fetched_at = cached
            if now - fetched_at < _ORG_DATA_CACHE_TTL_SECONDS:
                _log.debug("_fetch_org_data cache hit: org=%s age=%.1fs", org_id, now - fetched_at)
                return org_data
            _log.debug("_fetch_org_data cache expired: org=%s age=%.1fs", org_id, now - fetched_at)

    data = _OrgData()

    # 1. Email index
    email_rows = await conn.fetch(
        """
        SELECT lower(em.value) AS email_lower, em.entity_id
        FROM entity_metadata em
        JOIN entity_registry er ON er.id = em.entity_id
        WHERE em.key = 'email'
          AND er.org_id = $1
          AND er.status = 'active'
        """,
        org_id,
    )
    for row in email_rows:
        data.email_index[row["email_lower"]] = row["entity_id"]

    # 2. Alias index + fuzzy index (one query covers both)
    alias_rows = await conn.fetch(
        """
        SELECT ea.alias_normalized, ea.entity_id, er.entity_type, er.canonical_name
        FROM entity_aliases ea
        JOIN entity_registry er ON er.id = ea.entity_id AND er.org_id = ea.org_id
        WHERE ea.org_id = $1
          AND er.status = 'active'
        """,
        org_id,
    )
    # Build alias_index and accumulate per-entity alias sets for fuzzy_index.
    # _fuzzy_build: entity_id → (entity_type, canonical_lower, set_of_alias_norms)
    _fuzzy_build: dict[str, tuple[str, str, set[str]]] = {}
    for row in alias_rows:
        key = (row["alias_normalized"], row["entity_type"])
        data.alias_index.setdefault(key, []).append(row["entity_id"])

        eid = row["entity_id"]
        if eid not in _fuzzy_build:
            _fuzzy_build[eid] = (
                row["entity_type"],
                (row["canonical_name"] or "").lower(),
                set(),
            )
            if row["canonical_name"]:
                data.canonical_index[eid] = row["canonical_name"]
        alias_norm = row["alias_normalized"] or ""
        if alias_norm:
            _fuzzy_build[eid][2].add(alias_norm)

    for eid, (etype, canonical, aliases) in _fuzzy_build.items():
        data.fuzzy_index.setdefault(etype, []).append(
            (eid, tuple(aliases), canonical)
        )

    # 3. Role index (persons only)
    role_rows = await conn.fetch(
        """
        SELECT em.entity_id, em.value AS role_value
        FROM entity_metadata em
        JOIN entity_registry er ON er.id = em.entity_id
        WHERE em.key = 'role'
          AND er.org_id = $1
          AND er.status = 'active'
          AND er.entity_type = 'person'
        """,
        org_id,
    )
    for row in role_rows:
        data.role_index[row["entity_id"]] = row["role_value"].lower()

    if not consistent:
        _ORG_DATA_CACHE[org_id] = (data, time.monotonic())
        _log.debug(
            "_fetch_org_data cache populated: org=%s emails=%d aliases=%d roles=%d",
            org_id,
            len(data.email_index),
            len(data.alias_index),
            len(data.role_index),
        )
    return data


def _fuzzy_candidates_in_memory(
    org_data: _OrgData,
    entity_type: str,
    mention_lower: str,
    threshold: float,
    limit: int,
) -> list[tuple[str, float]]:
    """Return top fuzzy candidates for a mention from in-memory org data."""
    from rapidfuzz import fuzz

    entities = org_data.fuzzy_index.get(entity_type, [])
    candidates: list[tuple[str, float]] = []

    # Punctuation-stripped form so abbreviations ("Riya K." -> "riya k") score
    # against canonical names ("riya kapoor") instead of being dragged down by the
    # trailing period. token_set_ratio surfaces shared-token overlaps (shared first
    # name) that token_sort_ratio alone misses.
    mention_norm = _normalize(mention_lower)

    def _best(probe: str, target: str) -> float:
        if not target:
            return 0.0
        return max(
            fuzz.token_sort_ratio(probe, target),
            fuzz.token_set_ratio(probe, target),
        ) / 100.0

    for entity_id, aliases, canonical in entities:
        score = max(_best(mention_lower, canonical), _best(mention_norm, canonical))
        for alias in aliases:
            s = max(_best(mention_lower, alias), _best(mention_norm, alias))
            if s > score:
                score = s
        if score >= threshold:
            candidates.append((entity_id, score))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:limit]


def _index_alias_in_org_data(
    org_data: _OrgData,
    *,
    entity_type: str,
    entity_id: str,
    alias_norm: str,
    canonical_name: str | None = None,
) -> None:
    if not alias_norm:
        return

    alias_key = (alias_norm, entity_type)
    existing_ids = org_data.alias_index.setdefault(alias_key, [])
    if entity_id not in existing_ids:
        existing_ids.append(entity_id)

    fuzzy_entities = org_data.fuzzy_index.setdefault(entity_type, [])
    for idx, (existing_id, aliases, canonical) in enumerate(fuzzy_entities):
        if existing_id != entity_id:
            continue
        if alias_norm in aliases:
            return
        updated_aliases = tuple(sorted({*aliases, alias_norm}))
        fuzzy_entities[idx] = (existing_id, updated_aliases, canonical or alias_norm)
        return

    canonical_lower = (canonical_name or alias_norm).lower().strip() or alias_norm
    fuzzy_entities.append((entity_id, (alias_norm,), canonical_lower))


async def _maybe_write_extraction_alias(
    conn: Any,
    *,
    org_id: str,
    entity_type: str,
    entity_id: str,
    mention_text: str,
    org_data: _OrgData,
) -> None:
    alias_norm = _normalize(mention_text)
    if not alias_norm:
        return

    await conn.execute(
        """
        INSERT INTO entity_aliases
            (alias_normalized, entity_id, org_id, alias_raw, source)
        VALUES ($1, $2, $3, $4, 'extraction')
        ON CONFLICT (org_id, alias_normalized) DO NOTHING
        """,
        alias_norm,
        entity_id,
        org_id,
        mention_text,
    )
    existing = org_data.alias_index.get((alias_norm, entity_type), [])
    if not existing or entity_id in existing:
        _index_alias_in_org_data(
            org_data,
            entity_type=entity_type,
            entity_id=entity_id,
            alias_norm=alias_norm,
            canonical_name=mention_text,
        )


async def _register_surface_forms(
    conn: Any,
    *,
    org_id: str,
    entity_type: str,
    entity_id: str,
    forms: list[str],
    org_data: _OrgData,
    resolution_map: ResolutionMap,
    existing_entity_ids: set[str] | None,
    reserved_keys: set[str],
) -> None:
    """Bind every LLM-known surface form of an entity to it.

    Seeds the in-memory alias/fuzzy indexes (so later mentions in this run bind
    deterministically), records a resolution-map entry (so persistence writes the
    alias to ``entity_aliases``), and — for entities already in the registry —
    writes the alias immediately (FK-safe). Forms that collide with another
    primary mention are skipped so we never shadow an independent entity.
    """
    for form in forms:
        norm = _normalize(form)
        if not norm or norm in reserved_keys:
            continue

        existing = org_data.alias_index.get((norm, entity_type), [])
        if existing and entity_id not in existing:
            # Already owned by a different entity — don't steal the alias.
            continue

        _index_alias_in_org_data(
            org_data,
            entity_type=entity_type,
            entity_id=entity_id,
            alias_norm=norm,
            canonical_name=form,
        )

        if form not in resolution_map.entries:
            resolution_map.entries[form] = ResolutionMapEntry(
                mention_text=form,
                entity_id=entity_id,
                resolution_method="llm_alias",
                confidence=0.9,
            )

        if existing_entity_ids and entity_id in existing_entity_ids:
            await _maybe_write_extraction_alias(
                conn,
                org_id=org_id,
                entity_type=entity_type,
                entity_id=entity_id,
                mention_text=form,
                org_data=org_data,
            )


_MENTION_TYPE_TO_ENTITY_TYPE: dict[str, str] = {
    "person": "person",
    "service": "service",
    "org": "organization",
    "repo": "repository",
    "tool": "tool",
    "object": "object",
    "location": "location",
    "event": "event",
    "role_reference": "person",
    "pronoun": "person",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip()


def _collect_existing_entity_ids(org_data: _OrgData) -> set[str]:
    """Entity IDs already committed to the registry (snapshot at fetch time).

    Used to keep alias writeback FK-safe: aliases for entities minted during the
    current run are persisted downstream (after the registry row is inserted),
    never mid-resolution.
    """
    ids: set[str] = set()
    ids.update(org_data.email_index.values())
    for id_list in org_data.alias_index.values():
        ids.update(id_list)
    ids.update(org_data.role_index.keys())
    for entries in org_data.fuzzy_index.values():
        for entity_id, _aliases, _canonical in entries:
            ids.add(entity_id)
    return ids


def _find_entity_for_alias(
    resolution_map: ResolutionMap,
    org_data: _OrgData,
    base: str,
    entity_type: str,
) -> str | None:
    """Resolve the entity a decorated/subsumed surface form belongs to.

    Matches the form's bare ``base`` name against (1) the live alias index and
    (2) the surface forms already resolved this run, including the subsumed
    relationship where ``base`` is the first token / prefix of a fuller name
    ("Riya" -> "Riya Kapoor").
    """
    base_norm = _normalize(base)
    if not base_norm:
        return None

    hits = org_data.alias_index.get((base_norm, entity_type), [])
    if hits:
        return hits[0]

    best: str | None = None
    for mention_text, entry in resolution_map.entries.items():
        if entry.entity_id is None:
            continue
        m_norm = _normalize(mention_text)
        if not m_norm:
            continue
        if m_norm == base_norm:
            return entry.entity_id
        if m_norm.startswith(base_norm + " ") or m_norm.split(" ", 1)[0] == base_norm:
            best = entry.entity_id
    return best


async def _attach_alias_candidates(
    alias_candidates: list[tuple[MentionCandidate, str]],
    *,
    resolution_map: ResolutionMap,
    org_data: _OrgData,
    org_id: str,
    conn: Any,
    existing_entity_ids: set[str],
) -> None:
    """Bind decorated/subsumed surface forms as aliases of their resolved entity.

    Every form is recorded as a resolution entry (folded into the entity's
    variants downstream so it persists), and — when the target entity already
    exists in the registry — written straight into ``entity_aliases`` so the
    next document hits alias-exact instead of re-running fuzzy/LLM.
    """
    for mention, base in alias_candidates:
        text = mention.mention_text
        if text in resolution_map.entries:
            continue
        entity_type = _MENTION_TYPE_TO_ENTITY_TYPE.get(mention.mention_type, "object")
        target_id = _find_entity_for_alias(resolution_map, org_data, base, entity_type)
        if target_id is None:
            _log.debug(
                "No resolved entity for alias surface form %r (base=%r)", text, base
            )
            continue

        resolution_map.entries[text] = ResolutionMapEntry(
            mention_text=text,
            entity_id=target_id,
            resolution_method="alias_attached",
            confidence=mention.confidence,
        )
        _index_alias_in_org_data(
            org_data,
            entity_type=entity_type,
            entity_id=target_id,
            alias_norm=_normalize(text),
            canonical_name=text,
        )
        if target_id in existing_entity_ids:
            await _maybe_write_extraction_alias(
                conn,
                org_id=org_id,
                entity_type=entity_type,
                entity_id=target_id,
                mention_text=text,
                org_data=org_data,
            )


async def resolve_mentions(
    mentions: list[MentionCandidate],
    org_id: str,
    conn: Any,
) -> ResolutionMap:
    """
    Resolve all mention candidates into a ResolutionMap.

    Bulk-fetches all entity data for the org in 3 queries up front, then
    resolves each unique mention entirely in memory. Only LLM disambiguation
    (rare) still hits the DB.

    Skips pronouns and role_reference types — those are handled by
    processor.resolution.pronouns after this step.

    Args:
        mentions: Extracted mention candidates.
        org_id:   Organization scope.
        conn:     asyncpg connection.

    Returns:
        ResolutionMap with entries populated for all non-pronoun mentions.
    """
    org_data = await _fetch_org_data(conn, org_id, consistent=True)

    resolution_map = ResolutionMap(
        doc_id="",
        org_id=org_id,
        status="pending",
    )

    seen_mentions: set[str] = set()
    known_mentions = {
        m.mention_text
        for m in mentions
        if m.mention_type not in ("pronoun", "role_reference")
    }

    # Dedup by mention_text, but MERGE the LLM signal across duplicates: the same
    # name can appear in several chunks with different alias lists (chunk 0 sees
    # "Riya Kapoor"; chunk 2 sees it with aliases ["Riya (Data)", "Riya"]). Keep
    # the first candidate but union every duplicate's aliases/canonical into it so
    # no surface form is lost to dedup.
    by_text: dict[str, MentionCandidate] = {}
    resolvable: list[MentionCandidate] = []
    for mention in mentions:
        if mention.mention_type in ("pronoun", "role_reference"):
            continue
        if mention.mention_text in seen_mentions:
            kept = by_text.get(mention.mention_text)
            if kept is not None:
                if not kept.canonical_name and mention.canonical_name:
                    kept.canonical_name = mention.canonical_name
                merged = list(kept.aliases)
                seen_aliases = {_normalize(a) for a in merged}
                for alias in mention.aliases:
                    norm = _normalize(alias)
                    if norm and norm not in seen_aliases:
                        seen_aliases.add(norm)
                        merged.append(alias)
                kept.aliases = merged
            continue
        seen_mentions.add(mention.mention_text)
        by_text[mention.mention_text] = mention
        resolvable.append(mention)
    resolvable.sort(key=lambda m: len(m.mention_text), reverse=True)

    existing_entity_ids = _collect_existing_entity_ids(org_data)

    # Normalized primary mention forms — LLM-alias registration must never shadow
    # a surface form that should resolve to its own entity.
    reserved_keys = {_normalize(m.mention_text) for m in resolvable}

    # Role-decorated single-person forms ("Riya (Data)", "Riya Kapoor, Data Lead")
    # and subsumed first-name fragments ("Riya" when "Riya Kapoor" is present) are
    # NOT dropped. They are deferred and, after the primary pass resolves the full
    # names, attached as aliases of the entity their bare name maps to — so every
    # surface form lands in entity_aliases and future matches are deterministic.
    alias_candidates: list[tuple[MentionCandidate, str]] = []

    for mention in resolvable:
        text = mention.mention_text
        if mention.confidence < MIN_MENTION_CONFIDENCE:
            _log.debug(
                "Skipped mention (low confidence): %r conf=%.2f",
                text,
                mention.confidence,
            )
            continue

        base = derive_person_alias_base(text)
        if base is not None and _normalize(base) != _normalize(text):
            alias_candidates.append((mention, base))
            continue

        if is_subsumed_fragment(text, known_mentions):
            alias_candidates.append((mention, text))
            continue

        if is_junk_mention(text, known_mentions):
            _log.debug("Skipped junk mention: %r", text)
            continue

        entity_type = _MENTION_TYPE_TO_ENTITY_TYPE.get(mention.mention_type, "object")

        entry = await _resolve_single(
            mention=text,
            entity_type=entity_type,
            org_data=org_data,
            org_id=org_id,
            conn=conn,
            mention_confidence=mention.confidence,
            known_mentions=known_mentions,
            existing_entity_ids=existing_entity_ids,
            mention_canonical=mention.canonical_name,
            mention_aliases=mention.aliases,
        )
        if entry is None:
            continue
        if entry.resolution_method == "new":
            # Mint under the LLM's canonical_name when it gave one (e.g. surface
            # "Riya K." -> canonical "Riya Kapoor"), so variant forms converge on
            # one id; the surface form is still registered as an alias below.
            canonical = (mention.canonical_name or text).strip() or text
            provisional_id = entry.entity_id or build_entity_id(
                entity_type=entity_type,
                org_id=org_id,
                canonical_name=canonical,
            )
            entry.entity_id = provisional_id
            _index_alias_in_org_data(
                org_data,
                entity_type=entity_type,
                entity_id=provisional_id,
                alias_norm=_normalize(text),
                canonical_name=canonical,
            )
            if _normalize(canonical) != _normalize(text):
                _index_alias_in_org_data(
                    org_data,
                    entity_type=entity_type,
                    entity_id=provisional_id,
                    alias_norm=_normalize(canonical),
                    canonical_name=canonical,
                )
        resolution_map.entries[text] = entry

        # Bind every other surface form the LLM knows for this entity (bare first
        # name, initials, decorated forms) so intra-run mentions converge and the
        # aliases persist to entity_aliases for deterministic future matching.
        extra_forms = list(mention.aliases)
        if mention.canonical_name:
            extra_forms.append(mention.canonical_name)
        if extra_forms and entry.entity_id:
            await _register_surface_forms(
                conn,
                org_id=org_id,
                entity_type=entity_type,
                entity_id=entry.entity_id,
                forms=extra_forms,
                org_data=org_data,
                resolution_map=resolution_map,
                existing_entity_ids=existing_entity_ids,
                reserved_keys=reserved_keys,
            )

    await _attach_alias_candidates(
        alias_candidates,
        resolution_map=resolution_map,
        org_data=org_data,
        org_id=org_id,
        conn=conn,
        existing_entity_ids=existing_entity_ids,
    )

    _update_counts(resolution_map)

    _log.info(
        "resolve_mentions resolution_map: org=%s map=%s",
        org_id,
        resolution_map.model_dump_json(),
    )
    return resolution_map


async def _resolve_single(
    mention: str,
    entity_type: str,
    org_data: _OrgData,
    org_id: str,
    conn: Any,
    mention_confidence: float = 0.8,
    known_mentions: set[str] | frozenset[str] | None = None,
    existing_entity_ids: set[str] | None = None,
    mention_canonical: str | None = None,
    mention_aliases: list[str] | None = None,
) -> ResolutionMapEntry | None:
    mention_lower = mention.lower().strip()

    # ── Step 1: Email exact match ────────────────────────────────────────────
    if "@" in mention and "." in mention:
        entity_id = org_data.email_index.get(mention_lower)
        if entity_id:
            _log.debug("Email match: %r -> %s", mention, entity_id)
            return ResolutionMapEntry(
                mention_text=mention,
                entity_id=entity_id,
                resolution_method="email_exact",
                confidence=1.0,
            )

    # ── Step 2: Alias exact match ─────────────────────────────────────────────
    alias_norm = _normalize(mention)
    if alias_norm:
        alias_hits = org_data.alias_index.get((alias_norm, entity_type), [])
        if alias_hits:
            return ResolutionMapEntry(
                mention_text=mention,
                entity_id=alias_hits[0],
                resolution_method="alias_exact",
                confidence=0.95,
            )

    # ── Step 2b: LLM canonical / alias probe ─────────────────────────────────
    # The Phase 1 LLM maps surface forms to a canonical_name and lists known
    # aliases. Probe the alias index with those so e.g. "Riya K." (canonical
    # "Riya Kapoor") binds to the entity another mention already seeded — no
    # fuzzy guesswork, the LLM did the disambiguation.
    for probe in (mention_canonical, *(mention_aliases or [])):
        probe_norm = _normalize(probe or "")
        if not probe_norm or probe_norm == _normalize(mention):
            continue
        probe_hits = org_data.alias_index.get((probe_norm, entity_type), [])
        if probe_hits:
            return ResolutionMapEntry(
                mention_text=mention,
                entity_id=probe_hits[0],
                resolution_method="llm_canonical",
                confidence=0.9,
            )

    # ── Step 3: Role + org match ─────────────────────────────────────────────
    role_candidates: list[_Candidate] = []
    if entity_type == "person" and org_data.role_index:
        from rapidfuzz import fuzz
        for entity_id, role_lower in org_data.role_index.items():
            score = fuzz.token_set_ratio(mention_lower, role_lower) / 100.0
            if score >= 0.85:
                role_candidates.append(_Candidate(entity_id, score, "role_match"))

    # ── Step 4: Fuzzy shortlist + LLM disambiguation ─────────────────────────
    fuzzy_shortlist = [
        _Candidate(entity_id, score, "fuzzy_canonical")
        for entity_id, score in _fuzzy_candidates_in_memory(
            org_data=org_data,
            entity_type=entity_type,
            mention_lower=mention_lower,
            threshold=_LLM_FUZZY_THRESHOLD,
            limit=_LLM_FUZZY_MAX_CANDIDATES,
        )
    ]
    if fuzzy_shortlist:
        _log.debug(
            "Disambiguating %r with %d fuzzy candidates",
            mention,
            len(fuzzy_shortlist),
        )
        disambiguated = await _llm_disambiguate(mention, fuzzy_shortlist, conn)
        if disambiguated:
            matched = next(
                (candidate for candidate in fuzzy_shortlist if candidate.entity_id == disambiguated),
                fuzzy_shortlist[0],
            )
            # Write the surface form back only when the matched entity already
            # exists in the registry (FK-safe). Aliases for entities minted this
            # run are persisted downstream via ResolvedEntity.variants.
            if existing_entity_ids and disambiguated in existing_entity_ids:
                await _maybe_write_extraction_alias(
                    conn,
                    org_id=org_id,
                    entity_type=entity_type,
                    entity_id=disambiguated,
                    mention_text=mention,
                    org_data=org_data,
                )
            return ResolutionMapEntry(
                mention_text=mention,
                entity_id=disambiguated,
                resolution_method="llm_disambiguation",
                confidence=matched.confidence,
                alternatives=[
                    candidate.entity_id
                    for candidate in fuzzy_shortlist
                    if candidate.entity_id != disambiguated
                ],
            )
        _log.debug("LLM disambiguation returned null for %r", mention)
        if is_junk_mention(mention, known_mentions):
            _log.debug("Rejected junk mention after disambiguation: %r", mention)
            return None
        return ResolutionMapEntry(
            mention_text=mention,
            entity_id=None,
            resolution_method="new",
            confidence=mention_confidence,
        )

    if role_candidates:
        role_candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        best = role_candidates[0]
        _log.info(
            "resolve_mentions: %s match: %r -> %s (score=%.2f)",
            best.resolution_method,
            mention,
            best.entity_id,
            best.confidence,
        )
        if existing_entity_ids and best.entity_id in existing_entity_ids:
            await _maybe_write_extraction_alias(
                conn,
                org_id=org_id,
                entity_type=entity_type,
                entity_id=best.entity_id,
                mention_text=mention,
                org_data=org_data,
            )
        return ResolutionMapEntry(
            mention_text=mention,
            entity_id=best.entity_id,
            resolution_method=best.resolution_method,
            confidence=best.confidence,
        )

    if is_junk_mention(mention, known_mentions):
        _log.debug("Rejected junk mention: %r", mention)
        return None
    _log.debug("No fuzzy candidates: %r will be created as new entity", mention)
    return ResolutionMapEntry(
        mention_text=mention,
        entity_id=None,
        resolution_method="new",
        confidence=mention_confidence,
    )


async def _llm_disambiguate(
    mention: str,
    candidates: list[_Candidate],
    conn: Any,
) -> str | None:
    """
    Call the LLM with the mention and candidate list. Returns chosen entity_id or None.

    Fetches canonical_name + known aliases for each candidate to build a richer prompt.
    """
    from pydantic import BaseModel
    from processor.agents.base import create_agent, run_agent
    from processor.config import config
    from processor.prompts import get_prompt

    candidate_ids = list(dict.fromkeys(c.entity_id for c in candidates))
    rows = await conn.fetch(
        """
        SELECT
            er.id,
            er.canonical_name,
            COALESCE(
                array_agg(ea.alias_raw) FILTER (WHERE ea.alias_raw IS NOT NULL),
                ARRAY[]::text[]
            ) AS known_aliases
        FROM entity_registry er
        LEFT JOIN entity_aliases ea ON ea.entity_id = er.id
        WHERE er.id = ANY($1::text[])
        GROUP BY er.id, er.canonical_name
        """,
        candidate_ids,
    )
    id_to_meta = {
        row["id"]: {
            "canonical_name": row["canonical_name"],
            "known_aliases": [alias for alias in (row["known_aliases"] or []) if alias],
        }
        for row in rows
    }

    candidates_block = json.dumps([
        {
            "entity_id": c.entity_id,
            "canonical_name": id_to_meta.get(c.entity_id, {}).get("canonical_name", c.entity_id),
            "known_aliases": id_to_meta.get(c.entity_id, {}).get("known_aliases", []),
            "score": round(c.confidence, 3),
        }
        for c in candidates
    ], indent=2)

    mention_line = f'Mention: "{mention}"'
    prompt = (
        f"{mention_line}\n\n"
        f"Candidates:\n{candidates_block}"
    )

    class _DisambiguationResult(BaseModel):
        entity_id: str | None

    agent = create_agent(
        config.entity_model,
        system_prompt=get_prompt("disambiguation"),
        output_type=_DisambiguationResult,
        retries=2,
    )

    try:
        result = await run_agent(
            agent,
            prompt,
            estimated_tokens=300,
            call_site="resolver.disambiguation",
            prompt_sections={
                "mention": mention_line,
                "candidates": candidates_block,
            },
            profile_system_prompt=get_prompt("disambiguation"),
        )
        chosen = result.output.entity_id
        if chosen and any(c.entity_id == chosen for c in candidates):
            return chosen
        _log.warning(
            "LLM disambiguation returned unknown entity_id %r for mention %r",
            chosen,
            mention,
        )
        return None
    except Exception as exc:
        _log.warning("LLM disambiguation failed for %r: %s", mention, exc)
        return None


def _update_counts(resolution_map: ResolutionMap) -> None:
    total = len(resolution_map.entries)
    resolved = sum(
        1
        for e in resolution_map.entries.values()
        if e.entity_id is not None and e.resolution_method != "new"
    )
    new = sum(1 for e in resolution_map.entries.values() if e.resolution_method == "new")
    failed = max(0, total - resolved - new)

    resolution_map.counts = {
        "total": total,
        "resolved": resolved,
        "new": new,
        "failed": failed,
    }


async def resolve_trace_mentions(
    mentions: list[MentionCandidate],
    org_id: str,
    conn: Any,
) -> ResolutionMap:
    """
    Lightweight resolution for trace mentions (LLM-generated, high precision).

    Trace mentions never contain pronouns or role_references, so those are
    skipped. Quality-gate and subsumed-fragment filtering are also skipped —
    the upstream LLM filter already ensured high-signal candidates.

    Keeps both existing-entity matches and unresolved mentions (`resolution_method="new"`).
    New mentions are assigned provisional entity IDs via build_resolved_entities so
    downstream trace pipelines can emit AUTHORED_BY / MENTIONS edges and persist
    entities.
    """
    org_data = await _fetch_org_data(conn, org_id, consistent=True)
    existing_entity_ids = _collect_existing_entity_ids(org_data)

    resolution_map = ResolutionMap(
        doc_id="",
        org_id=org_id,
        status="pending",
    )

    seen_mentions: set[str] = set()

    for mention in mentions:
        if mention.mention_type in ("pronoun", "role_reference"):
            continue
        if mention.mention_text in seen_mentions:
            continue
        seen_mentions.add(mention.mention_text)

        entity_type = _MENTION_TYPE_TO_ENTITY_TYPE.get(mention.mention_type, "object")
        entry = await _resolve_single(
            mention=mention.mention_text,
            entity_type=entity_type,
            org_data=org_data,
            org_id=org_id,
            conn=conn,
            mention_confidence=mention.confidence,
            existing_entity_ids=existing_entity_ids,
        )
        if entry is None:
            continue
        if entry.resolution_method == "new":
            provisional_id = entry.entity_id or build_entity_id(
                entity_type=entity_type,
                org_id=org_id,
                canonical_name=mention.mention_text,
            )
            entry.entity_id = provisional_id
            _index_alias_in_org_data(
                org_data,
                entity_type=entity_type,
                entity_id=provisional_id,
                alias_norm=_normalize(mention.mention_text),
                canonical_name=mention.mention_text,
            )
        resolution_map.entries[mention.mention_text] = entry

    build_resolved_entities(resolution_map, mentions, org_id)
    _update_counts(resolution_map)

    _log.info(
        "resolve_trace_mentions: org=%s resolved=%d/%d",
        org_id,
        len(resolution_map.resolved_entities),
        len(seen_mentions),
    )
    return resolution_map


def build_resolved_entities(
    resolution_map: ResolutionMap,
    mentions: list[MentionCandidate],
    org_id: str,
) -> None:
    """
    Populate resolution_map.resolved_entities from entries.

    For existing entities (entity_id set): create ResolvedEntity(is_new=False).
    For new entities (resolution_method='new'): create ResolvedEntity(is_new=True)
    with a provisional entity_id derived from canonical name.
    """
    mention_type_map: dict[str, str] = {
        m.mention_text: _MENTION_TYPE_TO_ENTITY_TYPE.get(m.mention_type, "object")
        for m in mentions
    }
    mention_confidence_map: dict[str, float] = {
        m.mention_text: m.confidence for m in mentions
    }

    # Pre-seed from whatever is already in the list so re-calling this function
    # (or pre-populated entries) never creates duplicate ResolvedEntity objects.
    seen_entity_ids: set[str] = {e.entity_id for e in resolution_map.resolved_entities}

    known_mentions = frozenset(resolution_map.entries.keys())

    # Build a quick lookup: entity_id -> entity_type, populated as we create new
    # ResolvedEntity objects. Needed so llm_alias entries inherit the right type.
    entity_type_by_id: dict[str, str] = {}

    for mention_text, entry in resolution_map.entries.items():
        entity_type = mention_type_map.get(mention_text, "object")
        mention_confidence = entry.confidence or mention_confidence_map.get(
            mention_text, 0.0,
        )

        # llm_alias entries were already registered by _register_surface_forms;
        # they're aliases not primary mentions, so skip the quality gate (which
        # would filter them) and inherit entity type from the owning entity.
        is_llm_alias = entry.resolution_method == "llm_alias"
        if is_llm_alias:
            if entry.entity_id and entry.entity_id in entity_type_by_id:
                entity_type = entity_type_by_id[entry.entity_id]
        elif not passes_quality_gate(mention_text, mention_confidence, known_mentions):
            continue

        if entry.entity_id is not None and entry.resolution_method != "new":
            if entry.entity_id not in seen_entity_ids:
                seen_entity_ids.add(entry.entity_id)
                entity_type_by_id[entry.entity_id] = entity_type
                resolution_map.resolved_entities.append(
                    ResolvedEntity(
                        entity_id=entry.entity_id,
                        canonical_name=mention_text,
                        entity_type=entity_type,
                        variants=[mention_text],
                        is_new=False,
                        confidence=entry.confidence,
                    )
                )
            else:
                for entity in resolution_map.resolved_entities:
                    if entity.entity_id == entry.entity_id:
                        if mention_text not in entity.variants:
                            entity.variants.append(mention_text)
                        break
        else:
            # llm_alias entries must not mint new entities — they're aliases of
            # an existing entry; if the owning entity hasn't been seen yet just
            # skip (it will be added when we reach the primary mention).
            if is_llm_alias:
                if entry.entity_id and entry.entity_id in seen_entity_ids:
                    for entity in resolution_map.resolved_entities:
                        if entity.entity_id == entry.entity_id:
                            if mention_text not in entity.variants:
                                entity.variants.append(mention_text)
                            break
                continue

            provisional_id = entry.entity_id or build_entity_id(
                entity_type=entity_type,
                org_id=org_id,
                canonical_name=mention_text,
            )
            new_confidence = entry.confidence or mention_confidence_map.get(
                mention_text, 0.0,
            )
            if provisional_id not in seen_entity_ids:
                seen_entity_ids.add(provisional_id)
                entity_type_by_id[provisional_id] = entity_type
                resolution_map.resolved_entities.append(
                    ResolvedEntity(
                        entity_id=provisional_id,
                        canonical_name=mention_text,
                        entity_type=entity_type,
                        variants=[mention_text],
                        is_new=True,
                        confidence=new_confidence,
                    )
                )
            entry.entity_id = provisional_id
            entry.confidence = new_confidence


# ── Public helpers for pre-flight alias scan ─────────────────────────────────

async def fetch_org_data(conn: Any, org_id: str, consistent: bool = False) -> _OrgData:
    """Public wrapper around _fetch_org_data (cached by default)."""
    return await _fetch_org_data(conn, org_id, consistent=consistent)


# Cap on entities sent to the Phase 1 LLM. Below this, send the full org mapping
# (consistent across chunks). Above, the block is truncated — at that scale a
# fuzzy-shortlist against NER candidates should replace the full dump.
KNOWN_ENTITIES_BLOCK_MAX = 400


def build_known_entities_block(
    org_data: _OrgData,
    max_entities: int = KNOWN_ENTITIES_BLOCK_MAX,
) -> str:
    """Format the full known-entity mapping for Phase 1 LLM injection.

    One line per entity: ``type | Canonical Name | aka: alias1, alias2 | entity_id``.
    Sent identically to every chunk (consistent view, like resolution) so the LLM
    can map any surface variant ("Riya K.", "Riya") onto the canonical identity
    itself and reuse the existing entity_id instead of minting a duplicate.

    Aliases equal to the canonical form are omitted from the ``aka:`` list.
    """
    lines: list[str] = []
    for entity_type in sorted(org_data.fuzzy_index.keys()):
        for entity_id, aliases, canonical_lower in org_data.fuzzy_index[entity_type]:
            canonical = org_data.canonical_index.get(entity_id) or canonical_lower
            canonical_norm = _normalize(canonical)
            extra = sorted(
                {a for a in aliases if a and a != canonical_norm}
            )
            aka = f" | aka: {', '.join(extra)}" if extra else ""
            lines.append(f"{entity_type} | {canonical}{aka} | {entity_id}")
            if len(lines) >= max_entities:
                return "\n".join(lines)
    return "\n".join(lines)


_ENTITY_TYPE_TO_MENTION_TYPE: dict[str, str] = {
    "person": "person",
    "service": "service",
    "organization": "org",
    "repository": "repo",
    "tool": "tool",
    "object": "object",
    "location": "location",
    "event": "event",
}


def scan_chunks_for_known_aliases(
    chunks: list[dict],
    org_data: _OrgData,
) -> list[MentionCandidate]:
    """
    Pre-flight scan: find all known alias strings inside chunk texts before Phase 1 LLM.

    For each (alias_normalized, entity_type) in org_data.alias_index, checks if the
    alias text appears (case-insensitive, word-boundary) in any chunk. Matches get
    confidence=1.0 and source="preflight_alias".

    Returns a deduplicated list of MentionCandidates — one per (alias_raw, chunk_index)
    pair so duplicate mention_text across chunks is preserved for pronoun resolution,
    but the same alias won't produce multiple entries for the same chunk.
    """
    from processor.resolution.models import MentionCandidate  # noqa: PLC0415

    seen: set[tuple[str, int]] = set()
    results: list[MentionCandidate] = []

    # Build alias_norm → (alias_display, entity_id, entity_type, mention_type)
    # We want to scan for the alias_norm token; alias_display is what we surface.
    alias_entries: list[tuple[str, str, str, str]] = []
    for (alias_norm, entity_type), entity_ids in org_data.alias_index.items():
        if not alias_norm or not entity_ids:
            continue
        mention_type = _ENTITY_TYPE_TO_MENTION_TYPE.get(entity_type, "object")
        alias_entries.append((alias_norm, entity_ids[0], entity_type, mention_type))

    for chunk_index, chunk in enumerate(chunks):
        chunk_text_lower = chunk["text"].lower()
        for alias_norm, entity_id, entity_type, mention_type in alias_entries:
            key = (alias_norm, chunk_index)
            if key in seen:
                continue
            if alias_norm in chunk_text_lower:
                seen.add(key)
                results.append(
                    MentionCandidate(
                        mention_text=alias_norm,
                        mention_type=mention_type,  # type: ignore[arg-type]
                        chunk_index=chunk_index,
                        source="preflight_alias",
                        confidence=1.0,
                        evidence={"entity_id": entity_id, "entity_type": entity_type},
                    )
                )

    _log.info(
        "Pre-flight alias scan: chunks=%d alias_entries=%d hits=%d",
        len(chunks),
        len(alias_entries),
        len(results),
    )
    return results
