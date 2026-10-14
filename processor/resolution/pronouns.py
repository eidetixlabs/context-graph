"""
processor/resolution/pronouns.py — Pronoun and role-reference resolution.

Pronouns are bound to the most recent prior person-typed mention.
Role references are matched against entity_metadata where key='role',
filtered by the nearest preceding org mention.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from processor.resolution.models import MentionCandidate, ResolutionMap, ResolutionMapEntry

_log = logging.getLogger(__name__)

_SUBJECT_PRONOUNS = frozenset({"he", "she", "they"})
_OBJECT_PRONOUNS = frozenset({"him", "her", "them"})
_POSSESSIVE_PRONOUNS = frozenset({"his", "hers", "their", "theirs"})
_ALL_PRONOUNS = _SUBJECT_PRONOUNS | _OBJECT_PRONOUNS | _POSSESSIVE_PRONOUNS

_ROLE_REF_PREFIXES = re.compile(
    r"^(their|the|our|his|her|its|a|an)\s+", re.IGNORECASE
)

_DEFAULT_LOOKBACK_CHUNKS = 1


def _strip_role_prefix(text: str) -> str:
    return _ROLE_REF_PREFIXES.sub("", text).strip()


def resolve_pronouns(
    mentions: list[MentionCandidate],
    resolution_map: ResolutionMap,
    lookback_chunks: int = _DEFAULT_LOOKBACK_CHUNKS,
) -> None:
    """
    Bind pronoun mentions to the most recent person entity within lookback window.

    Mutates resolution_map.entries in place.
    Lookback window: same chunk + `lookback_chunks` chunks prior.
    """
    chunk_to_person_mentions: dict[int, list[tuple[str, int]]] = {}

    person_mentions_ordered = sorted(
        (m for m in mentions if m.mention_type == "person"),
        key=lambda m: (m.chunk_index, m.char_start or 0),
    )
    for mention in person_mentions_ordered:
        entry = resolution_map.entries.get(mention.mention_text)
        if entry and entry.entity_id:
            chunk_idx = mention.chunk_index
            if chunk_idx not in chunk_to_person_mentions:
                chunk_to_person_mentions[chunk_idx] = []
            chunk_to_person_mentions[chunk_idx].append((mention.mention_text, mention.char_start or 0))

    for mention in mentions:
        if mention.mention_type != "pronoun":
            continue

        pronoun_lower = mention.mention_text.lower().strip(".,!?;:\"'")
        if pronoun_lower not in _ALL_PRONOUNS:
            continue

        target_entity_id: str | None = None
        search_start = max(0, mention.chunk_index - lookback_chunks)
        pronoun_char_start = mention.char_start or 0

        for chunk_idx in range(mention.chunk_index, search_start - 1, -1):
            person_mentions_in_chunk = chunk_to_person_mentions.get(chunk_idx, [])
            if not person_mentions_in_chunk:
                continue
            if chunk_idx == mention.chunk_index:
                # Same chunk: only consider persons that precede the pronoun
                preceding = [
                    (m_text, m_start)
                    for m_text, m_start in person_mentions_in_chunk
                    if m_start < pronoun_char_start
                ]
                if not preceding:
                    continue
                last_person_mention = preceding[-1][0]
            else:
                last_person_mention = person_mentions_in_chunk[-1][0]
            entry = resolution_map.entries.get(last_person_mention)
            if entry and entry.entity_id:
                target_entity_id = entry.entity_id
                break

        key = (
            f"{mention.mention_text}@chunk{mention.chunk_index}"
            f"@pos{mention.char_start if mention.char_start is not None else -1}"
        )
        if target_entity_id:
            resolution_map.entries[key] = ResolutionMapEntry(
                mention_text=mention.mention_text,
                entity_id=target_entity_id,
                resolution_method="pronoun_lookback",
                confidence=0.7,
            )
            resolution_map.add_audit(
                mention.mention_text,
                f"pronoun bound to {target_entity_id} from chunk lookback",
            )
        else:
            resolution_map.entries[key] = ResolutionMapEntry(
                mention_text=mention.mention_text,
                entity_id=None,
                resolution_method="pronoun_unresolved",
                confidence=0.0,
            )
            resolution_map.add_audit(
                mention.mention_text,
                "pronoun could not be bound — no preceding person mention",
            )


async def resolve_role_references(
    mentions: list[MentionCandidate],
    resolution_map: ResolutionMap,
    org_id: str,
    conn: Any,
) -> None:
    """
    Resolve role-reference mentions against entity_metadata where key='role'.

    Uses token_set_ratio for matching. Filtered by nearest preceding org mention
    if available. Mutates resolution_map.entries in place.
    """
    rows = await conn.fetch(
        """
        SELECT em.entity_id, em.value AS role_value, er.canonical_name,
               org_em.value AS org_entity_id
        FROM entity_metadata em
        JOIN entity_registry er ON er.id = em.entity_id
        LEFT JOIN entity_metadata org_em
               ON org_em.entity_id = em.entity_id AND org_em.key = 'org'
        WHERE em.key = 'role'
          AND er.org_id = $1
          AND er.status = 'active'
        """,
        org_id,
    )

    if not rows:
        return

    from rapidfuzz import fuzz

    chunk_to_org_entity: dict[int, str | None] = {}
    org_mentions = sorted(
        (m for m in mentions if m.mention_type == "org"),
        key=lambda m: (m.chunk_index, m.char_start or 0),
    )
    for mention in org_mentions:
        entry = resolution_map.entries.get(mention.mention_text)
        if not entry or not entry.entity_id:
            continue
        if mention.chunk_index in chunk_to_org_entity:
            continue
        if entry.resolution_method == "new":
            continue
        chunk_to_org_entity[mention.chunk_index] = entry.entity_id

    for mention in mentions:
        if mention.mention_type != "role_reference":
            continue

        stripped = _strip_role_prefix(mention.mention_text)
        if not stripped:
            continue

        nearest_org: str | None = None
        for chunk_idx in range(mention.chunk_index, -1, -1):
            if chunk_idx in chunk_to_org_entity:
                nearest_org = chunk_to_org_entity[chunk_idx]
                break

        best_entity_id: str | None = None
        best_score = 0.0

        # When nearest_org is None (no resolved org in lookback) and multiple
        # distinct org-affiliated entities share the same role title, we cannot
        # filter safely.  Accept the match only when all candidate rows point to
        # the same entity, or when there is no org affiliation in any row at all
        # (org_entity_id is NULL for every row — single-org workspaces).
        org_affiliated_rows = [r for r in rows if r["org_entity_id"] is not None]
        skip_org_filter_safe = nearest_org is None and (
            len({r["entity_id"] for r in rows}) == 1
            or len(org_affiliated_rows) == 0
        )

        for row in rows:
            row_org = row["org_entity_id"]
            if (
                nearest_org is not None
                and row_org is not None
                and row_org != nearest_org
            ):
                continue
            if nearest_org is None and not skip_org_filter_safe:
                continue
            score = fuzz.token_set_ratio(stripped.lower(), row["role_value"].lower()) / 100.0
            if score > best_score:
                best_score = score
                best_entity_id = row["entity_id"]

        key = (
            f"{mention.mention_text}@chunk{mention.chunk_index}"
            f"@pos{mention.char_start if mention.char_start is not None else -1}"
        )
        if best_entity_id and best_score >= 0.8:
            resolution_map.entries[key] = ResolutionMapEntry(
                mention_text=mention.mention_text,
                entity_id=best_entity_id,
                resolution_method="role_reference",
                confidence=best_score,
            )
            resolution_map.add_audit(
                mention.mention_text,
                f"role reference resolved to {best_entity_id} (score={best_score:.2f})",
            )
        else:
            resolution_map.entries[key] = ResolutionMapEntry(
                mention_text=mention.mention_text,
                entity_id=None,
                resolution_method="role_ref_unresolved",
                confidence=0.0,
            )
