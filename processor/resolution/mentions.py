"""
processor/resolution/mentions.py — Extract MentionCandidates from document chunks.

Uses the HybridNERPipeline (spaCy + GLiNER) followed by LLM enrichment per chunk.
LLM entities are filtered (confidence, junk patterns), capped at 20 per chunk;
NER backfill only applies for high-confidence non-junk candidates.
"""

from __future__ import annotations

import asyncio
import logging
import re

from processor.executor import run_in_executor
from processor.extraction.llm_enricher import enrich_entities_in_chunks
from processor.extraction.ner import HybridNERPipeline, NERExtractResult, RawEntity
from processor.resolution.entity_quality import (
    MAX_LLM_ENTITIES_PER_CHUNK,
    MIN_MENTION_CONFIDENCE,
    MIN_NER_BACKFILL_CONFIDENCE,
    is_junk_mention,
)
from processor.resolution.models import MentionCandidate, MentionType

_log = logging.getLogger(__name__)

_NER_TYPE_TO_MENTION_TYPE: dict[str, MentionType] = {
    "person": "person",
    "organization": "org",
    "service": "service",
    "repository": "repo",
    "tool": "tool",
    "object": "object",
    "location": "location",
    "event": "event",
    "role_reference": "role_reference",
}

_PRONOUN_SET = frozenset({"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs"})

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Longer forms (theirs/hers) must precede shorter prefixes (their/her) to avoid
# partial matches under re alternation.
_PRONOUN_RE = re.compile(
    r"\b(theirs|hers|they|them|their|she|his|him|he|her)\b",
    re.IGNORECASE,
)
# Prefix article/possessive is optional so bare role phrases ("Lead Engineer",
# "VP Eng") match in addition to prefixed ones ("Their Lead Engineer").
_ROLE_REF_RE = re.compile(
    r"\b(?:(?:their|the|our|his|her|its|a|an)\s+)?"
    r"(?:(?:lead|principal|senior|staff|junior|head\s+of|director\s+of|chief|vp\s+of?|vp)\s+)?"
    r"(?:vp\s+eng|engineer(?:ing)?|architect|product\s+manager|designer|analyst|developer|"
    r"scientist|researcher|manager|officer|lead|exec(?:utive)?|cto|ceo|coo|cfo|vp)\b",
    re.IGNORECASE,
)


def _chunk_name_context(
    llm_entities: list,
    raw_entities: list[RawEntity],
) -> set[str]:
    """Collect mention strings in a chunk for possessive / fragment checks."""
    names: set[str] = set()
    for entity in llm_entities:
        mention = entity.mention.strip()
        if mention and not mention.endswith("'s"):
            names.add(mention)
    for raw in raw_entities:
        mention = raw.mention.strip()
        if mention and not mention.endswith("'s"):
            names.add(mention)
    return names


def _raw_entity_to_mention(entity: RawEntity, chunk_index: int) -> MentionCandidate:
    mention_type: MentionType = _NER_TYPE_TO_MENTION_TYPE.get(entity.entity_type, "object")
    return MentionCandidate(
        mention_text=entity.mention,
        mention_type=mention_type,
        chunk_index=chunk_index,
        char_start=entity.start,
        char_end=entity.end,
        source=entity.source,
        confidence=entity.confidence,
        evidence={"ner_source": entity.source},
    )


def _llm_entity_to_mention(
    mention: str,
    entity_type: str,
    chunk_index: int,
    confidence: float,
    source: str,
    canonical_name: str | None = None,
    aliases: list[str] | None = None,
) -> MentionCandidate:
    lower = mention.lower().strip()
    if lower in _PRONOUN_SET:
        mention_type: MentionType = "pronoun"
    else:
        mention_type = _NER_TYPE_TO_MENTION_TYPE.get(entity_type, "object")
    return MentionCandidate(
        mention_text=mention,
        mention_type=mention_type,
        chunk_index=chunk_index,
        source=source,
        confidence=confidence,
        evidence={"llm_source": source},
        canonical_name=(canonical_name or "").strip() or None,
        aliases=[a.strip() for a in (aliases or []) if a and a.strip()],
    )


async def extract_mentions(
    chunks: list[dict],
    ner_pipeline: HybridNERPipeline,
    per_chunk_known_aliases: list[str] | None = None,
) -> tuple[list[MentionCandidate], list[str]]:
    """
    Extract MentionCandidates from document chunks using NER + LLM enrichment.

    LLM output is filtered and capped per chunk; NER backfill is conservative.
    Pronoun and email heuristics run after structured extraction.

    Args:
        chunks:                    Pre-computed chunks with embeddings.
        ner_pipeline:              HybridNERPipeline instance.
        per_chunk_known_aliases:   Optional per-chunk pre-resolved alias blocks from
                                   pre-flight scan. Injected into LLM prompt so already-
                                   known entities aren't dropped or reclassified.

    Returns:
        Tuple of (mention_candidates, aggregated_keywords).
    """
    ner_tasks = [
        run_in_executor(ner_pipeline.extract, chunk["text"])
        for chunk in chunks
    ]
    ner_results: list[NERExtractResult] = list(await asyncio.gather(*ner_tasks))
    per_chunk_raw: list[list[RawEntity]] = [r.entities for r in ner_results]

    spacy_total = sum(r.spacy_count for r in ner_results)
    gliner_total = sum(r.gliner_count for r in ner_results)
    deduped_total = sum(len(r.entities) for r in ner_results)
    _log.info(
        "Phase 1 Mention NER: chunks=%d spacy=%d gliner=%d combined=%d deduped=%d",
        len(chunks),
        spacy_total,
        gliner_total,
        spacy_total + gliner_total,
        deduped_total,
    )

    chunk_results = await enrich_entities_in_chunks(
        list(zip(chunks, per_chunk_raw)),
        per_chunk_known_aliases=per_chunk_known_aliases,
    )

    candidates: list[MentionCandidate] = []
    seen_kw: set[str] = set()
    all_keywords: list[str] = []

    for chunk_index, (chunk, raw_entities, chunk_result) in enumerate(
        zip(chunks, per_chunk_raw, chunk_results)
    ):
        for kw in chunk_result.keywords:
            kw_lower = kw.lower().strip()
            if kw_lower and kw_lower not in seen_kw:
                seen_kw.add(kw_lower)
                all_keywords.append(kw)
        seen_mentions: set[tuple[str, str | int]] = set()
        rejected_llm_keys: set[tuple[str, str]] = set()
        chunk_known_mentions = _chunk_name_context(chunk_result.entities, raw_entities)

        llm_accepted: list[tuple[str, str, float, str, tuple[str, str], str, list[str]]] = []
        for entity in chunk_result.entities:
            key = (entity.mention.lower(), entity.entity_type)
            if entity.confidence < MIN_MENTION_CONFIDENCE or is_junk_mention(
                entity.mention, chunk_known_mentions,
            ):
                rejected_llm_keys.add(key)
                continue
            llm_accepted.append(
                (
                    entity.mention,
                    entity.entity_type,
                    entity.confidence,
                    entity.source,
                    key,
                    entity.canonical_name,
                    list(entity.aliases),
                ),
            )

        llm_accepted.sort(key=lambda item: item[2], reverse=True)
        for (
            mention,
            entity_type,
            confidence,
            source,
            key,
            canonical_name,
            aliases,
        ) in llm_accepted[:MAX_LLM_ENTITIES_PER_CHUNK]:
            if key in seen_mentions:
                continue
            seen_mentions.add(key)
            chunk_known_mentions.add(mention)
            candidates.append(
                _llm_entity_to_mention(
                    mention,
                    entity_type,
                    chunk_index,
                    confidence,
                    source,
                    canonical_name=canonical_name,
                    aliases=aliases,
                )
            )

        chunk_text = chunk["text"]

        for match in _PRONOUN_RE.finditer(chunk_text):
            pronoun = match.group(1)
            type_key = (pronoun.lower(), "pronoun")
            pos_key = (pronoun.lower(), match.start())
            if pos_key in seen_mentions:
                continue
            seen_mentions.add(pos_key)
            seen_mentions.add(type_key)
            candidates.append(
                MentionCandidate(
                    mention_text=pronoun,
                    mention_type="pronoun",
                    chunk_index=chunk_index,
                    char_start=match.start(),
                    char_end=match.end(),
                    source="heuristic",
                    confidence=0.9,
                    evidence={"heuristic": "pronoun_scan"},
                )
            )

    _log.info(
        "Mention extraction complete: chunks=%d total_candidates=%d keywords=%d",
        len(chunks),
        len(candidates),
        len(all_keywords),
    )
    return candidates, all_keywords
