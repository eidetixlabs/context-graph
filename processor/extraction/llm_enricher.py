"""
processor/extraction/llm_enricher.py — LLM enrichment of raw NER candidates.

Takes a list[RawEntity] from the hybrid NER pipeline and document text chunks,
then calls the LLM (via PydanticAI) per-chunk in parallel to:
  - Confirm/correct entity_type
  - Provide canonical_name and aliases
  - Catch clearly missed entities (confidence > 0.7)

Returns list[ExtractedEntityV2] (one flat list across all chunks).
"""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent

from processor.agents.base import create_agent, run_agent
from processor.config import config
from processor.prompts import get_prompt
from processor.schemas.identity import EntityExtractionResultV2, ExtractedEntityV2
from processor.extraction.ner import RawEntity
from processor.workspace_keys import current_credentials_id

_log = logging.getLogger(__name__)

_enricher_agents: dict[str, Agent[None, EntityExtractionResultV2]] = {}


def _get_enricher_agent() -> Agent[None, EntityExtractionResultV2]:
    cache_key = f"{config.entity_model}:{current_credentials_id()}"
    agent = _enricher_agents.get(cache_key)
    if agent is None:
        agent = create_agent(
            config.entity_model,
            output_type=EntityExtractionResultV2,
            system_prompt=get_prompt("identity_v2"),
            output_retries=2,
        )
        _enricher_agents[cache_key] = agent
    return agent


def _format_candidates(raw_entities: list[RawEntity]) -> str:
    """Serialise raw entities as compact pipe-delimited lines, deduped by mention."""
    seen: set[str] = set()
    lines: list[str] = []
    for e in raw_entities:
        key = e.mention.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{e.mention}|{e.entity_type}|{round(e.confidence, 1)}")
    return "\n".join(lines)


def _build_chunk_prompt(
    raw_entities: list[RawEntity],
    chunk_text: str,
    known_aliases_block: str = "",
) -> tuple[str, dict[str, str]]:
    candidates_str = _format_candidates(raw_entities)
    sections: dict[str, str] = {
        "chunk_text": chunk_text,
        "candidates_str": candidates_str,
    }
    known_section = ""
    if known_aliases_block:
        sections["known_aliases"] = known_aliases_block
        known_section = f"""
---
Known entities already in the graph (format: type | Canonical Name | aka: variants | entity_id).
If a mention in this chunk is the same real-world entity as one of these (any variant — abbreviation, first-name-only, role-decorated), set its canonical_name to EXACTLY the canonical shown here and include the listed variants in aliases. Do NOT mint a new identity for a known entity.
{known_aliases_block}
"""
    prompt = f"""Document chunk:
{chunk_text}

---
Pre-identified candidates (mention|type|confidence):
{candidates_str}
{known_section}
---
Also extract 3-8 keywords or short phrases from this chunk that would be useful for memory-search. Focus on specific concepts, decisions, technical terms, not generic words.
"""
    return prompt, sections


async def _enrich_single_chunk(
    raw_entities: list[RawEntity],
    chunk_text: str,
    known_aliases_block: str = "",
) -> EntityExtractionResultV2:
    agent = _get_enricher_agent()
    prompt, prompt_sections = _build_chunk_prompt(raw_entities, chunk_text, known_aliases_block)
    estimated_tokens = len(prompt) // 3 + 500
    try:
        result = await run_agent(
            agent,
            prompt,
            estimated_tokens=estimated_tokens,
            call_site="llm_enricher.chunk",
            prompt_sections=prompt_sections,
            profile_system_prompt=get_prompt("identity_v2"),
        )
        return result.output
    except Exception as exc:
        _log.warning("LLM enrichment failed for chunk: %s", exc)
        # Degrade gracefully: convert raw entities to V2 as-is
        fallback_entities = [
            ExtractedEntityV2(
                mention=e.mention,
                entity_type=e.entity_type,  # type: ignore[arg-type]
                canonical_name=e.mention,
                aliases=[],
                confidence=e.confidence,
                source=e.source,
            )
            for e in raw_entities
        ]
        return EntityExtractionResultV2(entities=fallback_entities)


async def enrich_entities_in_chunks(
    chunk_pairs: list[tuple[dict, list[RawEntity]]],
    per_chunk_known_aliases: list[str] | None = None,
) -> list[EntityExtractionResultV2]:
    """
    Run LLM enrichment for each chunk in parallel (asyncio.gather).

    Each chunk only sees its own NER candidates. If a chunk has no candidates,
    the LLM is still called (it may find entities the NER missed).

    Args:
        chunk_pairs: List of (chunk_dict, raw_entities_for_this_chunk) tuples.
        per_chunk_known_aliases: Optional list of pre-resolved alias blocks (one per chunk).
            Each entry is a formatted string of "alias → entity_id" lines for that chunk.
            When provided, these are injected into the LLM prompt so the LLM is aware of
            already-resolved entities and won't drop or reclassify them.

    Returns:
        One EntityExtractionResultV2 per chunk.
    """
    if not chunk_pairs:
        return []

    known: list[str] = per_chunk_known_aliases or [""] * len(chunk_pairs)
    tasks = [
        _enrich_single_chunk(raw_entities, chunk["text"], known_aliases_block)
        for (chunk, raw_entities), known_aliases_block in zip(chunk_pairs, known)
    ]

    results: list[EntityExtractionResultV2] = await asyncio.gather(*tasks)
    _log.info(
        "LLM enrichment complete: chunks=%d total_entities_raw=%d",
        len(chunk_pairs),
        sum(len(r.entities) for r in results),
    )
    return list(results)
