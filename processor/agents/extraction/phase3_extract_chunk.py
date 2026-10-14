"""
processor/agents/extraction/phase3_extract_chunk.py — Per-chunk Phase 3 extraction.

Each chunk gets:
  - Its own text (not the full document)
  - The full resolution map (entity IDs for all mentions)
  - Phase 2 context (shared across all chunks)

Concurrency is capped via `extraction_chunk_concurrency` config (default 4).
Single-chunk failures are logged and skipped; other chunks are unaffected.
"""

from __future__ import annotations

import logging
from typing import cast

from pydantic_ai import Agent

from processor.agents.base import create_agent, run_agent
from processor.agents.extraction.phase2_context import slim_context_for_prompt
from processor.config import config
from processor.prompts import get_prompt
from processor.resolution.entity_quality import PHASE3_MIN_CONFIDENCE
from processor.resolution.models import ChunkExtractionResult, ResolutionMap
from processor.schemas.context import ContextGatheringResult
from processor.schemas.extraction import ChunkExtractionOutput, ExtractionResult
from processor.schemas.scoring import ScoringResult
from processor.state_machine import DocumentRecord
from processor.workspace_keys import current_credentials_id

_log = logging.getLogger(__name__)

_chunk_extraction_agents: dict[str, Agent[None, ChunkExtractionOutput]] = {}


def _get_chunk_extraction_agent() -> Agent[None, ChunkExtractionOutput]:
    cache_key = f"{config.extraction_model}:{current_credentials_id()}"
    agent = _chunk_extraction_agents.get(cache_key)
    if agent is None:
        agent = cast(Agent[None, ChunkExtractionOutput], create_agent(
            config.extraction_model,
            output_type=ChunkExtractionOutput,
            system_prompt=get_prompt("extraction_chunk"),
            output_retries=2,
            # model_settings={"openai_reasoning_effort": "medium"},
        ))
        _chunk_extraction_agents[cache_key] = agent
    return agent


def _build_chunk_prompt(
    doc: DocumentRecord,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    resolution_map: ResolutionMap,
    context_result: ContextGatheringResult,
    scoring_result: ScoringResult,
    previous_summary: str = "",
    max_summary_tokens: int = 1000,
) -> tuple[str, dict[str, str]]:
    resolution_block = resolution_map.format_for_prompt(
        min_confidence=PHASE3_MIN_CONFIDENCE,
        dedupe=True,
    )
    context_json = slim_context_for_prompt(context_result).model_dump_json(indent=None)
    signal_types_text = ", ".join(scoring_result.signal_types) or "unknown"
    topics_text = ", ".join(scoring_result.suggested_topics) or "(none detected)"
    doc_header = (
        f"Title: {doc.title or '(untitled)'}\n"
        f"Source: {doc.source_provider} / {doc.source_scope}\n"
        f"Doc ID: {doc.id}\n"
        f"Chunk: {chunk_index + 1} of {total_chunks}"
    )

    if previous_summary:
        previous_context_section = (
            f"## CONTEXT FROM PREVIOUS CHUNKS\n\n"
            f"The following summary captures key information from earlier chunks. "
            f"Use this context to identify cross-chunk relationships (e.g., decisions in this chunk shaped by constraints from earlier chunks).\n\n"
            f"{previous_summary}\n\n---\n\n"
        )
    else:
        previous_context_section = ""

    sections = {
        "doc_header": doc_header,
        "chunk_text": chunk_text,
        "previous_summary": previous_context_section,
        "resolution_map": resolution_block,
        "context_json": context_json,
        "signal_types": signal_types_text,
        "topics": topics_text,
    }

    prompt = f"""Document:
{doc_header}

---
SCORING — Priority signal types: {signal_types_text}
Suggested topics: {topics_text}

---
CHUNK TEXT:
{chunk_text}

---
{previous_context_section}RESOLUTION MAP:
{resolution_block}

---
PHASE 2 — Graph Context:
{context_json}

---
Extract semantic nodes and edges from this chunk only, following the system prompt rules.
Include Decision, Constraint, Insight, Signal, Topic, Preference, Fact, and Entity nodes
that this chunk supports — not Entity nodes alone.
Use Resolution Map IDs for Entity nodes and MENTIONS edges when they match real entities;
omit noisy NER composites and create/use the correct Entity node instead.
Use existing node IDs from Phase 2 for SUPERSEDES, CONTRADICTS, and SHAPED_BY edges.
Provide a chunk_summary (max {max_summary_tokens} tokens) capturing key decisions,
constraints, insights, signals, and entities for use by subsequent chunks.
"""
    return prompt, sections


async def _extract_single_chunk(
    doc: DocumentRecord,
    chunk: dict,
    chunk_index: int,
    total_chunks: int,
    resolution_map: ResolutionMap,
    context_result: ContextGatheringResult,
    scoring_result: ScoringResult,
    previous_summary: str = "",
    max_summary_tokens: int = 1000,
) -> ChunkExtractionResult | None:
    prompt, prompt_sections = _build_chunk_prompt(
        doc=doc,
        chunk_text=chunk["text"],
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        resolution_map=resolution_map,
        context_result=context_result,
        scoring_result=scoring_result,
        previous_summary=previous_summary,
        max_summary_tokens=max_summary_tokens,
    )
    try:
        result = await run_agent(
            _get_chunk_extraction_agent(),
            prompt,
            estimated_tokens=len(prompt) // 3 + 1500,
            call_site=f"phase3_extract_chunk.{chunk_index}",
            prompt_sections=prompt_sections,
            profile_system_prompt=get_prompt("extraction_chunk"),
        )
        return ChunkExtractionResult(
            chunk_index=chunk_index,
            extraction=ExtractionResult(
                nodes=result.output.nodes,
                edges=result.output.edges,
                extraction_notes=result.output.extraction_notes,
            ),
            chunk_summary=result.output.chunk_summary,
        )
    except Exception as exc:
        _log.warning(
            "Phase 3 chunk extraction failed: doc=%s chunk=%d error=%s",
            doc.id,
            chunk_index,
            exc,
        )
        return None


async def run_chunk_extraction(
    doc: DocumentRecord,
    chunks: list[dict],
    resolution_map: ResolutionMap,
    context_result: ContextGatheringResult,
    scoring_result: ScoringResult,
    concurrency: int = 4,  # kept for API compat, unused now
) -> list[ChunkExtractionResult]:
    """
    Run Phase 3 extraction for each chunk sequentially, passing the cumulative
    summary from each chunk to the next for cross-chunk context.

    Args:
        doc:            Document record.
        chunks:         List of chunk dicts with 'text' key.
        resolution_map: Resolution map from Phase 1.
        context_result: Graph context from Phase 2.
        scoring_result: Scoring output (signal types + topics) guiding semantic extraction.
        concurrency:    Kept for API compatibility; unused (sequential processing).

    Returns:
        List of ChunkExtractionResult (failed chunks are excluded).
    """
    from processor.config import config as _cfg
    max_summary_tokens = getattr(_cfg, "chunk_summary_max_tokens", 1000)
    total = len(chunks)
    results = []
    cumulative_summary = ""

    for idx, chunk in enumerate(chunks):
        result = await _extract_single_chunk(
            doc=doc,
            chunk=chunk,
            chunk_index=idx,
            total_chunks=total,
            resolution_map=resolution_map,
            context_result=context_result,
            scoring_result=scoring_result,
            previous_summary=cumulative_summary,
            max_summary_tokens=max_summary_tokens,
        )
        if result is not None:
            results.append(result)
            cumulative_summary = result.chunk_summary
            type_counts: dict[str, int] = {}
            for node in result.extraction.nodes:
                if isinstance(node, dict):
                    node_type = str(node.get("type", "unknown"))
                else:
                    node_type = str(getattr(node, "type", "unknown"))
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            _log.info(
                "Phase 3 chunk extracted: doc=%s chunk=%d nodes=%d edges=%d types=%s",
                doc.id,
                idx,
                len(result.extraction.nodes),
                len(result.extraction.edges),
                type_counts,
            )

    _log.info(
        "Chunk extraction complete: doc=%s chunks=%d succeeded=%d failed=%d",
        doc.id, total, len(results), total - len(results),
    )
    return results
