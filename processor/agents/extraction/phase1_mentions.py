"""
processor/agents/extraction/phase1_mentions.py — Phase 1 (Mentions): extract all
mention candidates from document chunks using HybridNER + LLM enrichment.

This replaces the truncate-and-extract pattern. Output is list[MentionCandidate]
including pronouns and role references for downstream resolution.
"""

from __future__ import annotations

import json
import logging
import threading

from processor.extraction.ner import HybridNERPipeline
from processor.resolution.mentions import extract_mentions
from processor.resolution.models import MentionCandidate
from processor.state_machine import DocumentRecord

_log = logging.getLogger(__name__)

_ner_pipeline: HybridNERPipeline | None = None
_ner_lock = threading.Lock()


def _get_ner_pipeline() -> HybridNERPipeline:
    global _ner_pipeline
    if _ner_pipeline is None:
        with _ner_lock:
            if _ner_pipeline is None:
                _ner_pipeline = HybridNERPipeline()
    return _ner_pipeline


async def run_mention_extraction(
    doc: DocumentRecord,
    content: str,
    chunks: list[dict],
    per_chunk_known_aliases: list[str] | None = None,
) -> tuple[list[dict], list[MentionCandidate], list[str]]:
    """
    Phase 1 (Mentions): run NER + LLM enrichment on pre-chunked content,
    return (chunks, mention_candidates, keywords).

    Args:
        doc:                       Document record.
        content:                   Full document text (unused, kept for logging/compatibility).
        chunks:                    Pre-computed chunks with embeddings from chunk_and_embed().
        per_chunk_known_aliases:   Optional pre-resolved alias blocks from pre-flight scan
                                   (one entry per chunk). Injected into the LLM prompt so
                                   already-known entities are preserved verbatim.

    Returns:
        (chunks, mentions, keywords) — chunks for Phase 3 reuse, mentions for resolution,
        keywords aggregated from LLM enrichment for memory-search.
    """
    _log.info(
        "Phase 1 Mentions start: doc=%s content_len=%d chunks=%d preflight_chunks=%d",
        doc.id,
        len(content),
        len(chunks),
        sum(1 for b in (per_chunk_known_aliases or []) if b),
    )

    pipeline = _get_ner_pipeline()
    mentions, keywords = await extract_mentions(
        chunks, pipeline, per_chunk_known_aliases=per_chunk_known_aliases
    )

    _log.info(
        "Phase 1 Mentions complete: doc=%s chunks=%d mentions=%d keywords=%d",
        doc.id,
        len(chunks),
        len(mentions),
        len(keywords),
    )
    _log.info(
        "Phase 1 Mentions: doc=%s mentions=%s",
        doc.id,
        json.dumps([m.model_dump() for m in mentions]),
    )
    return chunks, mentions, keywords
