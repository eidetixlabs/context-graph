"""
processor/tools/chunk_reader.py — get_document_chunk tool for Phase 2 agent.

Provides the Phase 2 context agent a safety-net way to request raw chunk text
when the prefetched keywords + graph context is insufficient. The agent can
request any chunk by index without seeing all chunks upfront.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_chunks_registry: dict[str, list[dict]] = {}


def register_chunks(doc_id: str, chunks: list[dict]) -> None:
    """Register chunks for a document so the tool can serve them during Phase 2."""
    _chunks_registry[doc_id] = chunks


def unregister_chunks(doc_id: str) -> None:
    """Clean up after Phase 2 completes to avoid memory leaks."""
    _chunks_registry.pop(doc_id, None)


def get_document_chunk(doc_id: str, chunk_index: int) -> str:
    """
    Return the text of a specific document chunk by index.

    Used by the Phase 2 context agent when it needs to inspect raw content
    for a specific section. Prefer keywords and graph tools first.

    Args:
        doc_id:      Document ID (used to look up registered chunks).
        chunk_index: Zero-based chunk index.

    Returns:
        Chunk text, or an error message if index is out of range.
    """
    chunks = _chunks_registry.get(doc_id)
    if chunks is None:
        return f"No chunks registered for doc_id={doc_id!r}."
    if not (0 <= chunk_index < len(chunks)):
        return f"chunk_index={chunk_index} out of range. Valid: 0–{len(chunks) - 1}."
    text = chunks[chunk_index]["text"]
    _log.info("get_document_chunk: doc=%s chunk_index=%d chars=%d", doc_id, chunk_index, len(text))
    return text
