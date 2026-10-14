"""
processor/chunking.py — Chunking + bge-large-en-v1.5 embedding for document chunks.

Two chunking strategies are available (controlled by config.chunking_strategy):

  "legacy"
      Current RecursiveCharacterTextSplitter behaviour — unchanged.

  "structural_entity"  (default)
      Layer 1:  split on markdown headings / paragraph breaks into sections.
      Layer 1b: merge consecutive sections smaller than chunk_min_size so no
                chunk is needlessly tiny.
      Layer 2:  for sections that exceed chunk_size, sub-split at sentence
                boundaries using spaCy en_core_web_sm (never mid-entity).
                Falls back to regex sentence split if spaCy is unavailable.
"""

from __future__ import annotations

import logging
import re
import threading

from processor.config import config

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding clients (lazy singletons)
# ---------------------------------------------------------------------------

_openai_embed_clients: dict[str, object] = {}
_bge_client = None


def _get_openai_embed_client():
    from processor.workspace_keys import (  # noqa: PLC0415
        current_credentials_id,
        current_openai_api_key,
    )

    key = current_openai_api_key() or ""
    cache_key = current_credentials_id()
    client = _openai_embed_clients.get(cache_key)
    if client is None:
        from base.openai_embedding_client import OpenAIEmbeddingClient  # noqa: PLC0415

        client = OpenAIEmbeddingClient(api_key=key or None)
        _openai_embed_clients[cache_key] = client
    return client


def _get_bge_client():
    global _bge_client
    if _bge_client is None:
        from base.inference_client import InferenceClient  # noqa: PLC0415
        _bge_client = InferenceClient(config.inference_url)
    return _bge_client


# ---------------------------------------------------------------------------
# spaCy singleton (lazy-loaded, optional)
# ---------------------------------------------------------------------------

_spacy_nlp = None
_spacy_lock = threading.Lock()
_spacy_available: bool | None = None  # None = not yet probed


def _get_spacy_nlp():
    """Return a spaCy Language object or None if spaCy / en_core_web_sm is absent."""
    global _spacy_nlp, _spacy_available
    if _spacy_available is False:
        return None
    if _spacy_nlp is not None:
        return _spacy_nlp
    with _spacy_lock:
        if _spacy_nlp is not None:
            return _spacy_nlp
        try:
            import spacy  # noqa: PLC0415
            # Keep "senter" and "ner" enabled — both are needed:
            # senter (or parser) for doc.sents, ner for entity span protection.
            # Disabling only lemmatizer and tagger for speed.
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "lemmatizer"])
            _spacy_available = True
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "chunking: spaCy / en_core_web_sm not available — "
                "falling back to regex sentence split (%s)",
                exc,
            )
            _spacy_available = False
    return _spacy_nlp


# ---------------------------------------------------------------------------
# Strategy: legacy
# ---------------------------------------------------------------------------

def _split_legacy(content: str) -> list[dict]:
    """Split using RecursiveCharacterTextSplitter (original behaviour)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(content)

    chunks: list[dict] = []
    search_start = 0
    for idx, text in enumerate(splits):
        offset = content.find(text, search_start)
        if offset == -1:
            offset = search_start
        char_length = len(text)
        chunks.append({
            "chunk_index": idx,
            "text": text,
            "char_offset": offset,
            "char_length": char_length,
        })
        search_start = max(search_start, offset + char_length - config.chunk_overlap)

    return chunks


# ---------------------------------------------------------------------------
# Strategy: structural_entity — helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"(?=\n#{1,6}\s+)")
# Sentence boundary fallback (no spaCy)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class _CharSpan:
    """Char-range span for merged spaCy sentences."""

    __slots__ = ("start_char", "end_char", "text")

    def __init__(self, start_char: int, end_char: int, txt: str):
        self.start_char = start_char
        self.end_char = end_char
        self.text = txt


def _structural_sections(content: str) -> list[tuple[str, int]]:
    """
    Layer 1: split `content` into structural sections.

    Splits on markdown heading boundaries first.  If no headings are found,
    splits on double-newlines (paragraph breaks) instead.

    Returns a list of (section_text, start_offset_in_content) pairs.
    """
    raw_parts = _HEADING_RE.split(content)
    raw_parts = [p for p in raw_parts if p.strip()]
    if len(raw_parts) <= 1:
        raw_parts = [p for p in re.split(r"\n{2,}", content) if p.strip()]
    if not raw_parts:
        raw_parts = [content]

    result: list[tuple[str, int]] = []
    search_from = 0
    for part in raw_parts:
        idx = content.find(part, search_from)
        if idx == -1:
            idx = search_from
        result.append((part, idx))
        search_from = idx + len(part)
    return result


def _sentences_spacy(text: str) -> list[str]:
    """Return a list of sentence strings using spaCy sentence spans.

    Merges sentence pairs where the split falls inside a named-entity span so
    that entities are never cut across chunk boundaries.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return _sentences_regex(text)
    doc = nlp(text)
    raw_sents = list(doc.sents)
    if not raw_sents:
        return [text]

    # Build set of (start_char, end_char) for every entity
    protected = [(ent.start_char, ent.end_char) for ent in doc.ents]

    merged: list[str] = []
    i = 0
    while i < len(raw_sents):
        sent = raw_sents[i]
        # Check whether this sentence's end falls inside any entity span
        while i + 1 < len(raw_sents) and any(
            e_start < sent.end_char < e_end for e_start, e_end in protected
        ):
            i += 1
            next_sent = raw_sents[i]
            # Merge: reconstruct from the original text to preserve whitespace
            merged_text = text[sent.start_char:next_sent.end_char]
            sent = _CharSpan(sent.start_char, next_sent.end_char, merged_text)
        merged.append(sent.text)
        i += 1

    return merged if merged else [text]


def _sentences_regex(text: str) -> list[str]:
    """Return a list of sentence strings using a simple regex heuristic."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p for p in parts if p.strip()] or [text]


def _pack_sentences(
    sentences: list[str], chunk_size: int, chunk_overlap: int
) -> list[tuple[str, int]]:
    """
    Greedily pack sentences into chunks ≤ chunk_size.

    Overlap is implemented by carrying the last `chunk_overlap` characters of
    the previous chunk's text into the start of the next chunk as a plain prefix.

    Returns a list of (chunk_text, first_new_sentence_offset) tuples where
    first_new_sentence_offset is the char offset of the first *new* (non-overlap)
    sentence within the section text passed in.

    Bug 3 fix: a single sentence that exceeds chunk_size is hard-split at the
    nearest word boundary at chunk_size chars; the remainder is re-inserted at
    the front of the remaining sentences.
    """
    # Pre-process: hard-split any sentence that exceeds chunk_size
    expanded: list[str] = []
    for sent in sentences:
        if len(sent) <= chunk_size:
            expanded.append(sent)
        else:
            remainder = sent
            while len(remainder) > chunk_size:
                split_at = chunk_size
                # Walk back to nearest word boundary
                while split_at > 0 and remainder[split_at] not in (" ", "\n", "\t"):
                    split_at -= 1
                if split_at == 0:
                    split_at = chunk_size  # no word boundary found; hard cut
                expanded.append(remainder[:split_at])
                remainder = remainder[split_at:].lstrip()
            if remainder:
                expanded.append(remainder)
    sentences = expanded

    chunks: list[tuple[str, int]] = []
    current_parts: list[str] = []
    current_len = 0
    overlap_prefix = ""
    # Accumulate section-relative char position as we consume sentences
    section_pos = 0
    # Offset of the first sentence in the current chunk (section-relative)
    first_new_offset = 0

    for sent in sentences:
        sent_len = len(sent)
        separator = " " if current_parts else ""
        if current_len + len(separator) + sent_len > chunk_size and current_parts:
            # Flush current chunk
            chunk_text = overlap_prefix + "".join(
                (s if i == 0 else " " + s) for i, s in enumerate(current_parts)
            )
            chunks.append((chunk_text, first_new_offset))
            # Build overlap prefix from tail of flushed chunk
            raw_chunk = "".join(
                (s if i == 0 else " " + s) for i, s in enumerate(current_parts)
            )
            tail = raw_chunk[-chunk_overlap:] if len(raw_chunk) > chunk_overlap else raw_chunk
            overlap_prefix = tail
            # The first new sentence of the *next* chunk starts at current section_pos
            first_new_offset = section_pos
            current_parts = [sent]
            current_len = sent_len
        else:
            current_parts.append(sent)
            current_len += len(separator) + sent_len

        section_pos += sent_len + 1  # +1 for the space separator

    if current_parts:
        chunk_text = overlap_prefix + "".join(
            (s if i == 0 else " " + s) for i, s in enumerate(current_parts)
        )
        chunks.append((chunk_text, first_new_offset))

    return chunks if chunks else [("", 0)]


def _subsplit_section(
    section: str, chunk_size: int, chunk_overlap: int
) -> list[tuple[str, int]]:
    """
    Layer 2: sentence-boundary sub-split for sections exceeding chunk_size.

    Uses spaCy when available; otherwise falls back to regex.

    Returns list of (chunk_text, first_new_sentence_offset_in_section) tuples.
    """
    sentences = _sentences_spacy(section)
    return _pack_sentences(sentences, chunk_size, chunk_overlap)


# ---------------------------------------------------------------------------
# Strategy: structural_entity — main entry
# ---------------------------------------------------------------------------

def _merge_small_sections(
    sections: list[tuple[str, int]], min_size: int, max_size: int
) -> list[tuple[str, int]]:
    """
    Merge consecutive sections that are too small (< min_size chars) into their
    neighbours so that no section is gratuitously tiny.

    A pending micro-section is accumulated until either:
      • the accumulated text reaches min_size, OR
      • appending the next section would exceed max_size (flush first, then start
        a new accumulator with the next section).

    The char_offset of each merged group is anchored to the first section's offset.
    """
    merged: list[tuple[str, int]] = []
    acc_text = ""
    acc_start = 0

    for text, start in sections:
        if not acc_text:
            acc_text = text
            acc_start = start
        elif len(acc_text) + 1 + len(text) > max_size:
            # Flushing the accumulator before it would overflow
            merged.append((acc_text, acc_start))
            acc_text = text
            acc_start = start
        else:
            acc_text = acc_text + "\n\n" + text

        # Flush once the accumulator is large enough
        if len(acc_text) >= min_size:
            merged.append((acc_text, acc_start))
            acc_text = ""
            acc_start = 0

    if acc_text:
        if merged and len(acc_text) < min_size:
            # Tail is still tiny — absorb into the last flushed section
            prev_text, prev_start = merged[-1]
            merged[-1] = (prev_text + "\n\n" + acc_text, prev_start)
        else:
            merged.append((acc_text, acc_start))

    return merged


def _split_structural_entity(content: str) -> list[dict]:
    """
    Two-layer structural + entity-boundary split.

    1. Structural split on headings / paragraphs → (section_text, section_start).
    1b. Merge sections smaller than chunk_min_size into neighbours so no chunk
        is needlessly tiny.
    2. Oversized sections are sub-split at sentence boundaries (spaCy / regex),
       returning (chunk_text, first_new_sentence_offset_in_section) tuples.
    3. char_offset = section_start + first_new_sentence_offset (exact, no find()).
    """
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap
    chunk_min_size = config.chunk_min_size

    sections = _structural_sections(content)
    if chunk_min_size > 0:
        sections = _merge_small_sections(sections, chunk_min_size, chunk_size)

    # Each item: (text, char_offset_in_content)
    raw_chunks: list[tuple[str, int]] = []

    for section_text, section_start in sections:
        if len(section_text) <= chunk_size:
            raw_chunks.append((section_text, section_start))
        else:
            for chunk_text, first_new_offset in _subsplit_section(
                section_text, chunk_size, chunk_overlap
            ):
                raw_chunks.append((chunk_text, section_start + first_new_offset))

    chunks: list[dict] = []
    for idx, (text, char_offset) in enumerate(raw_chunks):
        if not text.strip():
            continue
        chunks.append({
            "chunk_index": idx,
            "text": text,
            "char_offset": char_offset,
            "char_length": len(text),
        })

    # Re-assign sequential chunk_index after blank-skip filtering
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


# ---------------------------------------------------------------------------
# Strategy dispatcher
# ---------------------------------------------------------------------------

def _split_whole_document(content: str) -> list[dict]:
    """Return the entire document as a single chunk (no splitting).

    Used for providers like Fireflies where the LLM benefits from seeing the
    full content in one pass and chunking would harm extraction quality.
    NOTE: the returned chunk carries no embedding — embedding is only applied
    by chunk_and_embed() for retrieval chunks.
    """
    return [
        {
            "chunk_index": 0,
            "text": content,
            "char_offset": 0,
            "char_length": len(content),
        }
    ]


def _split_text(content: str, strategy: str | None = None) -> list[dict]:
    """Dispatch to the configured chunking strategy."""
    effective = (strategy or config.chunking_strategy or "structural_entity").lower()
    if effective == "legacy":
        return _split_legacy(content)
    if effective == "whole_document":
        return _split_whole_document(content)
    return _split_structural_entity(content)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def _encode_chunks(chunks: list[dict]) -> list[dict]:
    """Encode chunk texts with 1024-dim vectors. Mutates chunks in-place.

    OpenAI (default): text-embedding-3-large at dims=1024.
    Inference:        BGE-large-en-v1.5 via inference-service.
    Both produce 1024-dim vectors matching document_chunks vector(1024).
    """
    if config.embedding_provider == "inference":
        texts = ["passage: " + c["text"] for c in chunks]
        vectors = await _get_bge_client().embed_bge(texts)
    else:
        texts = [c["text"] for c in chunks]
        vectors = await _get_openai_embed_client().embed_large(texts)
    for chunk, vec in zip(chunks, vectors):
        chunk["embedding"] = vec
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def chunk_and_embed(content: str) -> list[dict]:
    """
    Split `content` into overlapping chunks and embed each with bge-large-en-v1.5.

    Returns a list of dicts:
        {chunk_index, text, char_offset, char_length, embedding: list[float]}

    Splitting is CPU-bound but fast; embedding is CPU/GPU-bound and may block for
    seconds on large documents — both run in a thread pool executor.

    Always uses the globally configured chunking strategy so retrieval chunks
    are consistently sized and embeddable (bge-large caps at 512 tokens).
    For extraction-only chunks, use chunk_for_extraction() instead.
    """
    from processor.executor import run_in_executor  # noqa: PLC0415

    if not content or not content.strip():
        return []

    chunks = await run_in_executor(_split_text, content)
    if not chunks:
        return []

    chunks = await _encode_chunks(chunks)

    _log.debug(
        "chunk_and_embed: produced %d chunks (strategy=%s)",
        len(chunks),
        config.chunking_strategy,
    )
    return chunks


async def chunk_for_extraction(content: str, strategy: str | None = None) -> list[dict]:
    """
    Split `content` into chunks for LLM extraction only — no embeddings.

    Accepts an explicit strategy override so providers that benefit from
    single-shot extraction (e.g. Fireflies with strategy='whole_document')
    don't need to embed a 20k-token document.

    Returns a list of dicts:
        {chunk_index, text, char_offset, char_length}
    (no 'embedding' key)
    """
    from processor.executor import run_in_executor  # noqa: PLC0415

    if not content or not content.strip():
        return []

    effective_strategy = strategy or config.chunking_strategy
    chunks = await run_in_executor(_split_text, content, effective_strategy)

    _log.debug(
        "chunk_for_extraction: produced %d chunks (strategy=%s)",
        len(chunks),
        effective_strategy,
    )
    return chunks
