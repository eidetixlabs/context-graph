"""
processor/extraction/ner.py — Hybrid NER pipeline (Level 1: spaCy + Level 2: GLiNER).

SpacyExtractor     — fast structural NER for PERSON/ORG/GPE using en_core_web_sm
GlinerExtractor    — zero-shot NER for custom entity types using GLiNER
HybridNERPipeline  — runs both, deduplicates by span overlap, returns NERExtractResult

Both models are lazy-loaded (import + instantiation deferred to first call) and
cached at module level so they are only loaded once per process.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from processor.config import config

_log = logging.getLogger(__name__)

# ── Module-level model singletons (lazy) ─────────────────────────────────────

_spacy_nlp = None

_spacy_lock = threading.Lock()

# ── Label mappings ────────────────────────────────────────────────────────────

_SPACY_TYPE_MAP: dict[str, str] = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
}

_GLINER_LABELS: list[str] = [
    "service",
    "software system",
    "repository",
    "tool",
    "CLI",
    "platform",
    "event",
    "product",
    "location",
    "object",
]

_GLINER_TYPE_MAP: dict[str, str] = {
    "service": "service",
    "software system": "service",
    "platform": "service",
    "product": "service",
    "repository": "repository",
    "tool": "tool",
    "cli": "tool",
    "event": "event",
    "location": "location",
    "object": "object",
}

# ── Raw entity dataclass ──────────────────────────────────────────────────────

@dataclass
class RawEntity:
    mention: str
    entity_type: str
    source: Literal["spacy", "gliner"]
    confidence: float
    # Character offsets in the source text (used for overlap dedup)
    start: int = 0
    end: int = 0


@dataclass
class NERExtractResult:
    """Per-chunk NER output with pre-dedup source counts."""

    entities: list[RawEntity]
    spacy_count: int
    gliner_count: int


# ── Extractors ────────────────────────────────────────────────────────────────

class SpacyExtractor:
    """Extracts PERSON, ORG, GPE entities using spaCy en_core_web_sm."""

    def _get_nlp(self):
        global _spacy_nlp
        if _spacy_nlp is None:
            with _spacy_lock:
                if _spacy_nlp is None:  # double-checked locking
                    import spacy
                    _spacy_nlp = spacy.load("en_core_web_sm")
                    _log.info("spaCy model loaded: en_core_web_sm")
        return _spacy_nlp

    def extract(self, text: str) -> list[RawEntity]:
        nlp = self._get_nlp()
        doc = nlp(text)
        entities: list[RawEntity] = []
        for ent in doc.ents:
            mapped = _SPACY_TYPE_MAP.get(ent.label_)
            if mapped is None:
                continue
            entities.append(
                RawEntity(
                    mention=ent.text.strip(),
                    entity_type=mapped,
                    source="spacy",
                    confidence=config.ner_spacy_confidence,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )
        _log.debug("spaCy extracted %d entities", len(entities))
        return entities


class GlinerExtractor:
    """Extracts custom entity types via POST /ner on the inference-service."""

    def __init__(self) -> None:
        self._disabled = False

    def extract(self, text: str) -> list[RawEntity]:
        if self._disabled:
            return []

        inference_url = config.inference_url.rstrip("/")
        backoff = 0.5
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0) as http:
                    resp = http.post(
                        f"{inference_url}/ner",
                        json={"text": text, "labels": _GLINER_LABELS},
                    )
                    resp.raise_for_status()
                    raw_results = resp.json()["entities"]
                break
            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    self._disabled = True
                    _log.warning(
                        "GLiNER inference-service HTTP %s. Disabling GLiNER NER.",
                        exc.response.status_code,
                    )
                    return []
                last_exc = exc
                _log.warning(
                    "GLiNER inference attempt %d/3 → %s, retrying in %.1fs",
                    attempt + 1, exc.response.status_code, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            except httpx.ConnectError as exc:
                last_exc = exc
                _log.warning(
                    "GLiNER inference attempt %d/3 connect error: %s, retrying in %.1fs",
                    attempt + 1, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
        else:
            self._disabled = True
            _log.warning(
                "GLiNER inference-service unreachable after 3 retries. "
                "Falling back to spaCy-only NER for this worker process.",
            )
            return []

        entities: list[RawEntity] = []
        for item in raw_results:
            label_lower = item["label"].lower()
            mapped = _GLINER_TYPE_MAP.get(label_lower)
            if mapped is None:
                continue
            entities.append(
                RawEntity(
                    mention=item["text"].strip(),
                    entity_type=mapped,
                    source="gliner",
                    confidence=float(item.get("score", 0.7)),
                    start=item.get("start", 0),
                    end=item.get("end", 0),
                )
            )
        _log.debug("GLiNER extracted %d entities", len(entities))
        return entities


# ── Hybrid pipeline ───────────────────────────────────────────────────────────

class HybridNERPipeline:
    """
    Runs spaCy (Level 1) + GLiNER (Level 2) and merges results.

    Deduplication strategy: if two entities' character spans overlap by more
    than 50% of the shorter span's length, keep the one with higher confidence.
    """

    def __init__(self) -> None:
        self._spacy = SpacyExtractor()
        self._gliner = GlinerExtractor()

    def extract(self, text: str) -> NERExtractResult:
        spacy_entities = self._spacy.extract(text)
        gliner_entities = self._gliner.extract(text)
        combined = spacy_entities + gliner_entities
        deduped = _dedup_by_span_overlap(combined)
        _log.debug(
            "HybridNER: spacy=%d gliner=%d combined=%d after_dedup=%d",
            len(spacy_entities),
            len(gliner_entities),
            len(combined),
            len(deduped),
        )
        return NERExtractResult(
            entities=deduped,
            spacy_count=len(spacy_entities),
            gliner_count=len(gliner_entities),
        )


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _overlap_fraction(a: RawEntity, b: RawEntity) -> float:
    """Return the overlap fraction relative to the shorter span."""
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    if end <= start:
        return 0.0
    overlap_len = end - start
    shorter_len = min(a.end - a.start, b.end - b.start)
    if shorter_len == 0:
        return 0.0
    return overlap_len / shorter_len


def _dedup_by_span_overlap(entities: list[RawEntity]) -> list[RawEntity]:
    """
    Remove overlapping entities, keeping the higher-confidence one.

    Entities without meaningful char positions (start == end == 0) are only
    deduplicated by exact mention + type match to avoid false removals.
    """
    kept: list[RawEntity] = []
    for candidate in entities:
        dominated = False
        for i, existing in enumerate(kept):
            # Fall back to mention-based dedup when positions are unavailable
            if candidate.start == 0 and candidate.end == 0:
                if (
                    candidate.mention.lower() == existing.mention.lower()
                    and candidate.entity_type == existing.entity_type
                ):
                    if candidate.confidence > existing.confidence:
                        kept[i] = candidate
                    dominated = True
                    break
                continue

            if _overlap_fraction(candidate, existing) > 0.5:
                if candidate.confidence > existing.confidence:
                    kept[i] = candidate
                dominated = True
                break

        if not dominated:
            kept.append(candidate)
    return kept
