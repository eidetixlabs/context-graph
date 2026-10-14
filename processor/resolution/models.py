"""
processor/resolution/models.py — Domain models for canonical entity resolution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from processor.schemas.extraction import ExtractionResult


MentionType = Literal[
    "person", "service", "org", "repo", "tool", "object",
    "location", "event", "role_reference", "pronoun",
]

ResolutionMethod = str


class MentionCandidate(BaseModel):
    mention_text: str
    mention_type: MentionType
    chunk_index: int
    char_start: int = 0
    char_end: int = 0
    source: str = "ner"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    evidence: dict[str, Any] = Field(default_factory=dict)
    # LLM-supplied resolution signal (Phase 1 enrichment). canonical_name is the
    # formal full name the LLM mapped this surface form to; aliases are the other
    # surface forms it knows for the same entity. Carried into resolution so the
    # entity's known aliases seed the alias index and persist to entity_aliases.
    canonical_name: str | None = None
    aliases: list[str] = Field(default_factory=list)


class ResolvedEntity(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    variants: list[str] = Field(default_factory=list)
    is_new: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: dict[str, str] = Field(default_factory=dict)


class ResolutionMapEntry(BaseModel):
    mention_text: str
    entity_id: str | None = None
    resolution_method: ResolutionMethod = "none"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    alternatives: list[str] = Field(default_factory=list)


class ResolutionMap(BaseModel):
    doc_id: str
    doc_version: int = 1
    org_id: str
    status: Literal["pending", "resolved", "partial", "failed"] = "pending"
    entries: dict[str, ResolutionMapEntry] = Field(default_factory=dict)
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    audit: list[dict[str, Any]] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    newly_persisted_ids: list[str] = Field(default_factory=list)

    def get_entity_id(self, mention: str) -> str | None:
        entry = self.entries.get(mention)
        if entry is not None:
            return entry.entity_id
        mention_lower = mention.lower()
        for key, entry in self.entries.items():
            if key.lower() == mention_lower:
                return entry.entity_id
        return None

    def entity_stub(self, entity_id: str) -> dict[str, Any] | None:
        for entity in self.resolved_entities:
            if entity.entity_id == entity_id:
                return {
                    "type": "Entity",
                    "id": entity_id,
                    "canonical_name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                    "_stub": True,
                }
        return None

    def format_for_prompt(
        self,
        min_confidence: float = 0.0,
        dedupe: bool = False,
    ) -> str:
        lines: list[str] = ["## Resolution Map"]
        filtered = [
            (mention, entry)
            for mention, entry in self.entries.items()
            if entry.confidence >= min_confidence
        ]

        if dedupe:
            by_entity: dict[str, tuple[str, ResolutionMapEntry]] = {}
            for mention, entry in filtered:
                if not entry.entity_id:
                    continue
                canonical = mention
                for entity in self.resolved_entities:
                    if entity.entity_id == entry.entity_id:
                        canonical = entity.canonical_name
                        break
                existing = by_entity.get(entry.entity_id)
                if existing is None or entry.confidence > existing[1].confidence:
                    by_entity[entry.entity_id] = (canonical, entry)
            for canonical, entry in sorted(
                by_entity.values(),
                key=lambda item: (-item[1].confidence, item[0].lower()),
            ):
                lines.append(f"{canonical} → {entry.entity_id}")
            return "\n".join(lines) if len(lines) > 1 else "(no entities resolved)"

        for mention, entry in filtered:
            if entry.entity_id:
                lines.append(f"{mention} → {entry.entity_id}")
            else:
                entity_type = next(
                    (e.entity_type for e in self.resolved_entities if e.canonical_name == mention),
                    "unknown",
                )
                lines.append(f"{mention} → NEW {entity_type}")
        return "\n".join(lines) if len(lines) > 1 else "(no entities resolved)"

    def add_audit(self, mention: str, reason: str) -> None:
        self.audit.append({
            "mention": mention,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


class ChunkExtractionResult(BaseModel):
    chunk_index: int
    extraction: ExtractionResult
    chunk_summary: str = Field(default="")
