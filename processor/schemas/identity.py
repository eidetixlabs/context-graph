"""
processor/schemas/identity.py — Output schema for Phase 1: Identity Resolution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EntityType = Literal[
    "person", "service", "organization", "repository", "tool",
    "object", "location", "event",
]


class ResolvedIdentity(BaseModel):
    mention: str = Field(
        description="The raw text as it appears in the document (e.g., '@jane', 'Jane D.', 'postgres').",
    )
    entity_type: EntityType = Field(
        description="Type of the entity.",
    )
    resolved_id: str | None = Field(
        default=None,
        description="Existing entity ID from the graph if matched (e.g., 'entity:person:jane-doe'). "
                    "None if this is a new entity.",
    )
    is_new: bool = Field(
        description="True if this entity does not yet exist in the graph.",
    )
    proposed_canonical_name: str | None = Field(
        default=None,
        description="For new entities only: the proposed canonical name (e.g., 'Jane Doe').",
    )
    proposed_aliases: list[str] = Field(
        default_factory=list,
        description="For new entities only: other names/handles this entity is known by.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this resolution (0.0–1.0). "
                    "Use < 0.6 to flag uncertain matches.",
    )


class ExtractedEntityV2(BaseModel):
    """Enriched entity produced by the V2 hybrid NER + LLM pipeline."""

    mention: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # "spacy", "gliner", or "llm"


class EntityExtractionResultV2(BaseModel):
    """Structured output from one LLM enrichment call (per-chunk)."""

    entities: list[ExtractedEntityV2]
    keywords: list[str] = Field(
        default_factory=list,
        description="3-8 keywords or short phrases for memory-search (specific concepts, decisions, technical terms).",
    )


class IdentityResolutionResult(BaseModel):
    identities: list[ResolvedIdentity] = Field(
        default_factory=list,
        description="All identities resolved from the document content.",
    )
    document_author: ResolvedIdentity | None = Field(
        default=None,
        description="The resolved identity of the document author, if determinable.",
    )
    mentioned_in_metadata: list[str] = Field(
        default_factory=list,
        description="External user IDs from channel_members that appear in this document's scope "
                    "but could not be linked to specific mentions in the text.",
    )
    resolution_notes: str = Field(
        default="",
        description="Any ambiguities or low-confidence matches the extractor should be aware of.",
    )
