"""
processor/schemas/extraction.py — Output schema for Phase 3: Semantic Extraction.

Node types mirror the Neo4j schema defined in base/tools/ingest.py (NODE_FIELDS).
Each node is typed with a discriminated union so PydanticAI can validate the
correct fields per node type.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Node models — required fields match NODE_FIELDS in base/tools/ingest.py
# ---------------------------------------------------------------------------


class DecisionNode(BaseModel):
    type: Literal["Decision"] = "Decision"
    id: str = Field(description="Slug ID: 'decision:{slug}' e.g. 'decision:use-postgres-for-sessions'.")
    label: str = Field(description="Short human-readable label.")
    scope: str = Field(default="system", description="Scope: 'system', 'subsystem:X', or 'module:X'.")
    trigger: str = Field(default="", description="What prompted this decision.")
    rationale: str = Field(description="Why this choice was made.")
    alternatives: list[dict[str, str]] = Field(
        default_factory=list,
        description="Alternatives considered. Each dict may have 'option' and 'rejected_because' keys.",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: 'high', 'medium', 'low'.",
    )
    uncertainty: str | None = Field(default=None, description="Known unknowns, if any.")
    source_doc: str | None = Field(default=None, description="Source document ID.")
    source_doc_id: str | None = Field(default=None, description="Canonical source document ID.")
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node (e.g. 'cursor', 'claude', 'notion').")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class ConstraintNode(BaseModel):
    type: Literal["Constraint"] = "Constraint"
    id: str = Field(description="Slug ID: 'constraint:{slug}'.")
    label: str
    scope: str = Field(default="system")
    description: str = Field(description="What the constraint is and why it exists.")
    source: str = Field(default="", description="What imposes this (library, API, policy, org, person, budget).")
    hard: bool = Field(default=False, description="True if this cannot be worked around.")
    symptom: str | None = Field(
        default=None,
        description="Observable symptom when this constraint is violated.",
    )
    source_doc: str | None = Field(default=None)
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node.")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class InsightNode(BaseModel):
    type: Literal["Insight"] = "Insight"
    id: str = Field(description="Slug ID: 'insight:{slug}'.")
    label: str
    scope: str = Field(default="system")
    body: str = Field(description="The knowledge, observation, or learning.")
    failure_without: str = Field(
        default="",
        description="Impact of not knowing this (risk, time, cost, missed opportunity).",
    )
    source_doc: str | None = Field(default=None)
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node.")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class SignalNode(BaseModel):
    type: Literal["Signal"] = "Signal"
    id: str = Field(description="Slug ID: 'signal:{slug}'.")
    label: str
    scope: str = Field(default="system")
    observation: str = Field(description="The symptom, trend, or observable event.")
    appears_to_mean: str = Field(default="", description="The naive/wrong interpretation.")
    actually_means: str = Field(default="", description="The correct interpretation.")
    conditions: str = Field(default="", description="When/where this signal occurs.")
    evidence: str = Field(default="", description="How this was determined.")
    source_doc: str | None = Field(default=None)
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node.")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class TopicNode(BaseModel):
    type: Literal["Topic"] = "Topic"
    id: str = Field(description="Slug ID: 'topic:{slug}'.")
    label: str
    scope: str = Field(default="system")
    description: str | None = Field(default=None)
    source: str = Field(default="document-extraction")
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node.")
    created_at: str | None = Field(default=None)


class EntityNode(BaseModel):
    type: Literal["Entity"] = "Entity"
    id: str = Field(
        description=(
            "Slug ID: 'entity:{entity_type}:{org_slug}:{name_slug}' "
            "e.g. 'entity:person:acme-corp:jane-doe'."
        ),
    )
    label: str
    entity_type: str = Field(
        description="One of: person, service, organization, repository, tool, object, location, event.",
    )
    canonical_name: str = Field(description="Full canonical name (e.g., 'Jane Doe').")
    aliases: str = Field(
        default="[]",
        description="JSON array string of aliases (e.g., '[\"jane\", \"@jane\"]').",
    )
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node.")
    created_at: str | None = Field(default=None)


class PreferenceNode(BaseModel):
    type: Literal["Preference"] = "Preference"
    id: str = Field(description="Slug ID: 'preference:{slug}'.")
    category: str = Field(default="general")
    preference: str
    confidence: float = Field(default=0.7, description="Confidence level between 0 and 1.")
    context: str | None = Field(default=None)
    entity_id: str | None = Field(default=None, description="ID of the owning person Entity node.")
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    valid_from: str | None = Field(default=None, description="ISO8601 datetime string.")
    valid_until: str | None = Field(default=None, description="ISO8601 datetime string.")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class FactNode(BaseModel):
    type: Literal["Fact"] = "Fact"
    id: str = Field(description="Slug ID: 'fact:{slug}'.")
    statement: str = Field(description="Full natural language sentence expressing the fact.")
    category: str = Field(description="One of: ownership, timeline, metric, status, relationship, attribute, membership, process.")
    confidence: float = Field(default=0.8, description="Confidence level between 0 and 1.")
    valid_from: str | None = Field(default=None, description="ISO8601 datetime string — when fact became true.")
    valid_until: str | None = Field(default=None, description="ISO8601 datetime string — when fact stopped being true. Null means still valid.")
    source_doc_id: str | None = Field(default=None)
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


class ActionItemNode(BaseModel):
    type: Literal["ActionItem"] = "ActionItem"
    id: str = Field(description="Slug ID: 'actionitem:{slug}' e.g. 'actionitem:migrate-auth-to-jwt'.")
    label: str = Field(description="Short human-readable description of what needs to be done.")
    status: str = Field(
        default="open",
        description="Current status: 'open', 'in_progress', 'done', or 'cancelled'.",
    )
    due_date: str | None = Field(
        default=None,
        description="ISO8601 date string if a deadline was mentioned (e.g. '2026-07-01').",
    )
    source: str = Field(
        default="meeting",
        description="Where this action item originated (e.g. 'meeting', 'review').",
    )
    source_doc_id: str | None = Field(default=None, description="Source document ID.")
    source_doc_version: int | None = Field(default=None, ge=1)
    source_scope: str | None = Field(default=None, description="RBAC scope id for node visibility.")
    repo: str | None = Field(default=None, description="Canonical owner/repo source dimension.")
    doc_type: str | None = Field(default=None, description="Source document classification.")
    tag: str | None = Field(default=None)
    origin: str | None = Field(default=None, description="Connector or runtime that produced this node (e.g. 'fireflies').")
    created_at: str | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, exclude=True)


SemanticNode = Annotated[
    Union[DecisionNode, ConstraintNode, InsightNode, SignalNode, TopicNode, EntityNode, PreferenceNode, FactNode, ActionItemNode],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Edge model
# ---------------------------------------------------------------------------


class SemanticEdge(BaseModel):
    from_id: str = Field(description="ID of the source node.")
    rel: str = Field(
        description="Relationship type. Valid: SHAPED_BY, SUPERSEDES, ABANDONED, LIFTED, "
                    "CONTRADICTS, EXPLAINS, CAUSES, MEMBER_OF, SUBTOPIC_OF, MENTIONS, "
                    "HAS_PREFERENCE, APPLIES_TO, ABOUT, AFFECTS, OWNS, BLOCKS, INVOLVES, "
                    "AUTHORED_BY.",
    )
    to_id: str = Field(description="ID of the target node.")
    edge_context: str = Field(
        description="Mandatory. 1-2 sentences describing the relationship, readable in both directions.",
    )
    inferred: bool = Field(
        default=False,
        description="True if this relationship was inferred rather than explicitly stated.",
    )


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    nodes: list[SemanticNode] = Field(
        default_factory=list,
        description="Final list of all nodes extracted by the LLM. Every node is persisted and linked to the source Document.",
    )
    edges: list[SemanticEdge] = Field(
        default_factory=list,
        description="LLM-generated semantic edges between nodes. Nodes may be present even when they have no semantic edge.",
    )
    extraction_notes: str = Field(
        default="",
        description="Notes on anything ambiguous, skipped, or uncertain during extraction.",
    )


class ChunkExtractionOutput(BaseModel):
    """LLM output schema for chunk extraction with cumulative summary."""

    nodes: list[SemanticNode] = Field(default_factory=list)
    edges: list[SemanticEdge] = Field(default_factory=list)
    extraction_notes: str = Field(default="")
    chunk_summary: str = Field(
        default="",
        description="Condensed summary of decisions, insights, signals, constraints "
        "extracted from this chunk. Include entity names. Used as context "
        "for subsequent chunks.",
    )
