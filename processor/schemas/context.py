"""
processor/schemas/context.py — Output schema for Phase 2: Context Gathering.

The context agent traverses the knowledge graph and returns a structured
picture of existing nodes/relationships relevant to the document being processed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DiscoveredNode(BaseModel):
    node_id: str = Field(description="Full node ID (e.g., 'decision:use-postgres-for-sessions').")
    node_type: str = Field(description="Node label: Decision, Constraint, Insight, Signal, Entity, Topic.")
    label: str = Field(description="Human-readable label of the node.")
    scope: str = Field(default="", description="Scope where this node lives (system, subsystem:X, etc.).")
    key_fields: dict = Field(
        default_factory=dict,
        description="Relevant fields from the node (e.g., rationale, description, body). "
                    "Subset of full node properties — include only what Phase 3 needs.",
    )


class TraversalPath(BaseModel):
    start_node: str = Field(description="ID of the node where traversal started.")
    path: list[str] = Field(
        description="Sequence of edge labels and node IDs discovered. "
                    "e.g., ['--SHAPED_BY-->', 'constraint:no-vendor-lock-in', '--MEMBER_OF-->', 'topic:infra']",
    )
    end_node: str = Field(description="ID of the terminal node in this path.")
    significance: str = Field(
        description="Why this path matters for the document being processed.",
    )


class PotentialRelationship(BaseModel):
    existing_node_id: str = Field(description="ID of the already-existing node in the graph.")
    existing_node_label: str = Field(description="Label of the existing node.")
    relationship_type: str = Field(
        description="Proposed relationship type: SUPERSEDES, CONTRADICTS, SHAPED_BY, EXPLAINS, LIFTED, CAUSES.",
    )
    reasoning: str = Field(
        description="Explanation of why this relationship should be created.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this relationship is correct (0.0–1.0).",
    )


class TopicContext(BaseModel):
    topic_id: str | None = Field(
        default=None,
        description="Existing Topic node ID if this topic was found in the graph.",
    )
    topic_label: str = Field(description="Human-readable topic label.")
    related_decisions: list[str] = Field(
        default_factory=list,
        description="Node IDs of existing Decision nodes connected to this topic.",
    )
    related_constraints: list[str] = Field(
        default_factory=list,
        description="Node IDs of existing Constraint nodes connected to this topic.",
    )
    decision_chain_summary: str = Field(
        default="",
        description="Short summary of how decisions in this topic connect via SHAPED_BY chains.",
    )


class ContextGatheringResult(BaseModel):
    topics: list[TopicContext] = Field(
        default_factory=list,
        description="Topics identified from the document and their graph neighborhoods.",
    )
    discovered_nodes: list[DiscoveredNode] = Field(
        default_factory=list,
        description="All relevant nodes discovered through traversal.",
    )
    significant_paths: list[TraversalPath] = Field(
        default_factory=list,
        description="Key traversal paths discovered (decision chains, constraint relationships).",
    )
    potential_supersedes: list[PotentialRelationship] = Field(
        default_factory=list,
        description="Existing nodes that the document's new nodes may supersede.",
    )
    potential_contradicts: list[PotentialRelationship] = Field(
        default_factory=list,
        description="Existing constraints that conflict with what's in the document.",
    )
    potential_shaped_by: list[PotentialRelationship] = Field(
        default_factory=list,
        description="Existing constraints that likely shaped decisions in this document.",
    )
    potential_explains: list[PotentialRelationship] = Field(
        default_factory=list,
        description="Existing insights that clarify nodes expected from this document.",
    )
    context_summary: str = Field(
        description="Narrative summary of the graph state relevant to this document.",
    )
    extraction_guidance: str = Field(
        description="Specific guidance for Phase 3: what to extract, what to link, what to avoid duplicating.",
    )
