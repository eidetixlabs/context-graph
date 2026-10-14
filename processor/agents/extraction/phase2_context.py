"""
processor/agents/extraction/phase2_context.py — Phase 2: Context Gathering.

The context agent traverses the knowledge graph using tools:
  - search            — find relevant existing memory by topic/keyword
  - get_graph         — map related nodes around selected IDs
  - get_graph_expanded — read rich content for a focused set of IDs
  - get_document_chunk — fetch a specific raw chunk by index (safety net)

Raw document content is NOT sent upfront. The agent receives keywords extracted
from each chunk + prefetched graph results. It can request individual chunks via
get_document_chunk if more detail is needed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import cast

from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from processor.agents.base import create_agent, run_agent
from processor.config import config
from processor.prompts import get_prompt
from processor.resolution.entity_quality import MIN_MENTION_CONFIDENCE, is_junk_mention
from processor.resolution.metadata import DocumentMetadata
from processor.schemas.context import ContextGatheringResult, DiscoveredNode, TopicContext
from processor.schemas.identity import IdentityResolutionResult
from processor.schemas.scoring import ScoringResult
from processor.state_machine import DocumentRecord
from processor.tools.chunk_reader import (
    get_document_chunk,
    register_chunks,
    unregister_chunks,
)
from base.tools.query import PRIMARY_CONTENT_FIELDS
from processor.tools.eidetix import EidetixClient, mcp_server

_log = logging.getLogger(__name__)

_CONTEXT_TOOLS = {
    "search",
    "get_graph",
    "get_graph_expanded",
    "get_content",
}


def _get_context_agent(
    doc_id: str,
    chunks: list[dict],
) -> Agent[str, ContextGatheringResult]:
    """Build a per-call context agent with the chunk tool bound to this doc's chunks."""
    from pydantic_ai import Tool  # noqa: PLC0415

    def _chunk_tool(chunk_index: int) -> str:
        """Return text of document chunk at chunk_index. Use when keywords alone are insufficient."""
        return get_document_chunk(doc_id, chunk_index)

    chunk_tool = Tool(
        _chunk_tool,
        name="get_document_chunk",
        description=(
            "Fetch the raw text of a specific document chunk by zero-based index. "
            f"Valid range: 0–{max(0, len(chunks) - 1)}. "
            "Use sparingly — prefer graph tools and keywords first."
        ),
    )

    return cast(Agent[str, ContextGatheringResult], create_agent(
        config.extraction_model,
        output_type=ContextGatheringResult,
        system_prompt=get_prompt("context"),
        toolsets=[mcp_server.filtered(lambda _ctx, tool: tool.name in _CONTEXT_TOOLS)],
        tools=[chunk_tool],
        output_retries=2,
    ))


async def run_context_gathering(
    doc: DocumentRecord,
    chunks: list[dict],
    keywords: list[str],
    identity_result: IdentityResolutionResult,
    scoring_result: ScoringResult,
    project: str,
    metadata: DocumentMetadata | None = None,
) -> ContextGatheringResult:
    """
    Execute Phase 2: traverse the graph to gather context for extraction.

    Raw document content is not included in the prompt. The agent receives
    keywords extracted during Phase 1 enrichment, prefetched graph context,
    and document metadata. It can request specific chunk text via the
    get_document_chunk tool if needed.

    Args:
        doc:             Document record from Postgres.
        chunks:          Pre-computed chunks with embeddings (for chunk tool + prefetch).
        keywords:        Aggregated keywords from LLM enrichment across all chunks.
        identity_result: Output from Phase 1 (resolved entity IDs).
        scoring_result:  Output from scoring (signal types + suggested topics).
        project:         Project slug.
        metadata:        Optional document metadata (author, most-frequent entities).

    Returns:
        ContextGatheringResult with discovered nodes, paths, and potential
        relationships for Phase 3 to use.
    """
    register_chunks(doc.id, chunks)
    try:
        prefetched = await _prefetch_context(
            identity_result=identity_result,
            scoring_result=scoring_result,
            project=project,
            doc_id=doc.id,
            keywords=keywords,
        )

        if config.skip_context_llm:
            ctx = _prefetch_to_context_result(prefetched, scoring_result)
            _log.info(
                "Phase 2 complete (prefetch-only): doc=%s discovered_nodes=%d topics=%d",
                doc.id,
                len(ctx.discovered_nodes),
                len(ctx.topics),
            )
            return ctx

        user_prompt, prompt_sections = _build_user_prompt(
            doc=doc,
            chunks=chunks,
            keywords=keywords,
            identity_result=identity_result,
            scoring_result=scoring_result,
            project=project,
            prefetched=prefetched,
            metadata=metadata,
        )

        _log.debug(
            "Phase 2 context gathering: doc=%s model=%s topics=%s keywords=%d prefetched=%s",
            doc.id,
            config.extraction_model,
            scoring_result.suggested_topics,
            len(keywords),
            prefetched is not None,
        )

        agent = _get_context_agent(doc.id, chunks)
        result = await run_agent(
            agent,
            user_prompt,
            estimated_tokens=len(user_prompt) // 3 + 1500,
            call_site="phase2_context",
            prompt_sections=prompt_sections,
            profile_system_prompt=get_prompt("context"),
            deps=project,
            usage_limits=UsageLimits(request_limit=6),
        )
        ctx = result.output

        _log.info(
            "Phase 2 complete: doc=%s discovered_nodes=%d potential_supersedes=%d "
            "potential_contradicts=%d topics=%d",
            doc.id,
            len(ctx.discovered_nodes),
            len(ctx.potential_supersedes),
            len(ctx.potential_contradicts),
            len(ctx.topics),
        )

        return ctx
    finally:
        unregister_chunks(doc.id)


async def _prefetch_context(
    identity_result: IdentityResolutionResult,
    scoring_result: ScoringResult,
    project: str,
    doc_id: str,
    keywords: list[str],
) -> dict | None:
    """
    Best-effort pre-fetch of broad context before Phase 2 agent is invoked.

    Uses suggested topics + top keywords for text search, then expands via
    redesign graph tools through gather_context_batch.
    Falls through on any error.

    Args:
        keywords: Aggregated keywords from LLM enrichment (top 5 appended to query).
    """
    resolved_ids = list(dict.fromkeys(
        i.resolved_id
        for i in identity_result.identities
        if i.resolved_id
    ))[:30]
    if identity_result.document_author and identity_result.document_author.resolved_id:
        author_id = identity_result.document_author.resolved_id
        if author_id not in resolved_ids:
            resolved_ids.append(author_id)
    resolved_ids = resolved_ids[:30]

    # Build query from suggested topics + top keywords
    topic = next(iter(scoring_result.suggested_topics), None)
    if not topic and not keywords:
        return None

    query_parts = []
    if topic:
        query_parts.append(topic)
    if keywords:
        query_parts.extend(keywords[:5])
    query = " ".join(query_parts)

    _log.info(
        "Phase 2 pre-fetch: doc=%s query=%r resolved_ids=%d keywords=%d",
        doc_id,
        query[:80],
        len(resolved_ids),
        len(keywords),
    )

    try:
        async with EidetixClient() as client:
            result = await client.gather_context_batch(
                query=query,
                project=project,
                node_ids=resolved_ids or None,
            )
        if result:
            _log.info("Phase 2 pre-fetch succeeded: doc=%s", doc_id)
            return result
    except Exception as exc:
        _log.debug("Phase 2 pre-fetch failed (falling through): doc=%s error=%s", doc_id, exc)
    return None


# Content-bearing fields carried into the Phase 3 prompt.
# Derived from PRIMARY_CONTENT_FIELDS (the authoritative per-type primary field
# names) plus "content" and "snippet" — the generic keys emitted by
# get_graph_expanded and search respectively. Every other property (timestamps,
# provenance, internal flags) is dropped. Fields are kept WHOLE — never
# truncated; a field is either retained in full or omitted entirely.
_CONTENT_KEY_FIELDS: frozenset[str] = frozenset(PRIMARY_CONTENT_FIELDS.values()) | {"content", "snippet"}

# Cap on discovered nodes carried into context_json. Dense graphs can return
# 1000+ nodes from a single-hop expansion; we rank then keep the top slice.
_MAX_DISCOVERED_NODES = 120

# Relevance weighting used to rank discovered nodes before capping.
# Source weight reflects how the node was found; type priority favours durable
# reasoning nodes over raw entities. These form the relevance base.
_SOURCE_WEIGHT = {"search": 3.0, "subgraph": 2.0, "traversal": 1.0}
_NODE_TYPE_PRIORITY = {
    "decision": 3.0, "constraint": 3.0, "entity": 3.0, "insight": 2.5, "signal": 2.5,
    "topic": 1.5, "preference": 1.5, "fact": 1.5, 
}

# Recency only amplifies already-relevant nodes (multiplicative boost), so a
# recent-but-irrelevant node never outranks an older highly-relevant one.
# Half-life mirrors query._score_node's 180-day decay.
_RECENCY_HALFLIFE_DAYS = 180.0
_RECENCY_BOOST = 0.5
_RECENCY_DEFAULT_DAYS = 180  # used when created_at is missing/unparseable

def _node_label(node: dict) -> str:
    for key in ("label", "name", "title", "canonical_name", "text"):
        val = node.get(key)
        if val:
            return str(val)[:200]
    return str(node.get("id", ""))[:200]


def _recency_decay(node: dict) -> float:
    """Return a 0..1 recency factor from the node's created_at (1.0 = brand new)."""
    created_at = node.get("created_at")
    days_old = _RECENCY_DEFAULT_DAYS
    if created_at:
        try:
            if isinstance(created_at, str):
                ca = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                ca = created_at
            if ca.tzinfo is None:
                ca = ca.replace(tzinfo=timezone.utc)
            days_old = max((datetime.now(tz=timezone.utc) - ca).days, 0)
        except Exception:
            days_old = _RECENCY_DEFAULT_DAYS
    return 1.0 / (1.0 + days_old / _RECENCY_HALFLIFE_DAYS)


def _shape_key_fields(node: dict) -> dict:
    """Keep only content-bearing fields, each retained in full (no truncation)."""
    return {
        key: value
        for key, value in node.items()
        if key in _CONTENT_KEY_FIELDS and value is not None and value != ""
    }


def _candidate_score(node: dict, source: str) -> float:
    """Relevance base (source + type + search score) amplified by recency."""
    node_type = str(node.get("_type") or node.get("type") or "").lower()
    relevance = _SOURCE_WEIGHT.get(source, 1.0) + _NODE_TYPE_PRIORITY.get(node_type, 0.5)
    try:
        relevance += float(node.get("rrf_score") or node.get("score") or 0.0)
    except (TypeError, ValueError):
        pass
    return relevance * (1.0 + _RECENCY_BOOST * _recency_decay(node))


def slim_context_for_prompt(context_result: ContextGatheringResult) -> ContextGatheringResult:
    """Cap discovered nodes and strip non-content key_fields before serialization.

    Idempotent for the prefetch-only path (already shaped). Acts as a safety net
    for the Phase 2 LLM path, whose output can carry many richly-populated nodes
    that would otherwise bloat context_json in every chunk prompt.
    """
    before_count = len(context_result.discovered_nodes)
    slimmed_nodes = [
        DiscoveredNode(
            node_id=n.node_id,
            node_type=n.node_type,
            label=n.label,
            scope=n.scope,
            key_fields={k: v for k, v in n.key_fields.items() if k in _CONTENT_KEY_FIELDS},
        )
        for n in context_result.discovered_nodes[:_MAX_DISCOVERED_NODES]
    ]
    after_count = len(slimmed_nodes)
    if before_count != after_count:
        _log.info(
            "context_json pruned: nodes %d → %d (dropped %d)",
            before_count, after_count, before_count - after_count,
        )
    return context_result.model_copy(update={"discovered_nodes": slimmed_nodes})


def _prefetch_node_to_discovered(node: dict) -> DiscoveredNode | None:
    node_id = node.get("id") or ""
    if not node_id:
        return None
    node_type = node.get("_type") or node.get("type") or "Unknown"
    return DiscoveredNode(
        node_id=node_id,
        node_type=node_type,
        label=_node_label(node),
        scope=str(node.get("scope") or ""),
        key_fields=_shape_key_fields(node),
    )


def _prefetch_to_context_result(
    prefetched: dict | None,
    scoring_result: ScoringResult,
) -> ContextGatheringResult:
    """Build a ContextGatheringResult from prefetch data without an LLM call.

    Nodes are merged across sources (search/subgraph/traversal), scored by
    relevance × recency, then capped to the top _MAX_DISCOVERED_NODES so the
    serialized context_json stays small on dense graphs.
    """
    # Merge raw fields per id across sources and track the best candidate score.
    raw_by_id: dict[str, dict] = {}
    best_score: dict[str, float] = {}

    def _add_node(raw: dict, source: str) -> None:
        nid = raw.get("id") or ""
        if not nid:
            return
        merged = raw_by_id.get(nid)
        if merged is None:
            raw_by_id[nid] = dict(raw)
        else:
            # Fill missing/empty fields without clobbering existing content.
            for key, value in raw.items():
                if value not in (None, "") and merged.get(key) in (None, ""):
                    merged[key] = value
        score = _candidate_score(raw, source)
        if score > best_score.get(nid, float("-inf")):
            best_score[nid] = score

    # Collect traversal edges for topic→decision/constraint linkage
    traversal_edges: list[dict] = []

    if prefetched:
        subgraph = prefetched.get("subgraph") or {}
        for raw in subgraph.get("nodes", []):
            _add_node(raw, "subgraph")

        search = prefetched.get("search_results") or {}
        for raw in search.get("merged", []):
            _add_node(raw, "search")

        traversal = prefetched.get("traversal") or {}
        for result in traversal.get("results") or []:
            for raw in result.get("nodes") or []:
                _add_node(raw, "traversal")
            for edge in result.get("edges") or []:
                traversal_edges.append(edge)

    # Rank by relevance × recency, keep the top slice, then build nodes.
    total_candidates = len(raw_by_id)
    ranked_ids = sorted(
        raw_by_id,
        key=lambda nid: best_score.get(nid, 0.0),
        reverse=True,
    )[:_MAX_DISCOVERED_NODES]

    discovered: list[DiscoveredNode] = []
    for nid in ranked_ids:
        node = _prefetch_node_to_discovered(raw_by_id[nid])
        if node is not None:
            discovered.append(node)

    if total_candidates > len(discovered):
        _log.info(
            "Phase 2 prefetch pruned: candidates=%d → discovered_nodes=%d (dropped %d)",
            total_candidates, len(discovered), total_candidates - len(discovered),
        )

    topics = _build_topic_contexts(
        scoring_result.suggested_topics,
        raw_by_id,
        traversal_edges,
    )

    summary = (
        f"Prefetch-only context: {len(discovered)} nodes from graph search/traversal."
        if discovered
        else "No prefetch data available."
    )
    guidance = (
        "Use discovered_nodes to avoid duplicates and link to existing graph nodes. "
        "No supersedes/contradicts analysis (Phase 2 LLM skipped)."
    )

    return ContextGatheringResult(
        topics=topics,
        discovered_nodes=discovered,
        context_summary=summary,
        extraction_guidance=guidance,
    )


def _build_topic_contexts(
    suggested_topics: list[str],
    raw_by_id: dict[str, dict],
    traversal_edges: list[dict],
) -> list[TopicContext]:
    """
    Build TopicContext entries from scoring topics.

    Matches topic nodes from prefetch by label similarity (case-insensitive
    substring), then walks traversal edges to populate related_decisions
    and related_constraints.
    """
    # Index Topic nodes from prefetch by normalised label
    topic_nodes: dict[str, dict] = {
        nid: raw
        for nid, raw in raw_by_id.items()
        if (raw.get("_type") or raw.get("type") or "").lower() == "topic"
    }

    # Build adjacency from traversal edges: node_id → set of neighbour ids
    adjacency: dict[str, set[str]] = {}
    for edge in traversal_edges:
        src = edge.get("from") or edge.get("start") or ""
        tgt = edge.get("to") or edge.get("end") or ""
        if src and tgt:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

    result: list[TopicContext] = []
    for topic_label in suggested_topics:
        label_lower = topic_label.lower()

        # Find best-matching topic node by label substring
        matched_topic_id: str | None = None
        for nid, raw in topic_nodes.items():
            graph_label = _node_label(raw).lower()
            if label_lower in graph_label or graph_label in label_lower:
                matched_topic_id = nid
                break

        related_decisions: list[str] = []
        related_constraints: list[str] = []

        if matched_topic_id:
            for neighbour_id in adjacency.get(matched_topic_id, set()):
                neighbour = raw_by_id.get(neighbour_id)
                if not neighbour:
                    continue
                ntype = (neighbour.get("_type") or neighbour.get("type") or "").lower()
                if ntype == "decision":
                    related_decisions.append(neighbour_id)
                elif ntype == "constraint":
                    related_constraints.append(neighbour_id)

        result.append(TopicContext(
            topic_id=matched_topic_id,
            topic_label=topic_label,
            related_decisions=related_decisions,
            related_constraints=related_constraints,
        ))

    return result


def _build_user_prompt(
    doc: DocumentRecord,
    chunks: list[dict],
    keywords: list[str],
    identity_result: IdentityResolutionResult,
    scoring_result: ScoringResult,
    project: str,
    prefetched: dict | None = None,
    metadata: DocumentMetadata | None = None,
) -> tuple[str, dict[str, str]]:
    resolved_entities_text = _format_resolved_entities(identity_result)
    signal_types_text = ", ".join(scoring_result.signal_types) or "unknown"
    topics_text = ", ".join(scoring_result.suggested_topics) or "(none detected)"
    keywords_text = "\n".join(f"- {kw}" for kw in keywords) if keywords else "(none)"

    metadata_section = ""
    if metadata:
        parts = []
        if metadata.author:
            parts.append(f"- Author: {metadata.author}")
        if metadata.person_most_count:
            parts.append(f"- Most frequent person: {metadata.person_most_count}")
        if metadata.topic_most_count:
            parts.append(f"- Primary topic (by keyword frequency): {metadata.topic_most_count}")
        if parts:
            metadata_section = "Document Metadata:\n" + "\n".join(parts) + "\n\n"

    chunk_index_hint = (
        f"Document has {len(chunks)} chunk(s) (indices 0–{len(chunks) - 1}). "
        "Use get_document_chunk(index) to read a specific chunk if needed."
        if chunks else ""
    )

    prefetch_section = ""
    if prefetched:
        subgraph = prefetched.get("subgraph") or {}
        search = prefetched.get("search_results") or {}
        traversal = prefetched.get("traversal")

        subgraph_nodes = subgraph.get("nodes", [])
        subgraph_edges = subgraph.get("edges", [])
        merged_nodes = search.get("merged", [])

        parts = []
        if subgraph_nodes:
            parts.append(
                f"Subgraph ({len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges):\n"
                + json.dumps({"nodes": subgraph_nodes[:30], "edges": subgraph_edges[:40]}, default=str)
            )
        if merged_nodes:
            parts.append(
                f"Hybrid search ({len(merged_nodes)} merged nodes):\n"
                + json.dumps(merged_nodes[:20], default=str)
            )
        if traversal and traversal.get("results"):
            parts.append("Entity traversal:\n" + json.dumps(traversal["results"], default=str))

        if parts:
            prefetch_section = (
                "\n---\nPre-fetched Context (already gathered — review before calling tools):\n"
                + "\n\n".join(parts)
                + "\n"
            )

    instructions = (
        "1. Review pre-fetched context above for relevant nodes/edges.\n"
        "2. Use search/get_graph/get_graph_expanded for additional graph context.\n"
        f"3. {chunk_index_hint}"
    )

    sections = {
        "metadata": metadata_section,
        "keywords": keywords_text,
        "signal_types": signal_types_text,
        "topics": topics_text,
        "resolved_entities": resolved_entities_text,
        "prefetch": prefetch_section,
        "instructions": instructions,
    }

    _log.info(
        "Phase 2 user prompt: doc=%s project=%r signal_types=%r topics=%r "
        "keywords=%d resolved_entities=%r prefetch=%s",
        doc.id,
        project,
        signal_types_text,
        topics_text,
        len(keywords),
        resolved_entities_text[:120],
        "yes" if prefetch_section else "no",
    )

    prompt = f"""{metadata_section}Keywords extracted from document (use for graph search):
{keywords_text}

---
Signal Types Detected (from scoring):
{signal_types_text}

Suggested Topics (from scoring):
{topics_text}

---
Resolved Identities from Phase 1:
{resolved_entities_text}
{prefetch_section}
---
Instructions:
{instructions}
"""
    return prompt, sections


def _format_resolved_entities(identity_result: IdentityResolutionResult) -> str:
    existing_mentions = {i.mention for i in identity_result.identities}
    lines = []
    for identity in identity_result.identities:
        if identity.confidence < MIN_MENTION_CONFIDENCE:
            continue
        if is_junk_mention(identity.mention, existing_mentions):
            continue
        if identity.resolved_id:
            lines.append(f"{identity.mention} → {identity.resolved_id}")
        elif identity.is_new:
            lines.append(f"{identity.mention} → NEW [{identity.entity_type}]")
    if identity_result.document_author:
        a = identity_result.document_author
        author_id = a.resolved_id or f"NEW:{a.proposed_canonical_name}"
        lines.append(f"AUTHOR: {a.mention} → {author_id}")
    return "\n".join(lines) if lines else "(no identities resolved)"
