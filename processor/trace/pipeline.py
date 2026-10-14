"""
processor/trace/pipeline.py — Three-phase trace processing pipeline.

Phase 1 — Entity Resolution (deterministic):
    Converts trace.mentions → MentionCandidates → ResolutionMap via DB lookup.

Phase 2 — Context Prefetch:
    Seeds graph context from trace.keywords via EidetixClient.

Phase 3 — Transform + Edge Discovery:
    trace_to_node → AUTHORED_BY + MENTIONS edges → LLM linker for
    SUPERSEDES / CONTRADICTS / SHAPED_BY edges → dual_write.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from base.db.client import db_pool
from base.neo4j_client import Neo4jClient
from base.tools.ingest import ingest_entries
from processor.agents.base import create_agent, run_agent
from processor.config import config
from processor.dual_write import _upsert_semantic_nodes
from processor.resolution.models import MentionCandidate, ResolutionMap
from processor.resolution.persistence import persist_trace_new_entities_from_map
from processor.resolution.resolver import resolve_trace_mentions
from processor.schemas.extraction import ExtractionResult, SemanticEdge, SemanticNode
from processor.tools.eidetix import EidetixClient
from processor.trace.linker import discover_edges
from processor.trace.transformer import trace_to_node

_log = logging.getLogger(__name__)

_TRACE_DEFAULT_MENTION_CONFIDENCE = 0.92
_TRACE_MAX_ACCEPTED_MENTIONS = 5
_TRACE_LLM_FILTER_MAX_CANDIDATES = 20
_TRACE_ALLOWED_MENTION_TYPES = {
    "person",
    "service",
    "org",
    "repo",
    "tool",
}
_TRACE_LLM_FILTER_PROMPT = """
You are a mention quality gate for trace ingestion.

Rules:
- Accept only explicit, specific, high-signal named entities.
- Reject generic nouns, abstract phrases, low-context terms, and ambiguous mentions.
- If uncertain, reject.
- Keep reject_reason concise and deterministic.

Allowed mention_type values in this task: person, service, org, repo, tool.
""".strip()


def _trace_source_scope(trace: dict[str, Any]) -> str:
    author_employee_id = str(trace.get("_author_employee_id") or "").strip()
    if author_employee_id:
        return f"scope:trace:{author_employee_id}"
    return "scope:trace:unattributed"


class _TraceMentionDecision(BaseModel):
    mention_text: str
    mention_type: str
    accept: bool = False
    reject_reason: str = ""


class _TraceMentionFilterResult(BaseModel):
    decisions: list[_TraceMentionDecision] = Field(default_factory=list)

def _trace_mentions_to_candidates(trace: dict) -> list[MentionCandidate]:
    """Convert trace.mentions list -> list[MentionCandidate]."""
    mentions_raw = trace.get("mentions") or []
    candidates: list[MentionCandidate] = []
    valid_mention_types = _TRACE_ALLOWED_MENTION_TYPES

    if isinstance(mentions_raw, list):
        for item in mentions_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("mention_text")
            mention_type = item.get("mention_type")
            if not name or not isinstance(name, str):
                continue
            if not mention_type or not isinstance(mention_type, str):
                continue
            if mention_type not in valid_mention_types:
                continue
            confidence_raw = item.get("confidence")
            confidence = _TRACE_DEFAULT_MENTION_CONFIDENCE
            if isinstance(confidence_raw, (int, float)):
                confidence = max(0.0, min(float(confidence_raw), 1.0))
            candidates.append(
                MentionCandidate(
                    mention_text=name,
                    mention_type=mention_type,  # type: ignore[arg-type]
                    chunk_index=0,
                    source="trace_ingest",
                    confidence=confidence,
                )
            )
        return candidates

    return candidates


def _author_name_from_trace(trace: dict) -> str:
    explicit_author_name = trace.get("_author_name")
    if isinstance(explicit_author_name, str) and explicit_author_name.strip():
        return explicit_author_name.strip()

    return next(
        (
            str(item.get("mention_text"))
            for item in (trace.get("mentions") or [])
            if isinstance(item, dict)
            and item.get("mention_type") == "person"
            and item.get("mention_text")
        ),
        "",
    )


def _build_author_mention_candidate(trace: dict) -> MentionCandidate | None:
    author_name = _author_name_from_trace(trace)
    if not author_name:
        return None

    evidence: dict[str, Any] = {}
    author_employee_id = trace.get("_author_employee_id")
    if isinstance(author_employee_id, str) and author_employee_id.strip():
        evidence["author_employee_id"] = author_employee_id.strip()

    return MentionCandidate(
        mention_text=author_name,
        mention_type="person",
        chunk_index=0,
        source="trace_author",
        confidence=1.0,
        evidence=evidence,
    )

def _deterministic_trace_filter(
    candidates: list[MentionCandidate],
) -> tuple[list[MentionCandidate], dict[str, int]]:
    accepted: list[MentionCandidate] = []
    rejected: dict[str, int] = {}
    for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
        if candidate.mention_type not in _TRACE_ALLOWED_MENTION_TYPES:
            rejected["mention_type_not_allowed"] = rejected.get("mention_type_not_allowed", 0) + 1
            continue
        accepted.append(candidate)

    if len(accepted) > _TRACE_MAX_ACCEPTED_MENTIONS:
        dropped = len(accepted) - _TRACE_MAX_ACCEPTED_MENTIONS
        accepted = accepted[:_TRACE_MAX_ACCEPTED_MENTIONS]
        rejected["top_k_cap"] = rejected.get("top_k_cap", 0) + dropped

    return accepted, rejected


async def _llm_trace_filter(
    candidates: list[MentionCandidate],
    trace_id: str,
) -> tuple[list[MentionCandidate], dict[str, int]]:
    if not candidates:
        return [], {}

    prompt_items = [
        {
            "mention_text": c.mention_text,
            "mention_type": c.mention_type,
        }
        for c in candidates[:_TRACE_LLM_FILTER_MAX_CANDIDATES]
    ]

    prompt = (
        "Evaluate each candidate and return a decision per item.\n"
        f"trace_id={trace_id}\n"
        f"candidates={prompt_items}"
    )

    try:
        agent = create_agent(
            config.entity_model,
            system_prompt=_TRACE_LLM_FILTER_PROMPT,
            output_type=_TraceMentionFilterResult,
            retries=1,
        )
        result = await run_agent(
            agent,
            prompt,
            estimated_tokens=450,
            call_site="trace.pipeline.mention_filter",
            prompt_sections={
                "trace_id": trace_id,
                "candidates": str(prompt_items),
            },
            profile_system_prompt=_TRACE_LLM_FILTER_PROMPT,
        )
    except Exception as exc:
        _log.warning("Trace mention LLM filter failed: trace_id=%s error=%s", trace_id, exc)
        return candidates, {"llm_filter_error_passthrough": len(candidates)}

    decision_by_key: dict[tuple[str, str], _TraceMentionDecision] = {}
    for decision in result.output.decisions:
        key = (decision.mention_text.strip().lower(), decision.mention_type.strip().lower())
        if key[0] and key[1] and key not in decision_by_key:
            decision_by_key[key] = decision

    accepted: list[MentionCandidate] = []
    rejected: dict[str, int] = {}
    for candidate in candidates:
        key = (candidate.mention_text.strip().lower(), candidate.mention_type.strip().lower())
        decision = decision_by_key.get(key)
        if decision and decision.accept:
            accepted.append(candidate)
            continue
        reason = (
            decision.reject_reason.strip()
            if decision and decision.reject_reason.strip()
            else "llm_default_reject"
        )
        rejected[reason] = rejected.get(reason, 0) + 1

    return accepted, rejected


async def _filter_trace_mentions(
    candidates: list[MentionCandidate],
    trace_id: str,
) -> tuple[list[MentionCandidate], dict[str, int]]:
    deterministic_accepted, deterministic_rejected = _deterministic_trace_filter(candidates)
    llm_accepted, llm_rejected = await _llm_trace_filter(deterministic_accepted, trace_id)

    combined_rejected = dict(deterministic_rejected)
    for reason, count in llm_rejected.items():
        combined_rejected[reason] = combined_rejected.get(reason, 0) + count

    deterministic_rejected_count = sum(deterministic_rejected.values())
    llm_accepted_count = len(llm_accepted)
    llm_rejected_count = sum(llm_rejected.values())

    metrics = {
        "total_candidates": len(candidates),
        "accepted_mentions": llm_accepted_count,
        "rejected_mentions": max(0, len(candidates) - llm_accepted_count),
        "deterministic_rejected": deterministic_rejected_count,
        "llm_accepted": llm_accepted_count,
        "llm_rejected": llm_rejected_count,
    }

    _log.info(
        "Trace mention filter: trace_id=%s total=%d accepted=%d rejected=%d "
        "deterministic_rejected=%d llm_accepted=%d llm_rejected=%d reasons=%s",
        trace_id,
        metrics["total_candidates"],
        metrics["accepted_mentions"],
        metrics["rejected_mentions"],
        metrics["deterministic_rejected"],
        metrics["llm_accepted"],
        metrics["llm_rejected"],
        combined_rejected,
    )
    return llm_accepted, metrics


def _normalize_node(raw: dict[str, Any]) -> dict[str, Any] | None:
    node_id = raw.get("id")
    node_type = raw.get("type")
    if not isinstance(node_id, str) or not node_id:
        return None
    if not isinstance(node_type, str) or not node_type:
        return None
    return raw


def _node_type_from_id(node_id: str) -> str:
    prefix = node_id.split(":", 1)[0].lower()
    type_map = {
        "decision": "Decision",
        "constraint": "Constraint",
        "insight": "Insight",
        "signal": "Signal",
        "topic": "Topic",
        "entity": "Entity",
        "preference": "Preference",
        "fact": "Fact",
    }
    return type_map.get(prefix, "Decision")


def _lookup_node_type(client: Neo4jClient, node_id: str) -> str:
    rows = client.run(
        "MATCH (n {id: $id}) RETURN labels(n)[0] AS label LIMIT 1",
        {"id": node_id},
    )
    if rows:
        label = rows[0].get("label")
        if isinstance(label, str) and label:
            return label
    return _node_type_from_id(node_id)


def _build_edge_entries(
    result: ExtractionResult,
    neo4j_client: Neo4jClient,
    source_scope: str,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat()
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in result.nodes:
        raw = node if isinstance(node, dict) else node.model_dump()
        normalized = _normalize_node(raw)
        if normalized is None:
            continue
        nodes_by_id[normalized["id"]] = normalized

    entries: list[dict[str, Any]] = []
    for edge in result.edges:
        from_raw = nodes_by_id.get(edge.from_id)
        if from_raw is None:
            from_raw = {
                "id": edge.from_id,
                "type": _lookup_node_type(neo4j_client, edge.from_id),
                "_stub": True,
            }
        to_raw = nodes_by_id.get(edge.to_id)
        if to_raw is None:
            to_raw = {
                "id": edge.to_id,
                "type": _lookup_node_type(neo4j_client, edge.to_id),
                "_stub": True,
            }

        entries.append(
            {
                "id": f"trace-edge:{edge.from_id}:{edge.rel}:{edge.to_id}",
                "created_at": now,
                "from": from_raw,
                "to": to_raw,
                "rel": edge.rel,
                "properties": {
                    "edge_context": edge.edge_context,
                    "inferred": edge.inferred,
                    "source_scope": source_scope,
                },
            }
        )
    return entries


def _write_trace_graph(
    neo4j_client: Neo4jClient,
    project: str,
    result: ExtractionResult,
    source_scope: str,
) -> dict[str, int]:
    nodes_upserted = _upsert_semantic_nodes(neo4j_client, result, project)

    rels_upserted = 0
    edge_entries = _build_edge_entries(result, neo4j_client, source_scope)
    if edge_entries:
        summary = ingest_entries(neo4j_client, project, edge_entries)
        rels_upserted = int(summary.get("rels_upserted", 0))

    return {
        "nodes_upserted": nodes_upserted,
        "rels_upserted": rels_upserted,
    }


async def run_trace_pipeline(
    trace: dict,
    neo4j_client: Neo4jClient,
) -> ExtractionResult:
    """
    Run the full three-phase trace processing pipeline.

    Args:
        trace:        Raw trace dict as stored in S3 (including _author_name).
        neo4j_client: Neo4j client for graph writes.

    Returns:
        ExtractionResult with all nodes and edges ready for persistence.
    """
    org_id: str = trace.get("org_id") or ""
    project: str = trace.get("project") or org_id.removeprefix("org:")
    trace_id: str = trace.get("trace_id") or ""

    _log.info("Trace pipeline start: trace_id=%s org_id=%s", trace_id, org_id)

    # ── Phase 1: Entity Resolution ────────────────────────────────────────────
    mention_candidates = _trace_mentions_to_candidates(trace)
    author_mention_candidate = _build_author_mention_candidate(trace)
    mentions, mention_metrics = await _filter_trace_mentions(mention_candidates, trace_id)
    if author_mention_candidate:
        mentions.append(author_mention_candidate)
    _log.info(
        "Trace pipeline cleared mentions: trace_id=%s mentions=%s",
        trace_id,
        [m.mention_text for m in mentions],
    )
    resolution_map: ResolutionMap

    pool = db_pool.get()
    async with pool.acquire() as conn:
        resolution_map = await resolve_trace_mentions(
            mentions=mentions,
            org_id=org_id,
            conn=conn,
        )
        resolution_map.doc_id = f"trace:{trace_id}"
        await persist_trace_new_entities_from_map(
            trace_id=trace_id,
            org_id=org_id,
            resolution_map=resolution_map,
            conn=conn,
        )

    _log.info(
        "Trace pipeline Phase 1 complete: trace_id=%s resolved=%d mention_candidates=%d mention_accepted=%d",
        trace_id,
        len(resolution_map.entries),
        mention_metrics["total_candidates"],
        mention_metrics["accepted_mentions"],
    )

    # ── Phase 2: Context Prefetch ─────────────────────────────────────────────
    keywords: list[str] = trace.get("keywords") or []
    if keywords:
        query = " ".join(keywords[:20])
        try:
            async with EidetixClient() as client:
                await client.search_context(query=query)
        except Exception as exc:
            _log.warning(
                "Trace pipeline Phase 2 context prefetch failed: trace_id=%s error=%s",
                trace_id,
                exc,
            )

    # ── Phase 3: Transform + Edge Discovery ───────────────────────────────────
    source_scope = _trace_source_scope(trace)
    node: SemanticNode = trace_to_node(trace)
    node_id: str = getattr(node, "id", "")

    edges: list[SemanticEdge] = []

    # The author is resolved like any other mention (by name) — employees do
    # not carry a pre-assigned entity_id, so we look the author up in the
    # resolution map produced above instead of trusting an ingest-time id.
    author_name: str = _author_name_from_trace(trace)
    author_entry = resolution_map.entries.get(author_name) if author_name else None
    author_entity_id: str = (author_entry.entity_id or "") if author_entry else ""

    if author_name and not author_entity_id:
        _log.warning(
            "Trace pipeline author entity unresolved after mention resolution: trace_id=%s author_name=%r",
            trace_id,
            author_name,
        )

    if author_entity_id:
        edges.append(
            SemanticEdge(
                from_id=node_id,
                rel="AUTHORED_BY",
                to_id=author_entity_id,
                edge_context=f"Trace {trace_id} was authored by {author_name}.",
                inferred=False,
            )
        )

    for mention_text, entry in resolution_map.entries.items():
        if entry.entity_id and entry.entity_id != author_entity_id:
            edges.append(
                SemanticEdge(
                    from_id=node_id,
                    rel="MENTIONS",
                    to_id=entry.entity_id,
                    edge_context=f"Trace node mentions {mention_text}.",
                    inferred=False,
                )
            )

    llm_edges = await discover_edges(node, trace)
    edges.extend(llm_edges)

    entity_stubs: list[SemanticNode] = []
    for entity in resolution_map.resolved_entities:
        from processor.schemas.extraction import EntityNode  # noqa: PLC0415
        entity_stubs.append(
            EntityNode(
                id=entity.entity_id,
                label=entity.canonical_name,
                entity_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                origin="trace_pipeline",
            )
        )

    result = ExtractionResult(
        nodes=[node, *entity_stubs],
        edges=edges,
    )

    _log.info(
        "Trace pipeline Phase 3 complete: trace_id=%s nodes=%d edges=%d llm_edges=%d",
        trace_id,
        len(result.nodes),
        len(result.edges),
        len(llm_edges),
    )

    try:
        _log.info(
            "Trace pipeline graph write attempt: trace_id=%s nodes=%d edges=%d",
            trace_id,
            len(result.nodes),
            len(result.edges),
        )
        write_summary = await asyncio.to_thread(
            _write_trace_graph,
            neo4j_client,
            project,
            result,
            source_scope,
        )
        _log.info(
            "Trace pipeline graph write: trace_id=%s summary=%s",
            trace_id,
            write_summary,
        )
    except Exception as exc:
        _log.warning("Trace pipeline graph write failed: trace_id=%s error=%s", trace_id, exc)

    return result
