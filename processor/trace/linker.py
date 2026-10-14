"""
processor/trace/linker.py — LLM-driven edge discovery for trace nodes.

Uses a PydanticAI agent with a single `find_related_nodes` tool that delegates
to EidetixClient for vector/graph search. Discovers SUPERSEDES, CONTRADICTS,
and SHAPED_BY edges.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel
from pydantic_ai import Agent, Tool
from pydantic_ai.usage import UsageLimits

from processor.agents.base import create_agent, run_agent
from processor.config import config
from processor.schemas.extraction import SemanticEdge, SemanticNode
from processor.tools.eidetix import EidetixClient

_log = logging.getLogger(__name__)

LINKER_SYSTEM_PROMPT = """\
You are a knowledge-graph linker. Given a new node just extracted from a trace, \
your job is to discover edges to existing nodes in the graph.

Use the `find_related_nodes` tool to search for related nodes. Then determine \
which relationships exist. Only create edges for these three relationship types:
- SUPERSEDES: the new node replaces or updates an existing decision/constraint/insight
- CONTRADICTS: the new node conflicts with an existing node
- SHAPED_BY: the new node was influenced by an existing node

Only create edges when you are confident a relationship exists (relevance score > 0.7).
Return an empty list if no strong relationships are found.
"""


class _LinkerResult(BaseModel):
    edges: list[SemanticEdge] = []


def _build_linker_agent() -> Agent[None, _LinkerResult]:
    async def find_related_nodes(
        node_summary: str,
        node_type: str,
        keywords: list[str],
        relationship_type: str,
    ) -> str:
        """
        Search the knowledge graph for nodes related to the given summary and type.

        Args:
            node_summary:      Summary or description of the node to match against.
            node_type:         Type of the source node (e.g. Decision, Insight).
            keywords:          Keywords to enrich the search query.
            relationship_type: Intended edge type (SUPERSEDES, CONTRADICTS, SHAPED_BY).

        Returns:
            JSON array of candidate node objects with id, label, type, relevance_score.
        """
        query = f"{node_summary} {' '.join(keywords)}".strip()

        try:
            async with EidetixClient() as client:
                context = await client.search_context(query=query)
        except Exception as exc:
            _log.warning("find_related_nodes search failed: %s", exc)
            return json.dumps([])

        candidates: list[dict] = context.get("results") or []
        if not candidates:
            return json.dumps([])

        max_score: float = max(
            (float(n.get("score", 0)) for n in candidates if n.get("score")),
            default=1.0,
        ) or 1.0

        results = []
        for node in candidates[:10]:
            raw_score = float(node.get("score", 0))
            relevance_score = round(raw_score / max_score, 4)
            results.append({
                "id": node.get("id", ""),
                "label": node.get("label", ""),
                "type": node.get("type", ""),
                "relevance_score": relevance_score,
            })

        return json.dumps(results)

    tool = Tool(
        find_related_nodes,
        name="find_related_nodes",
        description=(
            "Search the knowledge graph for nodes related to the given summary. "
            "Returns ranked candidates with relevance_score in [0, 1]. "
            "Only create edges for candidates with relevance_score > 0.7."
        ),
    )

    return create_agent(
        config.trace_linker_model,
        output_type=_LinkerResult,
        system_prompt=LINKER_SYSTEM_PROMPT,
        tools=[tool],
    )


_linker_agent: Agent[None, _LinkerResult] | None = None


def _get_linker_agent() -> Agent[None, _LinkerResult]:
    global _linker_agent
    if _linker_agent is None:
        _linker_agent = _build_linker_agent()
    return _linker_agent


async def discover_edges(node: SemanticNode, trace: dict) -> list[SemanticEdge]:
    """
    Use an LLM agent to discover SUPERSEDES, CONTRADICTS, and SHAPED_BY edges
    from the new node to existing graph nodes.

    Args:
        node:  The freshly-transformed SemanticNode.
        trace: The raw trace dict (provides keywords and summary).

    Returns:
        List of SemanticEdge instances. Empty list if nothing confident found.
    """
    node_type = getattr(node, "type", "")
    node_id = getattr(node, "id", "")
    label = getattr(node, "label", "")
    keywords: list[str] = trace.get("keywords") or []

    prompt = (
        f"New node:\n"
        f"  id: {node_id}\n"
        f"  type: {node_type}\n"
        f"  label: {label}\n"
        f"  keywords: {', '.join(keywords)}\n\n"
        f"Search for existing nodes this new node might SUPERSEDE, CONTRADICT, "
        f"or have been SHAPED_BY. Return only high-confidence edges."
    )

    agent = _get_linker_agent()
    try:
        result = await run_agent(
            agent,
            prompt,
            estimated_tokens=2000,
            call_site="trace_linker",
            usage_limits=UsageLimits(request_limit=config.trace_linker_max_iterations),
        )
        edges = result.output.edges
    except Exception as exc:
        _log.warning("trace linker agent failed for node=%s: %s", node_id, exc)
        return []

    allowed_rels = {"SUPERSEDES", "CONTRADICTS", "SHAPED_BY"}
    return [e for e in edges if e.rel in allowed_rels]
