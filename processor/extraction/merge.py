"""
processor/extraction/merge.py — Merge and dedup per-chunk extraction results.

Node dedup:    key (node.type, node.id) — keep higher confidence, merge evidence.
Edge dedup:    key (from_id, rel, to_id) — keep first-seen context, OR inferred flags.
Provenance:    source_chunks list per node.
Validation:    all valid LLM nodes are preserved; Document provenance edges are
               added later by dual_write.
"""

from __future__ import annotations

import logging
from typing import Any

from processor.resolution.models import ChunkExtractionResult
from processor.schemas.extraction import ExtractionResult, SemanticEdge

_log = logging.getLogger(__name__)


def _node_id(node: Any) -> str:
    if isinstance(node, dict):
        return node.get("id", "")
    return getattr(node, "id", "")


def _node_type(node: Any) -> str:
    if isinstance(node, dict):
        return node.get("type", "")
    return getattr(node, "type", "")


def _node_confidence(node: Any) -> float:
    if isinstance(node, dict):
        return float(node.get("confidence", 0.0))
    conf = getattr(node, "confidence", None)
    if conf is None:
        return 0.0
    if isinstance(conf, str):
        return {"high": 1.0, "medium": 0.65, "low": 0.3}.get(conf.lower(), 0.5)
    return float(conf)


def _merge_node_evidence(existing: Any, incoming: Any) -> None:
    """Merge evidence fields (description, body, rationale) from incoming into existing."""
    evidence_fields = ("description", "body", "rationale", "observation")
    for field in evidence_fields:
        if isinstance(existing, dict):
            if not existing.get(field) and (isinstance(incoming, dict) and incoming.get(field)):
                existing[field] = incoming[field] if isinstance(incoming, dict) else getattr(incoming, field, None)
        else:
            if not getattr(existing, field, None):
                incoming_val = incoming.get(field) if isinstance(incoming, dict) else getattr(incoming, field, None)
                if incoming_val:
                    try:
                        setattr(existing, field, incoming_val)
                    except Exception:
                        pass


def merge_chunk_extractions(
    chunk_results: list[ChunkExtractionResult],
) -> ExtractionResult:
    """
    Merge per-chunk ExtractionResults into a single deduplicated ExtractionResult.

    Nodes with no LLM edges are preserved. dual_write will attach every final
    node to the source Document with EXTRACTED_FROM.
    """
    if not chunk_results:
        return ExtractionResult(nodes=[], edges=[], extraction_notes="No chunks extracted.")

    node_map: dict[tuple[str, str], Any] = {}
    node_source_chunks: dict[tuple[str, str], list[int]] = {}
    edge_map: dict[tuple[str, str, str], SemanticEdge] = {}

    for chunk_result in chunk_results:
        chunk_idx = chunk_result.chunk_index
        extraction = chunk_result.extraction

        for node in extraction.nodes:
            nid = _node_id(node)
            ntype = _node_type(node)
            if not nid or not ntype:
                continue

            key = (ntype, nid)
            if key in node_map:
                existing = node_map[key]
                if _node_confidence(node) > _node_confidence(existing):
                    node_map[key] = node
                else:
                    _merge_node_evidence(existing, node)
                node_source_chunks[key].append(chunk_idx)
            else:
                node_map[key] = node
                node_source_chunks[key] = [chunk_idx]

        for edge in extraction.edges:
            ekey = (edge.from_id, edge.rel, edge.to_id)
            if ekey not in edge_map:
                edge_map[ekey] = edge
            else:
                existing_edge = edge_map[ekey]
                merged_inferred = existing_edge.inferred and edge.inferred
                edge_map[ekey] = SemanticEdge(
                    from_id=existing_edge.from_id,
                    rel=existing_edge.rel,
                    to_id=existing_edge.to_id,
                    edge_context=existing_edge.edge_context,
                    inferred=merged_inferred,
                )

    final_nodes: list[Any] = []
    for (ntype, nid), node in node_map.items():
        chunks = node_source_chunks.get((ntype, nid), [])
        if isinstance(node, dict):
            node["source_chunks"] = chunks
        else:
            try:
                object.__setattr__(node, "source_chunks", chunks)
            except Exception:
                pass

        final_nodes.append(node)

    notes_parts = []
    notes_parts.append(
        f"Merged {len(chunk_results)} chunks into "
        f"{len(final_nodes)} nodes and {len(edge_map)} edges."
    )

    _log.info(
        "merge_chunk_extractions: chunks=%d nodes=%d edges=%d",
        len(chunk_results),
        len(final_nodes),
        len(edge_map),
    )

    return ExtractionResult(
        nodes=final_nodes,
        edges=list(edge_map.values()),
        extraction_notes=" ".join(notes_parts),
    )
