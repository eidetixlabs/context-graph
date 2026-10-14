"""
processor/storage/ingest.py — Write extraction results directly to Neo4j.

Reuses base/tools/ingest.py (ingest_entries) which performs batched UNWIND
MERGE operations so writes are ≤6+N round-trips rather than one per node.

Neo4j credentials are read from the environment exactly as the webserver does:
    NEO4J_URI      bolt://localhost:7687
    NEO4J_USER     neo4j
    NEO4J_PASSWORD <password>
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from base.neo4j_client import Neo4jClient
from base.tools.ingest import ingest_entries
from processor.config import config
from processor.schemas.extraction import ExtractionResult, SemanticEdge
from processor.state_machine import DocumentRecord

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neo4j client singleton (initialised once per process)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _neo4j_client() -> Neo4jClient:
    client = Neo4jClient()
    client.verify()
    return client


# ---------------------------------------------------------------------------
# Entry-format conversion (ExtractionResult → semantics.json entry list)
# ---------------------------------------------------------------------------


def _build_entries(result: ExtractionResult, doc: DocumentRecord) -> list[dict]:
    """Convert ExtractionResult to the semantics.json entry list format."""
    now = datetime.now(timezone.utc).isoformat()
    doc_ts = doc.source_created_at.isoformat() if doc.source_created_at is not None else now

    bucket = config.aws_s3_bucket
    source_s3_url = f"s3://{bucket}/{doc.s3_key}" if bucket else doc.s3_key

    nodes_by_id: dict[str, dict] = {}
    for node in result.nodes:
        if isinstance(node, dict):
            raw: dict[str, Any] = node
        else:
            raw = node.model_dump()
            # embedding is excluded from model_dump (exclude=True on field);
            # carry it over manually so ingest_entries can write it to Neo4j.
            if hasattr(node, "embedding") and node.embedding is not None:
                raw["embedding"] = node.embedding
        normalized = _normalize_node(raw)
        if normalized is None:
            continue
        nid = normalized.get("id")
        if nid:
            if normalized["type"] == "Entity":
                # Entity nodes are global directory records. Do not attach
                # private document provenance to the shared node.
                for field in ("source_scope", "source_doc_id", "source_s3_url", "source_chat"):
                    normalized.pop(field, None)
            else:
                normalized["source_scope"] = doc.source_scope
                normalized["source_doc_id"] = doc.id
                normalized["source_s3_url"] = source_s3_url
                if doc.repo is not None:
                    normalized["repo"] = doc.repo
                if doc.doc_type is not None:
                    normalized["doc_type"] = doc.doc_type
                normalized.setdefault("created_at", doc_ts)
            nodes_by_id[nid] = normalized

    entries: list[dict] = []
    seen_entry_ids: set[str] = set()

    for edge in result.edges:
        from_node = _normalize_node(nodes_by_id.get(edge.from_id), fallback_id=edge.from_id)
        to_node = _normalize_node(nodes_by_id.get(edge.to_id), fallback_id=edge.to_id)

        if from_node is None:
            _log.warning(
                "Edge references unknown from_id=%s (rel=%s to=%s) — skipping",
                edge.from_id, edge.rel, edge.to_id,
            )
            continue

        if to_node is None:
            # Edge points to a node that already exists in the graph (e.g. SUPERSEDES).
            # Create a minimal stub — ingest_entries will MATCH it by id.
            _log.debug(
                "Edge to_id=%s not in local nodes — creating stub for existing graph node",
                edge.to_id,
            )
            to_node = _stub_node(edge.to_id)

        entry_id = _entry_id(edge)
        if entry_id in seen_entry_ids:
            continue
        seen_entry_ids.add(entry_id)

        entries.append({
            "id": entry_id,
            "created_at": doc_ts,
            "from": from_node,
            "rel": edge.rel,
            "to": to_node,
            "properties": {
                "edge_context": edge.edge_context,
                "inferred": edge.inferred,
                "source_scope": doc.source_scope,
            },
        })

    return entries


def _entry_id(edge: SemanticEdge) -> str:
    safe = lambda s: s.replace(":", "-").replace("/", "-")
    return f"rel-{safe(edge.from_id)}-{edge.rel}-{safe(edge.to_id)}"


def _node_type_from_id(node_id: str) -> str:
    type_map = {
        "decision": "Decision",
        "constraint": "Constraint",
        "insight": "Insight",
        "signal": "Signal",
        "topic": "Topic",
        "entity": "Entity",
    }
    prefix = node_id.split(":", 1)[0].lower() if node_id else ""
    return type_map.get(prefix, "Decision")


def _normalize_node(raw: dict | None, fallback_id: str | None = None) -> dict | None:
    if raw is None:
        return None
    node = dict(raw)
    node_id = node.get("id") or fallback_id
    if not node_id:
        return None
    node["id"] = node_id
    node["type"] = node.get("type") or _node_type_from_id(node_id)
    node["label"] = node.get("label") or node_id
    return node


def _stub_node(node_id: str) -> dict:
    return _normalize_node({"id": node_id}, fallback_id=node_id) or {
        "id": node_id,
        "type": "Decision",
        "label": node_id,
    }


# ---------------------------------------------------------------------------
# Public write function
# ---------------------------------------------------------------------------


async def write_extraction_result(doc: DocumentRecord, result: ExtractionResult) -> dict:
    """
    Write an ExtractionResult directly to Neo4j using ingest_entries().

    Runs the synchronous Neo4j driver in a thread pool so the asyncio event
    loop is not blocked.

    Args:
        doc:    Source document (used to derive the project slug from org_id).
        result: ExtractionResult from Phase 3 of the extraction pipeline.

    Returns:
        Summary dict: {"nodes_upserted": N, "rels_upserted": M}
    """
    project = doc.org_id.removeprefix("org:")
    entries = _build_entries(result, doc)

    node_type_counts: dict[str, int] = {}
    for node in result.nodes:
        ntype = node.get("type") if isinstance(node, dict) else getattr(node, "type", None)
        if ntype:
            node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

    _log.info(
        "Neo4j write start: doc=%s project=%s entries=%d node_types=%s edges=%d",
        doc.id,
        project,
        len(entries),
        node_type_counts,
        len(result.edges),
    )

    if not entries:
        _log.warning("Neo4j write skipped: doc=%s no_entries", doc.id)
        return {"nodes_upserted": 0, "rels_upserted": 0}

    client = _neo4j_client()
    summary = await asyncio.to_thread(ingest_entries, client, project, entries)

    _log.info(
        "Neo4j write complete: doc=%s project=%s nodes_upserted=%d rels_upserted=%d",
        doc.id,
        project,
        summary.get("nodes_upserted", 0),
        summary.get("rels_upserted", 0),
    )
    return summary
