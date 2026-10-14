"""
processor/trace/transformer.py — Deterministic trace dict → SemanticNode conversion.

Trace format:
    {
        "trace_id": "<uuid>",
        "ts": "<ISO8601>",
        "session_id": "<uuid>",
        "node": { "type": "Decision|Constraint|Insight|Signal", ...fields },
        "mentions": [...],
        "keywords": [...],
    }

The processor always overrides `id` (derived from type + trace_id) and
clears `source_doc` — whatever the author wrote in those fields is ignored.
"""

from __future__ import annotations

import logging
import re

from base.db.access import trace_personal_scope
from processor.schemas.extraction import (
    ConstraintNode,
    DecisionNode,
    FactNode,
    InsightNode,
    PreferenceNode,
    SemanticNode,
    SignalNode,
    TopicNode,
    EntityNode,
)

_log = logging.getLogger(__name__)
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_TYPE_MAP: dict[str, type] = {
    "Decision":   DecisionNode,
    "Constraint": ConstraintNode,
    "Insight":    InsightNode,
    "Signal":     SignalNode,
    "Fact":       FactNode,
    "Preference": PreferenceNode,
    "Topic":      TopicNode,
    "Entity":     EntityNode,
}


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def trace_to_node(trace: dict) -> SemanticNode:
    """
    Convert a trace dict (node-embedded format) to a SemanticNode.

    `id` is always derived from node type + trace_id.
    `source_doc` is always cleared — processor sets this from the trace reference.
    `created_at` falls back to the top-level `ts` if not set in the node.
    Unknown `type` falls back to InsightNode with a warning.
    """
    trace_id: str = trace.get("trace_id") or ""
    node_raw: dict = dict(trace.get("node") or {})
    author_employee_id = str(trace.get("_author_employee_id") or "").strip()
    source_scope = trace_personal_scope(author_employee_id or "unattributed")

    node_type: str = node_raw.get("type") or "Insight"
    model_cls = _TYPE_MAP.get(node_type)
    if model_cls is None:
        _log.warning(
            "trace_to_node: unknown node type %r in trace_id=%s, defaulting to Insight",
            node_type,
            trace_id,
        )
        model_cls = InsightNode
        node_type = "Insight"
        node_raw["type"] = "Insight"

    # Server-assigned fields — always override author values
    node_raw["id"] = f"{node_type.lower()}:{_slugify(trace_id) if trace_id else 'unknown'}"
    node_raw["source_doc"] = None
    node_raw["source_scope"] = source_scope
    node_raw.pop("source_doc_id", None)  # FactNode uses source_doc_id
    if trace.get("repo"):
        node_raw["repo"] = trace["repo"]

    # created_at: use node value if present, else fall back to top-level ts
    if not node_raw.get("created_at"):
        node_raw["created_at"] = trace.get("ts")

    # Strip embedding if somehow present
    node_raw.pop("embedding", None)

    try:
        return model_cls(**node_raw)
    except Exception as exc:
        _log.warning(
            "trace_to_node: validation failed for type=%s trace_id=%s error=%s — falling back to Insight",
            node_type,
            trace_id,
            exc,
        )
        fallback = {
            "type": "Insight",
            "id": f"insight:{_slugify(trace_id) if trace_id else 'unknown'}",
            "label": node_raw.get("label", trace_id),
            "body": node_raw.get("body") or node_raw.get("label") or trace_id,
            "source_scope": source_scope,
            "repo": trace.get("repo"),
            "origin": node_raw.get("origin"),
            "tag": node_raw.get("tag"),
            "created_at": node_raw.get("created_at"),
        }
        return InsightNode(**fallback)
