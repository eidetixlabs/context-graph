"""
tools/ingest.py — Ingest a semantics.json payload into Neo4j.

Neo4j is simpler than Kùzu here: no typed relationship tables.
EXPLAINS, CAUSES, MEMBER_OF, MENTIONS are all single rel types with a
`from_type` / `to_type` property for filtering — we don't need variant tables.

Idempotent: MERGE on (id) for nodes, MERGE on rel identity for edges.
"""

from __future__ import annotations

import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any

from base.neo4j_client import Neo4jClient

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node property maps — same fields as local ingest.py
# ---------------------------------------------------------------------------

_DEFAULT_NODE_FIELDS: dict[str, list[str]] = {
    "Decision": [
        "id", "label", "scope", "trigger", "rationale", "alternatives",
        "confidence", "uncertainty", "source_chat", "source_doc_id", "source_s3_url",
        "source_scope", "source_doc_version", "repo", "doc_type", "project", "tag", "origin", "created_at", "superseded_by", "embedding",
    ],
    "Constraint": [
        "id", "label", "scope", "description", "source", "hard",
        "symptom", "source_chat", "source_doc_id", "source_s3_url",
        "source_scope", "source_doc_version", "repo", "doc_type", "project", "tag", "origin", "created_at", "superseded_by", "embedding",
    ],
    "Insight": [
        "id", "label", "scope", "body", "failure_without",
        "source_chat", "source_doc_id", "source_doc_version", "source_s3_url", "source_scope", "repo", "doc_type", "project", "tag", "origin", "created_at", "superseded_by", "embedding",
    ],
    "Signal": [
        "id", "label", "scope", "observation", "appears_to_mean",
        "actually_means", "conditions", "evidence",
        "source_chat", "source_doc_id", "source_doc_version", "source_s3_url", "source_scope", "repo", "doc_type", "project", "tag", "origin", "created_at", "superseded_by", "embedding",
    ],
    "Topic": [
        "id", "label", "description", "summary", "project",
        "scope", "source", "source_chat", "source_doc_id", "source_doc_version", "source_s3_url", "source_scope", "repo", "doc_type", "tag", "origin", "created_at",
    ],
    "Entity": [
        "id", "label", "entity_type", "canonical_name",
        "aliases", "project", "source_chat", "tag", "origin", "created_at",
    ],
    "Preference": [
        "id", "category", "preference", "context", "confidence", "entity_id",
        "source_scope", "source_doc_id", "source_doc_version", "repo", "doc_type", "valid_from", "valid_until", "embedding", "project", "created_at",
    ],
    "Fact": [
        "id", "statement", "category", "confidence",
        "source_scope", "valid_from", "valid_until", "source_doc_id", "source_doc_version", "repo", "doc_type", "embedding", "project", "created_at",
    ],
    "ActionItem": [
        "id", "label", "status", "due_date", "source",
        "source_doc_id", "source_doc_version", "source_scope", "repo", "doc_type", "tag", "origin", "created_at", "embedding", "project",
    ],
}


def _load_node_fields() -> dict[str, list[str]]:
    path = Path(os.environ.get("ONTOLOGY_PATH", "/etc/eidetix/ontology.yaml"))
    if not path.exists():
        return _DEFAULT_NODE_FIELDS
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in ontology file {path}: {exc}") from exc
    if not isinstance(data, dict) or "node_types" not in data:
        raise RuntimeError(
            f"Ontology file {path} must be a YAML mapping with a 'node_types' key. Got: {type(data)}"
        )
    node_types = data["node_types"]
    if not isinstance(node_types, dict):
        raise RuntimeError(
            f"'node_types' in {path} must be a mapping, got: {type(node_types)}"
        )
    result: dict[str, list[str]] = {}
    for type_name, type_def in node_types.items():
        if not isinstance(type_def, dict) or "fields" not in type_def:
            raise RuntimeError(
                f"node_types.{type_name} in {path} must have a 'fields' list"
            )
        fields = type_def["fields"]
        if not isinstance(fields, list):
            raise RuntimeError(
                f"node_types.{type_name}.fields in {path} must be a list, got: {type(fields)}"
            )
        result[type_name] = [str(f) for f in fields]
    return result


NODE_FIELDS = _load_node_fields()

# Semantic node types (not Topic/Entity)
SEMANTIC_TYPES = {"Decision", "Constraint", "Insight", "Signal", "Preference", "Fact", "ActionItem"}

# Content fields used to build the embedding text for each semantic type.
# Only string fields that carry meaningful semantic content are included;
# metadata fields (ids, dates, tags) are intentionally excluded.
EMBED_FIELDS: dict[str, list[str]] = {
    "Decision":    ["label", "trigger", "rationale"],
    "Constraint":  ["label", "description", "symptom"],
    "Insight":     ["label", "body", "failure_without"],
    "Signal":      ["label", "observation", "appears_to_mean", "actually_means"],
    "Preference":  ["category", "preference", "context"],
    "Fact":        ["statement", "category"],
    "ActionItem":  ["label", "source"],
}


def build_embed_text(raw: dict) -> str:
    """Concatenate non-empty content fields of a semantic node into a single string."""
    node_type = raw.get("type", "")
    fields = EMBED_FIELDS.get(node_type, [])
    parts = []
    for f in fields:
        val = raw.get(f)
        if val and isinstance(val, str):
            parts.append(val.strip())
    return " | ".join(parts)

# Relationship types that are valid in the semantics.json schema
VALID_RELS = {
    "SHAPED_BY", "SUPERSEDES", "ABANDONED", "LIFTED", "CONTRADICTS",
    "EXPLAINS", "CAUSES", "MEMBER_OF", "SUBTOPIC_OF", "MENTIONS",
    "EXTRACTED_FROM", "AUTHORED_BY", "HAS_PREFERENCE", "ABOUT", "APPLIES_TO",
    "AFFECTS", "OWNS", "BLOCKS", "INVOLVES",
}

_ID_PREFIX_TO_TYPE: dict[str, str] = {
    "decision": "Decision",
    "constraint": "Constraint",
    "insight": "Insight",
    "signal": "Signal",
    "topic": "Topic",
    "entity": "Entity",
    "preference": "Preference",
    "fact": "Fact",
    "actionitem": "ActionItem",
    "document": "Document",
}


def _node_type_from_id(node_id: str) -> str | None:
    prefix = node_id.split(":", 1)[0].lower() if node_id else ""
    return _ID_PREFIX_TO_TYPE.get(prefix)


def _resolve_endpoint(raw: dict | None) -> dict | None:
    """Return endpoint with ``type`` set, inferring from id prefix when absent."""
    if not raw:
        return None
    node_id = raw.get("id")
    if not node_id:
        return None
    node_type = raw.get("type") or raw.get("category") or _node_type_from_id(node_id)
    if not node_type:
        return None
    if raw.get("type") == node_type:
        return raw
    resolved = dict(raw)
    resolved["type"] = node_type
    return resolved


def _coerce_node(raw: dict, project: str) -> dict[str, Any]:
    node_type = raw["type"]
    fields = NODE_FIELDS.get(node_type, [])
    props: dict[str, Any] = {}
    for f in fields:
        if f == "id":
            continue  # id is the MERGE key, not in SET payload
        val = raw.get(f)
        if val is None:
            continue
        if f == "alternatives" and isinstance(val, list):
            val = json.dumps(val)
        if f == "hard":
            val = bool(val)
        if f in ("valid_from", "valid_until") and isinstance(val, str) and val:
            try:
                from base.temporal import parse_neo4j_datetime  # noqa: PLC0415
                val = parse_neo4j_datetime(val)
            except Exception:
                _log.warning(
                    "_coerce_node: invalid %s value %r for node %s — field skipped",
                    f, val, raw.get("id"),
                )
                val = None
        if val is None:
            continue
        props[f] = val
    # Provenance is mandatory for document-derived semantic nodes even when an
    # environment supplies an older custom ontology that does not list it yet.
    if node_type != "Entity":
        for field in ("source_doc_id", "source_doc_version", "repo", "doc_type"):
            if raw.get(field) is not None:
                props[field] = raw[field]
    props["project"] = project
    props["_type"] = node_type  # stored as a property for filtering
    if node_type == "Entity":
        # Entities are global directory records. Provenance belongs on
        # relationships, never on the shared node itself.
        props.pop("source_scope", None)
        props.pop("source_doc_id", None)
        props.pop("source_s3_url", None)
        props.pop("source_chat", None)
    return props


def _upsert_node(client: Neo4jClient, raw: dict, project: str) -> None:
    node_type = raw["type"]
    props = _coerce_node(raw, project)
    node_id = raw["id"]

    # SET n += $props merges all map properties onto the node — no $props.field syntax needed.
    remove_scope = "REMOVE n.source_scope" if node_type == "Entity" else ""
    cypher = f"""
        MERGE (n:{node_type} {{id: $id}})
        SET n += $props
        {remove_scope}
    """
    client.run_write(cypher, {"id": node_id, "props": props})


def _upsert_rel(client: Neo4jClient, entry: dict) -> None:
    rel_type = entry["rel"]
    if rel_type not in VALID_RELS:
        print(f"  WARNING: unknown rel type '{rel_type}', skipping")
        return

    from_raw = _resolve_endpoint(entry.get("from"))
    to_raw   = _resolve_endpoint(entry.get("to"))
    if from_raw is None or to_raw is None:
        _log.warning(
            "_upsert_rel: skipping entry rel=%s — unresolved endpoint(s) from=%r to=%r",
            rel_type, entry.get("from"), entry.get("to"),
        )
        return
    from_type = from_raw["type"]
    to_type   = to_raw["type"]

    props = entry.get("properties", {})

    params = {
        "from_id":      from_raw["id"],
        "to_id":        to_raw["id"],
        "edge_context": props.get("edge_context", ""),
        "inferred":     bool(props.get("inferred", False)),
        "created_at":   entry.get("created_at", ""),
        "from_type":    from_type,
        "to_type":      to_type,
        "source_scope": props.get("source_scope"),
    }
    if props.get("source_scope") is not None:
        cypher = f"""
            MATCH (a:{from_type} {{id: $from_id}})
            MATCH (b:{to_type}   {{id: $to_id}})
            MERGE (a)-[r:{rel_type} {{source_scope: $source_scope}}]->(b)
            ON CREATE SET r.created_at = $created_at
            SET r.edge_context = $edge_context,
                r.inferred     = $inferred,
                r.from_type    = $from_type,
                r.to_type      = $to_type
        """
    else:
        cypher = f"""
            MATCH (a:{from_type} {{id: $from_id}})
            MATCH (b:{to_type}   {{id: $to_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            ON CREATE SET r.created_at = $created_at
            SET r.edge_context = $edge_context,
                r.inferred     = $inferred,
                r.from_type    = $from_type,
                r.to_type      = $to_type
        """
    client.run_write(cypher, params)


def _build_node_batch(
    nodes: dict[str, dict], project: str, ingested_by: str | None = None
) -> dict[str, list[dict]]:
    """Group nodes by type for UNWIND batch upserts.

    Entity nodes get a separate ``create_project`` key so the caller can use
    ON CREATE SET for project — preventing cross-project re-ingests from
    clobbering the project field on shared/global Entity nodes.
    """
    by_type: dict[str, list[dict]] = {}
    for raw in nodes.values():
        node_type = raw.get("type", "")
        if node_type not in NODE_FIELDS:
            continue
        props = _coerce_node(raw, project)
        if ingested_by and node_type != "Entity":
            props["ingested_by"] = ingested_by
        node_id = raw["id"]
        if node_type == "Entity":
            # Strip project out of the mutable props dict so SET n += row.props
            # does not overwrite it; project is set only ON CREATE below.
            create_project = props.pop("project", project)
            create_ts = props.pop("created_at", None)
            batch_row = {
                "id": node_id,
                "props": props,
                "create_project": create_project,
                "create_ts": create_ts,
            }
        else:
            create_ts = props.pop("created_at", None)
            create_ingested_by = props.pop("ingested_by", None)
            batch_row = {
                "id": node_id,
                "props": props,
                "create_ts": create_ts,
                "create_ingested_by": create_ingested_by,
            }
        by_type.setdefault(node_type, []).append(batch_row)
    return by_type


def _build_rel_batches(entries: list[dict]) -> dict[str, list[dict]]:
    """Group rel entries by (rel_type, from_type, to_type) for UNWIND batch upserts."""
    by_key: dict[str, list[dict]] = {}
    for entry in entries:
        rel_type = entry.get("rel", "")
        if rel_type not in VALID_RELS:
            continue
        from_raw = _resolve_endpoint(entry.get("from"))
        to_raw   = _resolve_endpoint(entry.get("to"))
        if from_raw is None or to_raw is None:
            _log.warning(
                "_build_rel_batches: skipping rel=%s — unresolved endpoint(s) from=%r to=%r",
                rel_type, entry.get("from"), entry.get("to"),
            )
            continue
        properties = entry.get("properties", {})
        provenance = "versioned" if properties.get("source_doc_id") is not None else "legacy"
        key = f"{rel_type}__{from_raw['type']}__{to_raw['type']}__{provenance}"
        by_key.setdefault(key, []).append({
            "from_id":      from_raw["id"],
            "to_id":        to_raw["id"],
            "edge_context": entry.get("properties", {}).get("edge_context", ""),
            "inferred":     bool(entry.get("properties", {}).get("inferred", False)),
            "source_scope": properties.get("source_scope"),
            "source_doc_id": properties.get("source_doc_id"),
            "source_doc_version": properties.get("source_doc_version"),
            "created_at":   entry.get("created_at", ""),
            "from_type":    from_raw["type"],
            "to_type":      to_raw["type"],
        })
    return by_key


def ingest_entries(
    client: Neo4jClient,
    project: str,
    entries: list[dict],
    ingested_by: str | None = None,
) -> dict:
    """
    Ingest a semantics.json entry list into Neo4j.
    Uses UNWIND to batch nodes by type (≤6 round-trips) and rels by
    (rel_type, from_type, to_type) (≤N_distinct_combos round-trips).
    Returns a summary dict with node/edge counts.

    ``ingested_by`` is stamped when node is first created, as the canonical
    name of authenticated employee who triggered ingest (e.g. "Vaibhav Sharma").
    """
    # Deduplicate nodes across all entries (last write wins for same id).
    # Skip stubs (_stub=True) — these are references to pre-existing nodes
    # fetched from Neo4j; we must not overwrite them with empty properties.
    nodes: dict[str, dict] = {}
    for entry in entries:
        for side in ("from", "to"):
            raw = _resolve_endpoint(entry.get(side))
            if raw is None:
                continue
            if raw.get("_stub"):
                continue
            nodes[raw["id"]] = raw

    # Batch upsert nodes grouped by label (one UNWIND per label = 6 round-trips max)
    batches = _build_node_batch(nodes, project, ingested_by=ingested_by)
    for node_type, rows in batches.items():
        if node_type == "Entity":
            # Entity nodes are global/shared across projects.  Only stamp
            # `project` on first creation; subsequent re-ingests from other
            # projects must not overwrite the original owner.
            # `created_at` is likewise protected: set only on first creation.
            cypher = f"""
                UNWIND $rows AS row
                MERGE (n:{node_type} {{id: row.id}})
                ON CREATE SET n.project = row.create_project,
                              n.created_at = row.create_ts
                SET n += row.props
                REMOVE n.source_scope, n.ingested_by
            """
        else:
            cypher = f"""
                UNWIND $rows AS row
                MERGE (n:{node_type} {{id: row.id}})
                ON CREATE SET n.created_at = row.create_ts,
                              n.ingested_by = row.create_ingested_by
                SET n += row.props
            """
        client.run_write(cypher, {"rows": rows})

    # Batch upsert rels grouped by (rel_type, from_label, to_label) —
    # one UNWIND per combo instead of one query per edge.
    rel_batches = _build_rel_batches(entries)
    rels_upserted = 0
    for key, rows in rel_batches.items():
        rel_type, from_type, to_type, provenance = key.split("__")
        scoped_rows = [row for row in rows if row["source_scope"] is not None]
        unscoped_rows = [row for row in rows if row["source_scope"] is None]
        if unscoped_rows:
            merge_props = (
                "{source_doc_id: row.source_doc_id, "
                "source_doc_version: row.source_doc_version}"
                if provenance == "versioned"
                else ""
            )
            cypher = f"""
                UNWIND $rows AS row
                MATCH (a:{from_type} {{id: row.from_id}})
                MATCH (b:{to_type}   {{id: row.to_id}})
                MERGE (a)-[r:{rel_type} {merge_props}]->(b)
                ON CREATE SET r.created_at = row.created_at
                SET r.edge_context = row.edge_context,
                    r.inferred     = row.inferred,
                    r.from_type    = row.from_type,
                    r.to_type      = row.to_type,
                    r.source_doc_id = row.source_doc_id,
                    r.source_doc_version = row.source_doc_version
            """
            client.run_write(cypher, {"rows": unscoped_rows})
        if scoped_rows:
            merge_props = (
                "{source_scope: row.source_scope, "
                "source_doc_id: row.source_doc_id, "
                "source_doc_version: row.source_doc_version}"
                if provenance == "versioned"
                else "{source_scope: row.source_scope}"
            )
            cypher = f"""
                UNWIND $rows AS row
                MATCH (a:{from_type} {{id: row.from_id}})
                MATCH (b:{to_type}   {{id: row.to_id}})
                MERGE (a)-[r:{rel_type} {merge_props}]->(b)
                ON CREATE SET r.created_at = row.created_at
                SET r.edge_context = row.edge_context,
                    r.inferred     = row.inferred,
                    r.from_type    = row.from_type,
                    r.to_type      = row.to_type,
                    r.source_doc_id = row.source_doc_id,
                    r.source_doc_version = row.source_doc_version
            """
            client.run_write(cypher, {"rows": scoped_rows})
        rels_upserted += len(rows)

    return {"nodes_upserted": len(nodes), "rels_upserted": rels_upserted}
