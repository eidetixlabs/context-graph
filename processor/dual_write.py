"""
processor/dual_write.py — Coordinated write to PgVector (document_chunks) and Neo4j.

Extraction output is first persisted as a canonical S3 payload and registered
in document_index_jobs. PgVector and Neo4j then advance independently. A retry
loads that payload and runs only incomplete sinks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from base.db.client import db_pool
from base.neo4j_client import Neo4jClient
from base.tools.ingest import NODE_FIELDS, _build_node_batch, ingest_entries
from processor.schemas.extraction import ExtractionResult
from processor.state_machine import DocumentRecord, StaleDocumentWork
from processor.storage.s3 import read_json_object, write_json_object

_log = logging.getLogger(__name__)


@dataclass
class IndexJob:
    doc_id: str
    document_version: int
    payload_hash: str
    payload_s3_key: str
    vector_state: str
    graph_state: str

    @classmethod
    def from_record(cls, row: Any) -> IndexJob:
        return cls(
            doc_id=str(row["doc_id"]),
            document_version=int(row["document_version"]),
            payload_hash=str(row["payload_hash"]),
            payload_s3_key=str(row["payload_s3_key"]),
            vector_state=str(row["vector_state"]),
            graph_state=str(row["graph_state"]),
        )


class IndexingIncompleteError(RuntimeError):
    """Raised until both durable indexing sinks have completed."""

    def __init__(self, job: IndexJob, errors: dict[str, str] | None = None):
        self.job = job
        self.errors = errors or {}
        details = ", ".join(f"{sink}: {error}" for sink, error in self.errors.items())
        super().__init__(
            f"indexing incomplete for {job.doc_id}@v{job.document_version}"
            + (f" ({details})" if details else "")
        )


class StaleIndexWork(StaleDocumentWork):
    """Raised when an index job no longer targets the canonical document version."""


class IndexJobAlreadyClaimed(StaleIndexWork):
    """Raised when another replica holds the session advisory lock for a job."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _namespace_gmail_semantic_ids(
    extraction_result: ExtractionResult,
    doc_id: str,
) -> None:
    """Make sensitive Gmail semantic nodes unique to their source message."""
    suffix = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12]
    marker = f":gmail:{suffix}"
    remapped_ids: dict[str, str] = {}

    for node in extraction_result.nodes:
        raw_type = node.get("type") if isinstance(node, dict) else getattr(node, "type", None)
        old_id = node.get("id") if isinstance(node, dict) else getattr(node, "id", None)
        if raw_type in ("Entity", "Document") or not isinstance(old_id, str):
            continue
        new_id = old_id if old_id.endswith(marker) else f"{old_id}{marker}"
        remapped_ids[old_id] = new_id
        if isinstance(node, dict):
            node["id"] = new_id
        else:
            node.id = new_id

    for edge in extraction_result.edges:
        if edge.from_id in remapped_ids:
            edge.from_id = remapped_ids[edge.from_id]
        if edge.to_id in remapped_ids:
            edge.to_id = remapped_ids[edge.to_id]


def _upsert_doc_node(client: Neo4jClient, doc: DocumentRecord, project: str) -> None:
    """MERGE a Document node into Neo4j with full DocumentRecord fields."""
    parent_id = f"doc:{doc.source_provider}:{doc.parent_ref}" if doc.parent_ref else None
    client.run_write(
        """
        MERGE (d:Document {id: $id})
        SET d.org_id           = $org_id,
            d.project          = $project,
            d.repo             = $repo,
            d.doc_type         = $doc_type,
            d.source_provider  = $source_provider,
            d.source_scope     = $source_scope,
            d.source_ref       = $source_ref,
            d.title            = $title,
            d.s3_key           = $s3_key,
            d.processing_state = $processing_state,
            d.current_version  = $current_version
        WITH d
        OPTIONAL MATCH (p:Document {id: $parent_id})
        FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END |
            MERGE (d)-[r:SUBTOPIC_OF]->(p)
            SET r.source_scope = $source_scope)
        """,
        {
            "id":               doc.id,
            "org_id":           doc.org_id,
            "project":          project,
            "repo":             doc.repo,
            "doc_type":         doc.doc_type,
            "source_provider":  doc.source_provider,
            "source_scope":     doc.source_scope,
            "source_ref":       doc.source_ref,
            "title":            doc.title or "",
            "s3_key":           doc.s3_key,
            "processing_state": doc.processing_state,
            "current_version":  doc.current_version,
            "parent_id":        parent_id,
        },
    )


async def _write_chunks_pg_conn(
    conn: Any,
    doc_id: str,
    chunk_result: list[dict],
) -> int:
    """Replace one document's retrieval chunks inside the caller transaction."""
    rows = [
        (
            doc_id,
            c["chunk_index"],
            c["char_offset"],
            c["char_length"],
            "[" + ",".join(str(v) for v in c["embedding"]) + "]",
        )
        for c in chunk_result
    ]
    await conn.execute(
        "DELETE FROM document_chunks WHERE doc_id = $1 AND chunk_index >= $2",
        doc_id,
        len(rows),
    )
    if rows:
        await conn.executemany(
            """
            INSERT INTO document_chunks
                (doc_id, chunk_index, char_offset, char_length, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (doc_id, chunk_index) DO UPDATE
                SET char_offset = EXCLUDED.char_offset,
                    char_length = EXCLUDED.char_length,
                    embedding   = EXCLUDED.embedding
            """,
            rows,
        )
    return len(rows)


def _coerce_temporal_fields(raw: dict) -> dict:
    """Convert valid_from / valid_until ISO strings to neo4j.time.DateTime objects.

    Only applied to Preference and Fact nodes. Other node types pass through unchanged.
    Returns a new dict (does not mutate the input).
    """
    node_type = raw.get("type", "")
    if node_type not in ("Preference", "Fact"):
        return raw
    from base.temporal import parse_neo4j_datetime  # noqa: PLC0415
    result = dict(raw)
    for field in ("valid_from", "valid_until"):
        val = result.get(field)
        if val and isinstance(val, str):
            try:
                result[field] = parse_neo4j_datetime(val)
            except Exception:
                _log.warning(
                    "_coerce_temporal_fields: invalid %s value %r for node %s — field skipped",
                    field, val, result.get("id"),
                )
                result[field] = None
    return result


def _normalise_node_raw(node) -> dict | None:
    """Coerce a node (dict or Pydantic model) to a raw dict. Returns None if invalid."""
    if isinstance(node, dict):
        raw = node
    elif hasattr(node, "model_dump"):
        raw = node.model_dump()
    else:
        raw = dict(node)

    nid = raw.get("id", "")
    ntype = raw.get("type", "")
    if not nid or not ntype:
        return None
    return raw


def _raw_node_for_diagnostics(node) -> dict:
    """Best-effort raw node conversion used only for skip diagnostics."""
    try:
        if isinstance(node, dict):
            return node
        if hasattr(node, "model_dump"):
            dumped = node.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        coerced = dict(node)
        return coerced if isinstance(coerced, dict) else {}
    except Exception:
        return {}


def _upsert_semantic_nodes(
    client: Neo4jClient,
    extraction_result: ExtractionResult,
    project: str,
    source_scope: str | None = None,
    preserve_existing_scope: bool = False,
    source_doc_id: str | None = None,
    source_doc_version: int | None = None,
    repo: str | None = None,
    doc_type: str | None = None,
) -> int:
    """
    Directly upsert final nodes from ExtractionResult into Neo4j.

    Uses the same _build_node_batch / UNWIND pattern as ingest_entries but
    without requiring a dummy edge entry per node. Returns count of nodes upserted.
    """
    nodes: dict[str, dict] = {}
    skipped_missing_id = 0
    skipped_missing_type = 0
    skipped_uncoercible = 0
    for node in extraction_result.nodes:
        raw = _normalise_node_raw(node)
        if raw is None:
            diagnostic_raw = _raw_node_for_diagnostics(node)
            if not diagnostic_raw:
                skipped_uncoercible += 1
            elif not diagnostic_raw.get("id"):
                skipped_missing_id += 1
            elif not diagnostic_raw.get("type"):
                skipped_missing_type += 1
            else:
                skipped_uncoercible += 1
            continue
        nodes[raw["id"]] = _coerce_temporal_fields(raw)

    if source_doc_id is not None:
        for raw in nodes.values():
            if raw.get("type") == "Entity":
                continue
            raw["source_doc_id"] = source_doc_id
            raw["source_doc_version"] = source_doc_version
            if repo is not None:
                raw["repo"] = repo
            if doc_type is not None:
                raw["doc_type"] = doc_type

    existing_node_ids: set[str] = set()
    if source_scope and preserve_existing_scope and nodes:
        try:
            rows = client.run(
                "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id",
                {"ids": list(nodes.keys())},
            )
            existing_node_ids = {str(r["id"]) for r in rows if r.get("id")}
        except Exception as exc:
            _log.warning("Neo4j existing-node lookup failed while preserving scope: %s", exc)
            existing_node_ids = set()

    if source_scope:
        for node_id, raw in nodes.items():
            # Entity nodes are global directory records. Their document-derived
            # relationships carry RBAC provenance; applying a document scope to
            # the shared entity would make it inaccessible from other documents.
            if raw.get("type") == "Entity":
                raw.pop("source_scope", None)
                continue
            if preserve_existing_scope and node_id in existing_node_ids:
                raw.pop("source_scope", None)
            else:
                # Enforce document-derived RBAC scope on newly created or
                # provider-owned semantic nodes.
                raw["source_scope"] = source_scope

    unknown_type_counts: dict[str, int] = {}
    for raw in nodes.values():
        node_type = str(raw.get("type", ""))
        if node_type not in NODE_FIELDS:
            unknown_type_counts[node_type] = unknown_type_counts.get(node_type, 0) + 1

    skipped_unknown_type = sum(unknown_type_counts.values())
    total_generated = len(extraction_result.nodes)
    total_skipped = skipped_missing_id + skipped_missing_type + skipped_uncoercible + skipped_unknown_type

    if total_skipped > 0:
        _log.critical(
            "Neo4j node ingest skip: generated=%d accepted=%d skipped_total=%d "
            "missing_id=%d missing_type=%d unknown_type=%d uncoercible=%d unknown_types=%s",
            total_generated,
            len(nodes) - skipped_unknown_type,
            total_skipped,
            skipped_missing_id,
            skipped_missing_type,
            skipped_unknown_type,
            skipped_uncoercible,
            unknown_type_counts,
        )

    if not nodes:
        return 0

    batches = _build_node_batch(nodes, project)
    for node_type, rows in batches.items():
        if node_type == "Entity":
            cypher = f"""
                UNWIND $rows AS row
                MERGE (n:{node_type} {{id: row.id}})
                ON CREATE SET n.project = row.create_project
                SET n += row.props
                REMOVE n.source_scope
            """
        else:
            cypher = f"""
                UNWIND $rows AS row
                MERGE (n:{node_type} {{id: row.id}})
                SET n += row.props
            """
        client.run_write(cypher, {"rows": rows})

    return sum(len(rows) for rows in batches.values())


def _fetch_node_stub(client: Neo4jClient, node_id: str) -> dict | None:
    """
    Look up a node that exists in Neo4j but was not part of the current
    extraction result.  Returns a minimal stub sufficient for Cypher MATCH,
    or None if the node is not found.

    Stubs are marked with ``_stub: True`` so downstream code (ingest_entries)
    can skip them during node batching — we must not overwrite existing node
    properties with an empty stub.
    """
    rows = client.run(
        "MATCH (n {id: $id}) RETURN labels(n)[0] AS label LIMIT 1",
        {"id": node_id},
    )
    if not rows:
        return None
    label = rows[0].get("label")
    if not label:
        return None
    return {"type": label, "id": node_id, "_stub": True}


def _resolve_node(
    extraction_nodes: list,
    node_id: str,
    client: Neo4jClient,
) -> dict | None:
    """
    Resolve a node ID to a raw dict for edge construction.

    Lookup order:
      1. In-memory extraction nodes.
      2. Live Neo4j lookup for existing graph context nodes.
    """
    node = _find_node_raw(extraction_nodes, node_id)
    if node is not None:
        return node
    return _fetch_node_stub(client, node_id)


def _existing_extraction_node_ids(
    client: Neo4jClient,
    extraction_result: ExtractionResult,
) -> set[str]:
    """Return extraction-node IDs that existed before this document write."""
    node_ids: list[str] = []
    for node in extraction_result.nodes:
        raw = _normalise_node_raw(node)
        if raw is not None:
            node_ids.append(raw["id"])
    if not node_ids:
        return set()
    rows = client.run(
        "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id",
        {"ids": node_ids},
    )
    return {str(row["id"]) for row in rows if row.get("id")}


def _build_edge_entries(
    extraction_result: ExtractionResult,
    doc_id: str,
    client: Neo4jClient,
    resolution_map=None,
    source_scope: str | None = None,
    existing_node_ids: set[str] | None = None,
    source_doc_version: int | None = None,
) -> list[dict]:
    """
    Build ingest_entries-format edge list from ExtractionResult.

    Includes:
    - All LLM edges (SHAPED_BY, EXPLAINS, MENTIONS, etc.)
    - EXTRACTED_FROM edges from each final extraction node to the Document node

    Edges whose endpoints are not in the current extraction result are resolved
    against the live graph so that cross-document relationships are preserved.

    No self-reference entries — nodes are already upserted by _upsert_semantic_nodes.
    """
    doc_node_raw = {"type": "Document", "id": doc_id}
    entries: list[dict] = []

    for edge in extraction_result.edges:
        from_node = _resolve_node(extraction_result.nodes, edge.from_id, client)
        to_node   = _resolve_node(extraction_result.nodes, edge.to_id, client)
        if from_node is None or to_node is None:
            _log.warning(
                "Dropping edge %s -[%s]-> %s: node(s) not found in extraction "
                "result or graph (from_found=%s, to_found=%s)",
                edge.from_id, edge.rel, edge.to_id,
                from_node is not None, to_node is not None,
            )
            continue
        if existing_node_ids:
            if from_node["id"] in existing_node_ids:
                from_node = {"type": from_node["type"], "id": from_node["id"], "_stub": True}
            if to_node["id"] in existing_node_ids:
                to_node = {"type": to_node["type"], "id": to_node["id"], "_stub": True}
        entries.append({
            "from": from_node,
            "to":   to_node,
            "rel":  edge.rel,
            "properties": {
                "edge_context": edge.edge_context,
                "inferred":     edge.inferred,
                "source_scope": source_scope,
                "source_doc_id": doc_id,
                "source_doc_version": source_doc_version,
            },
        })

    for node in extraction_result.nodes:
        raw = _normalise_node_raw(node)
        if raw is None:
            continue
        if existing_node_ids and raw["id"] in existing_node_ids:
            raw = {"type": raw["type"], "id": raw["id"], "_stub": True}
        entries.append({
            "from": raw,
            "to":   doc_node_raw,
            "rel":  "EXTRACTED_FROM",
            "properties": {
                "edge_context": "extracted from source document",
                "inferred": False,
                "source_scope": source_scope,
                "source_doc_id": doc_id,
                "source_doc_version": source_doc_version,
            },
        })

    return entries


def _find_node_raw(nodes: list, node_id: str) -> dict | None:
    for node in nodes:
        if isinstance(node, dict):
            if node.get("id") == node_id:
                raw = node
            else:
                continue
        else:
            if getattr(node, "id", None) == node_id:
                raw = node.model_dump() if hasattr(node, "model_dump") else dict(node)
            else:
                continue
        if not raw.get("type"):
            return None
        return raw
    return None


# ---------------------------------------------------------------------------
# Durable indexing jobs
# ---------------------------------------------------------------------------


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _serialise_extraction_result(result: ExtractionResult) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node in result.nodes:
        raw = node if isinstance(node, dict) else node.model_dump(mode="json")
        raw = dict(raw)
        embedding = (
            raw.get("embedding")
            if isinstance(node, dict)
            else getattr(node, "embedding", None)
        )
        if embedding is not None:
            raw["embedding"] = embedding
        nodes.append(raw)
    return {
        "nodes": nodes,
        "edges": [
            edge if isinstance(edge, dict) else edge.model_dump(mode="json")
            for edge in result.edges
        ],
        "extraction_notes": result.extraction_notes,
    }


def _build_index_payload(
    project: str,
    chunk_result: list[dict],
    extraction_result: ExtractionResult,
) -> dict[str, Any]:
    return {
        "schema_version": "document_index_payload.v1",
        "project": project,
        "chunks": chunk_result,
        "extraction_result": _serialise_extraction_result(extraction_result),
    }


async def _load_job(doc_id: str, document_version: int) -> IndexJob | None:
    pool = db_pool.get()
    row = await pool.fetchrow(
        """
        SELECT doc_id, document_version, payload_hash, payload_s3_key,
               vector_state, graph_state
        FROM document_index_jobs
        WHERE doc_id = $1 AND document_version = $2
        ORDER BY created_at
        LIMIT 1
        """,
        doc_id,
        document_version,
    )
    return IndexJob.from_record(row) if row is not None else None


@asynccontextmanager
async def _index_job_lease(job: IndexJob):
    """Hold a Postgres session advisory lock for one complete sink operation."""
    pool = db_pool.get()
    lock_name = f"document-index:{job.doc_id}:v{job.document_version}"
    async with pool.acquire() as conn:
        acquired = await conn.fetchval(
            "SELECT pg_try_advisory_lock(hashtextextended($1, 0))",
            lock_name,
        )
        if not acquired:
            raise IndexJobAlreadyClaimed(
                f"index job already claimed: {job.doc_id}@v{job.document_version}"
            )
        try:
            yield conn
        finally:
            await conn.fetchval(
                "SELECT pg_advisory_unlock(hashtextextended($1, 0))",
                lock_name,
            )


async def _lock_current_document(conn: Any, job: IndexJob) -> None:
    row = await conn.fetchrow(
        """
        SELECT current_version, processing_state
        FROM documents
        WHERE id = $1
        FOR UPDATE
        """,
        job.doc_id,
    )
    if (
        row is None
        or int(row["current_version"]) != job.document_version
        or str(row["processing_state"]) != "indexing"
    ):
        raise StaleIndexWork(
            f"stale index work cancelled: {job.doc_id}@v{job.document_version}"
        )


async def _ensure_index_job(
    doc: DocumentRecord,
    project: str,
    chunk_result: list[dict],
    extraction_result: ExtractionResult,
) -> tuple[IndexJob, dict[str, Any]]:
    """Persist one canonical extraction payload and transition to indexing."""
    existing = await _load_job(doc.id, doc.current_version)
    if existing is not None:
        try:
            payload = await _read_index_payload(existing)
        except (TypeError, ValueError):
            # Migration 018 was initially wired to enqueue the raw profile JSON.
            # It is not resumable index data; remove only that known legacy row
            # and replace it with the canonical extraction payload below.
            if existing.payload_s3_key != doc.s3_key:
                raise
            pool = db_pool.get()
            await pool.execute(
                """
                DELETE FROM document_index_jobs
                WHERE doc_id = $1 AND document_version = $2 AND payload_hash = $3
                """,
                existing.doc_id,
                existing.document_version,
                existing.payload_hash,
            )
        else:
            from processor.state_machine import transition_indexing  # noqa: PLC0415

            await transition_indexing(doc.id, doc.current_version)
            return existing, payload

    payload = _build_index_payload(project, chunk_result, extraction_result)
    payload_bytes = _canonical_json_bytes(payload)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    payload_s3_key = (
        f"{doc.org_id}/{doc.source_provider}/index/"
        f"{doc.source_ref}/v{doc.current_version}/{payload_hash}.json"
    )

    pool = db_pool.get()
    async with pool.acquire() as conn, conn.transaction():
        document_row = await conn.fetchrow(
            """
            SELECT current_version, processing_state
            FROM documents
            WHERE id = $1
            FOR UPDATE
            """,
            doc.id,
        )
        if (
            document_row is None
            or int(document_row["current_version"]) != doc.current_version
            or str(document_row["processing_state"]) != "extracting"
        ):
            raise StaleIndexWork(
                f"stale extraction payload cancelled: {doc.id}@v{doc.current_version}"
            )
        row = await conn.fetchrow(
            """
            SELECT doc_id, document_version, payload_hash, payload_s3_key,
                   vector_state, graph_state
            FROM document_index_jobs
            WHERE doc_id = $1 AND document_version = $2
            ORDER BY created_at
            LIMIT 1
            """,
            doc.id,
            doc.current_version,
        )
        if row is None:
            await write_json_object(payload_s3_key, payload_bytes)
            await conn.execute(
                """
                INSERT INTO document_index_jobs (
                    doc_id, document_version, payload_hash, payload_s3_key
                )
                VALUES ($1, $2, $3, $4)
                """,
                doc.id,
                doc.current_version,
                payload_hash,
                payload_s3_key,
            )
            job = IndexJob(
                doc_id=doc.id,
                document_version=doc.current_version,
                payload_hash=payload_hash,
                payload_s3_key=payload_s3_key,
                vector_state="pending",
                graph_state="pending",
            )
        else:
            job = IndexJob.from_record(row)
            payload = await _read_index_payload(job)
        status = await conn.execute(
            """
            UPDATE documents
            SET processing_state = 'indexing', updated_at = now()
            WHERE id = $1
              AND current_version = $2
              AND processing_state = 'extracting'
            """,
            doc.id,
            doc.current_version,
        )
        if status != "UPDATE 1":
            raise StaleIndexWork(
                f"stale indexing transition cancelled: {doc.id}@v{doc.current_version}"
            )
    return job, payload


async def _read_index_payload(job: IndexJob) -> dict[str, Any]:
    payload = await read_json_object(job.payload_s3_key)
    actual_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual_hash != job.payload_hash:
        raise ValueError(
            f"index payload hash mismatch for {job.doc_id}@v{job.document_version}"
        )
    if payload.get("schema_version") != "document_index_payload.v1":
        raise ValueError(f"unsupported index payload for {job.doc_id}")
    return payload


async def _set_sink_state(
    conn: Any,
    job: IndexJob,
    sink: str,
    state: str,
    error: str | None = None,
) -> None:
    if sink not in {"vector", "graph"}:
        raise ValueError(f"unknown index sink: {sink}")
    status = await conn.execute(
        f"""
        UPDATE document_index_jobs
        SET {sink}_state = $1,
            {sink}_error = $2,
            updated_at = now()
        WHERE doc_id = $3 AND document_version = $4 AND payload_hash = $5
          AND EXISTS (
              SELECT 1
              FROM documents
              WHERE id = $3
                AND current_version = $4
                AND processing_state = 'indexing'
          )
        """,
        state,
        error[:2048] if error else None,
        job.doc_id,
        job.document_version,
        job.payload_hash,
    )
    if status != "UPDATE 1":
        raise StaleIndexWork(
            f"stale sink completion cancelled: {job.doc_id}@v{job.document_version}"
        )
    setattr(job, f"{sink}_state", state)


async def _write_vector_sink(
    conn: Any,
    job: IndexJob,
    payload: dict[str, Any],
) -> int:
    """Write chunks and completion state atomically in Postgres."""
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("index payload chunks must be a list")
    async with conn.transaction():
        await _lock_current_document(conn, job)
        chunks_written = await _write_chunks_pg_conn(conn, job.doc_id, chunks)
        status = await conn.execute(
            """
            UPDATE document_index_jobs
            SET vector_state = 'completed', vector_error = NULL, updated_at = now()
            WHERE doc_id = $1 AND document_version = $2 AND payload_hash = $3
            """,
            job.doc_id,
            job.document_version,
            job.payload_hash,
        )
        if status != "UPDATE 1":
            raise StaleIndexWork(
                f"stale vector completion cancelled: "
                f"{job.doc_id}@v{job.document_version}"
            )
    job.vector_state = "completed"
    return chunks_written


def _reconcile_graph(
    client: Neo4jClient,
    doc: DocumentRecord,
    document_version: int,
) -> None:
    client.run_write(
        """
        MATCH ()-[r]->()
        WHERE r.source_doc_id = $doc_id
          AND coalesce(r.source_doc_version, -1) <> $document_version
        DELETE r
        """,
        {"doc_id": doc.id, "document_version": document_version},
    )
    client.run_write(
        """
        MATCH (n)
        WHERE n.source_doc_id = $doc_id
          AND coalesce(n.source_doc_version, -1) < $document_version
          AND (n:Decision OR n:Constraint OR n:Insight OR n:Signal)
          AND NOT (n)--()
        DELETE n
        """,
        {"doc_id": doc.id, "document_version": document_version},
    )


async def _write_graph_sink(
    conn: Any,
    job: IndexJob,
    doc: DocumentRecord,
    payload: dict[str, Any],
    client: Neo4jClient,
) -> dict[str, int]:
    project = str(payload.get("project") or doc.org_id.removeprefix("org:"))
    raw_extraction = payload.get("extraction_result")
    if not isinstance(raw_extraction, dict):
        raise TypeError("index payload extraction_result must be an object")
    extraction_result = ExtractionResult.model_validate(raw_extraction)
    existing_gmail_node_ids: set[str] = set()
    if doc.source_provider == "gmail":
        _namespace_gmail_semantic_ids(extraction_result, doc.id)
        existing_gmail_node_ids = _existing_extraction_node_ids(
            client, extraction_result
        )

    async with conn.transaction():
        # Hold the canonical document row while Neo4j replacement data is
        # written. A publisher cannot advance current_version during this block.
        await _lock_current_document(conn, job)
        _upsert_doc_node(client, doc, project)
        nodes_upserted = _upsert_semantic_nodes(
            client,
            extraction_result,
            project,
            source_scope=doc.source_scope,
            preserve_existing_scope=(doc.source_provider == "gmail"),
            source_doc_id=doc.id,
            source_doc_version=job.document_version,
            repo=doc.repo,
            doc_type=doc.doc_type,
        )
        entries = _build_edge_entries(
            extraction_result,
            doc.id,
            client,
            source_scope=doc.source_scope,
            existing_node_ids=existing_gmail_node_ids,
            source_doc_version=job.document_version,
        )
        rels_upserted = 0
        if entries:
            result = ingest_entries(client, project, entries)
            rels_upserted = int(result.get("rels_upserted", 0))
        # Old relationships remain queryable until every replacement write has
        # succeeded. MERGE identities make a retry safe after partial writes.
        _reconcile_graph(client, doc, job.document_version)
    return {"nodes_upserted": nodes_upserted, "rels_upserted": rels_upserted}


async def _increment_attempt(conn: Any, job: IndexJob) -> None:
    status = await conn.execute(
        """
        UPDATE document_index_jobs
        SET attempt_count = attempt_count + 1, updated_at = now()
        WHERE doc_id = $1 AND document_version = $2 AND payload_hash = $3
          AND EXISTS (
              SELECT 1 FROM documents
              WHERE id = $1
                AND current_version = $2
                AND processing_state = 'indexing'
          )
        """,
        job.doc_id,
        job.document_version,
        job.payload_hash,
    )
    if status != "UPDATE 1":
        raise StaleIndexWork(
            f"stale index attempt cancelled: {job.doc_id}@v{job.document_version}"
        )


async def _record_sink_failure(
    conn: Any,
    job: IndexJob,
    sink: str,
    error: Exception,
) -> None:
    """Best effort failure metadata; the incomplete job itself remains durable."""
    try:
        await _set_sink_state(conn, job, sink, "failed", str(error))
    except Exception as state_exc:  # noqa: BLE001 - preserve retry semantics
        setattr(job, f"{sink}_state", "failed")
        _log.error(
            "Could not persist %s failure state for %s@v%s: %s",
            sink,
            job.doc_id,
            job.document_version,
            state_exc,
        )


async def _complete_index_sinks(
    conn: Any,
    job: IndexJob,
    doc: DocumentRecord,
    payload: dict[str, Any],
    client: Neo4jClient,
) -> dict[str, Any]:
    """Run only incomplete sinks and raise until both have durably completed."""
    try:
        await _increment_attempt(conn, job)
    except StaleIndexWork:
        raise
    except Exception as exc:
        raise IndexingIncompleteError(job, {"job": str(exc)}) from exc
    summary: dict[str, Any] = {
        "chunks_written": 0,
        "nodes_upserted": 0,
        "rels_upserted": 0,
    }
    errors: dict[str, str] = {}

    if job.vector_state != "completed":
        try:
            summary["chunks_written"] = await _write_vector_sink(conn, job, payload)
            job.vector_state = "completed"
        except StaleIndexWork:
            raise
        except Exception as exc:  # noqa: BLE001 - persist retry state
            errors["vector"] = str(exc)
            await _record_sink_failure(conn, job, "vector", exc)

    if job.graph_state != "completed":
        try:
            summary.update(await _write_graph_sink(conn, job, doc, payload, client))
            await _set_sink_state(conn, job, "graph", "completed")
        except StaleIndexWork:
            raise
        except Exception as exc:  # noqa: BLE001 - persist retry state
            errors["graph"] = str(exc)
            await _record_sink_failure(conn, job, "graph", exc)

    summary["vector_state"] = job.vector_state
    summary["graph_state"] = job.graph_state
    if job.vector_state != "completed" or job.graph_state != "completed":
        raise IndexingIncompleteError(job, errors)
    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def dual_write(
    doc: DocumentRecord,
    project: str,
    chunk_result: list[dict],
    extraction_result: ExtractionResult,
    client: Neo4jClient,
) -> dict:
    """Persist an index job and durably complete both indexing sinks."""
    job, payload = await _ensure_index_job(
        doc, project, chunk_result, extraction_result
    )
    async with _index_job_lease(job) as conn:
        return await _complete_index_sinks(conn, job, doc, payload, client)


async def resume_index_job(
    doc: DocumentRecord,
    client: Neo4jClient,
) -> dict:
    """Resume an indexing-state document without scoring or extraction."""
    job = await _load_job(doc.id, doc.current_version)
    if job is None:
        raise RuntimeError(
            f"indexing document has no durable job: {doc.id}@v{doc.current_version}"
        )
    payload = await _read_index_payload(job)
    async with _index_job_lease(job) as conn:
        return await _complete_index_sinks(conn, job, doc, payload, client)
