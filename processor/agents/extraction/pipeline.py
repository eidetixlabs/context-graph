"""
processor/agents/extraction/pipeline.py — Extraction orchestrator.

Uses the canonical_resolution strategy:
  mention-extract → resolve → pronoun → sync existing
  → Phase 2 → Phase 3 (per-chunk) → merge → persist new → sync new
  → dual_write → save map.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from base.db.client import db_pool
from base.neo4j_client import Neo4jClient
from processor.agents.extraction.phase2_context import run_context_gathering
from processor.chunking import chunk_and_embed, chunk_for_extraction
from processor.dual_write import dual_write
from processor.embedding import embed_text
from base.db.identity import _employee_entity_id
from processor.resolution.models import ResolutionMap
from processor.schemas.extraction import EntityNode, ExtractionResult, SemanticEdge
from processor.schemas.identity import IdentityResolutionResult, ResolvedIdentity
from processor.schemas.scoring import ScoringResult
from processor.state_machine import DocumentRecord

_log = logging.getLogger(__name__)


async def run_extraction_pipeline(
    doc: DocumentRecord,
    content: str,
    scoring_result: ScoringResult,
    neo4j_client: Neo4jClient | None = None,
) -> ExtractionResult:
    """
    Execute the full three-phase extraction pipeline.

    Phase 1 — Identity Resolution:
        Resolves all mentions of people, services, projects, tools to canonical
        entity IDs. Pre-fetches existing entities from the graph for warm-start.

    Phase 2 — Context Gathering:
        Actively traverses the knowledge graph using search, get_graph, and
        get_graph_expanded tools to map the existing context terrain.

    Phase 3 — Semantic Extraction:
        Uses the document + Phase 1 + Phase 2 outputs to extract structured
        Decision/Constraint/Insight/Signal nodes with edges. No tools.

    Args:
        doc:            Document record from Postgres.
        content:        Full text content read from S3.
        scoring_result: Scoring output (signal_types, suggested_topics).

    Returns:
        ExtractionResult with nodes and edges ready for ingestion.
    """
    project = _derive_project(doc.org_id)

    _log.info(
        "Extraction pipeline start: doc=%s provider=%s project=%s",
        doc.id,
        doc.source_provider,
        project,
    )

    # Step 1: chunk_and_embed — always uses the global strategy so retrieval
    # chunks are small and embeddable (bge-large caps at 512 tokens).
    # These chunks go to PgVector for recall() chunk search.
    retrieval_chunks = await chunk_and_embed(content)

    # Step 2: Determine extraction chunks separately.
    # Providers that benefit from single-shot extraction (e.g. Fireflies) use
    # whole_document strategy so the LLM sees the full meeting in one pass.
    # This is decoupled from retrieval so bge embeddings are never truncated.
    extraction_strategy = _extraction_strategy_for_provider(doc.source_provider)
    if extraction_strategy == "whole_document":
        extraction_chunks = await chunk_for_extraction(content, strategy="whole_document")
        _log.info(
            "Extraction pipeline: provider=%s using whole_document extraction "
            "(%d retrieval chunk(s) still written for recall)",
            doc.source_provider,
            len(retrieval_chunks),
        )
    else:
        # Default: extraction uses the same chunks as retrieval (pre-computed,
        # reuses splitting without re-embedding — embeddings only needed for PgVector)
        extraction_chunks = [
            {k: v for k, v in c.items() if k != "embedding"}
            for c in retrieval_chunks
        ]

    # Step 3: Run extraction phases with extraction-specific chunks
    extraction_result = await _run_extraction_phases(
        doc, content, scoring_result, project, neo4j_client, chunks=extraction_chunks
    )

    _log.info(
        "Extraction pipeline complete: doc=%s nodes=%d edges=%d "
        "retrieval_chunks=%d extraction_chunks=%d",
        doc.id,
        len(extraction_result.nodes),
        len(extraction_result.edges),
        len(retrieval_chunks),
        len(extraction_chunks),
    )

    # Durable dual write. Any incomplete sink raises so the worker leaves the
    # document in indexing and resumes from the canonical payload next cycle.
    if neo4j_client is not None:
        write_summary = await dual_write(
            doc=doc,
            project=project,
            chunk_result=retrieval_chunks,
            extraction_result=extraction_result,
            client=neo4j_client,
        )
        _log.info("dual_write result for doc=%s: %s", doc.id, write_summary)
    else:
        _log.debug("dual_write skipped for doc=%s (no neo4j_client supplied)", doc.id)

    return extraction_result


async def _run_extraction_phases(
    doc: DocumentRecord,
    content: str,
    scoring_result: ScoringResult,
    project: str,
    neo4j_client: Neo4jClient | None = None,
    chunks: list[dict] | None = None,
) -> ExtractionResult:
    """Run the extraction pipeline and return the result."""
    extraction_result = await _run_canonical_pipeline(
        doc=doc,
        content=content,
        scoring_result=scoring_result,
        project=project,
        neo4j_client=neo4j_client,
        chunks=chunks,
    )
    await _embed_semantic_nodes(extraction_result)
    return extraction_result


async def _run_canonical_pipeline(
    doc: DocumentRecord,
    content: str,
    scoring_result: ScoringResult,
    project: str,
    neo4j_client: Neo4jClient | None = None,
    chunks: list[dict] | None = None,
) -> ExtractionResult:
    """
    Canonical resolution pipeline:
      mention-extract → resolve → pronoun → sync existing → Phase 2
      → Phase 3 (per-chunk) → merge → persist new → sync new → save map
    """
    from processor.agents.extraction.phase1_mentions import run_mention_extraction  # noqa: PLC0415
    from processor.agents.extraction.phase3_extract_chunk import run_chunk_extraction  # noqa: PLC0415
    from processor.extraction.merge import merge_chunk_extractions  # noqa: PLC0415
    from processor.resolution.persistence import (  # noqa: PLC0415
        persist_entities_from_extraction,
    )
    from processor.resolution.pronouns import resolve_pronouns, resolve_role_references  # noqa: PLC0415
    from processor.resolution.resolver import (  # noqa: PLC0415
        build_known_entities_block,
        build_resolved_entities,
        fetch_org_data,
        resolve_mentions,
        scan_chunks_for_known_aliases,
    )
    _log.info(
        "Phase 1 CanonicalPipeline start: doc=%s content_len=%d",
        doc.id,
        len(content),
    )
    pool = db_pool.get()
    chunks = chunks or []

    # ── Pre-flight: fetch org alias data before Phase 1 LLM ──────────────────
    # Two uses: (a) inject the FULL known-entity mapping (canonical ← aliases)
    # into the Phase 1 LLM prompt — identical for every chunk, a consistent view
    # like resolution — so the LLM maps surface variants ("Riya K.", "Riya") onto
    # the existing canonical identity itself instead of minting duplicates; and
    # (b) inject pre-resolved MentionCandidates (confidence=1.0) for known aliases
    # whose literal text appears in a chunk but the LLM might still miss.
    preflight_mentions: list = []
    per_chunk_known_aliases: list[str] = [""] * len(chunks)
    async with pool.acquire() as _preflight_conn:
        org_data = await fetch_org_data(_preflight_conn, doc.org_id, consistent=False)

    if org_data.fuzzy_index:
        known_entities_block = build_known_entities_block(org_data)
        # Same consistent mapping for every chunk.
        per_chunk_known_aliases = [known_entities_block] * len(chunks)
        preflight_mentions = scan_chunks_for_known_aliases(chunks, org_data)
        _log.info(
            "Pre-flight known-entity mapping: doc=%s entities=%d block_chars=%d "
            "preflight_mentions=%d",
            doc.id,
            sum(len(v) for v in org_data.fuzzy_index.values()),
            len(known_entities_block),
            len(preflight_mentions),
        )

    # ── Step 1: Mention extraction (chunks passed from caller) ────────────────
    chunks, mentions, keywords = await run_mention_extraction(
        doc=doc,
        content=content,
        chunks=chunks,
        per_chunk_known_aliases=per_chunk_known_aliases,
    )

    # Merge preflight mentions that the LLM didn't already surface
    if preflight_mentions:
        existing_keys: set[tuple[str, str]] = {
            (m.mention_text.lower(), m.mention_type) for m in mentions
        }
        injected = 0
        for pm in preflight_mentions:
            key = (pm.mention_text.lower(), pm.mention_type)
            if key not in existing_keys:
                existing_keys.add(key)
                mentions.append(pm)
                injected += 1
        if injected:
            _log.info(
                "Injected %d preflight mentions not found by LLM: doc=%s",
                injected,
                doc.id,
            )

    _log.info("Phase 1 keywords extracted: doc=%s count=%d", doc.id, len(keywords))

    # ── Step 2: Deterministic resolution ─────────────────────────────────────
    async with pool.acquire() as conn:
        resolution_map = await resolve_mentions(
            mentions=mentions,
            org_id=doc.org_id,
            conn=conn,
        )
        resolution_map.doc_id = doc.id
        resolution_map.doc_version = doc.current_version or 1

        build_resolved_entities(resolution_map, mentions, doc.org_id)

        # ── Step 3: Pronoun and role-reference resolution ─────────────────────
        resolve_pronouns(mentions, resolution_map)
        await resolve_role_references(mentions, resolution_map, doc.org_id, conn)

        # Sanity-check: resolved_entities must be deduplicated by entity_id.
        _seen = set()
        _dupes = [e.entity_id for e in resolution_map.resolved_entities if e.entity_id in _seen or _seen.add(e.entity_id)]  # type: ignore[func-returns-value]
        if _dupes:
            seen_ids: set[str] = set()
            resolution_map.resolved_entities = [
                e for e in resolution_map.resolved_entities
                if e.entity_id not in seen_ids and not seen_ids.add(e.entity_id)  # type: ignore[func-returns-value]
            ]

        _log.info(
            "Resolution pipeline resolution_map: doc=%s map=%s",
            doc.id,
            resolution_map.model_dump_json(),
        )

    # ── Step 5: Phase 2: Context Gathering ────────────────────────────────────
    from processor.resolution.metadata import compute_document_metadata  # noqa: PLC0415
    identity_result = _resolution_map_to_identity_result(resolution_map)
    doc_metadata = compute_document_metadata(doc=doc, mentions=mentions, keywords=keywords)
    context_result = await run_context_gathering(
        doc=doc,
        chunks=chunks,
        keywords=keywords,
        identity_result=identity_result,
        scoring_result=scoring_result,
        project=project,
    metadata=doc_metadata,
    )

    # ── Step 6: Phase 3 per-chunk extraction ──────────────────────────────────
    from processor.config import config as _cfg  # noqa: PLC0415
    concurrency = getattr(_cfg, "extraction_chunk_concurrency", 4)
    chunk_results = await run_chunk_extraction(
        doc=doc,
        chunks=chunks,
        resolution_map=resolution_map,
        context_result=context_result,
        scoring_result=scoring_result,
        concurrency=concurrency,
    )

    # ── Step 7: Merge chunk extractions ──────────────────────────────────────
    merged_extraction = merge_chunk_extractions(chunk_results)

    # ── Step 7.5: Inject document author EntityNode + AUTHORED_BY edge ───────
    await _inject_document_author(doc, merged_extraction, pool)

    # ── Step 7.6: Inject all resolved attendees (Fireflies meetings) ─────────
    # Ensures every person in the resolution_map appears in the graph even
    # when the LLM didn't happen to extract them as explicit Entity nodes.
    if doc.source_provider == "fireflies":
        _inject_attendee_entities(doc, merged_extraction, resolution_map)

    # ── Step 7.7: Gmail RBAC participants from authoritative scope members ───
    if doc.source_provider == "gmail":
        await _inject_gmail_scope_members(doc, merged_extraction, pool)

    # ── Step 7.8: Jira participants from payload metadata ────────────────────
    if doc.source_provider == "jira":
        await _inject_jira_participants(doc, content, merged_extraction, pool)

    # Repository profiles always anchor extracted claims/entities to one
    # canonical GitHub repository entity, even if the LLM omitted it.
    if doc.source_provider == "github-profile":
        _inject_profile_repository(doc, merged_extraction)

    # ── Step 8: Persist final LLM Entity nodes ───────────────────────────────
    async with pool.acquire() as conn:
        await persist_entities_from_extraction(
            doc=doc,
            extraction_result=merged_extraction,
            resolution_map=resolution_map,
            conn=conn,
        )

    _log.info(
        "Canonical pipeline complete: doc=%s nodes=%d edges=%d map_entries=%d",
        doc.id,
        len(merged_extraction.nodes),
        len(merged_extraction.edges),
        len(resolution_map.entries),
    )
    return merged_extraction


def _resolution_map_to_identity_result(
    resolution_map: ResolutionMap,
) -> IdentityResolutionResult:
    """Convert a ResolutionMap into a legacy IdentityResolutionResult for Phase 2."""
    identities: list[ResolvedIdentity] = []
    for entity in resolution_map.resolved_entities:
        identities.append(
            ResolvedIdentity(
                mention=entity.canonical_name,
                entity_type=entity.entity_type if entity.entity_type in (  # type: ignore[arg-type]
                    "person", "service", "organization", "repository",
                    "tool", "object", "location", "event"
                ) else "object",
                resolved_id=entity.entity_id,
                is_new=entity.is_new,
                proposed_canonical_name=entity.canonical_name if entity.is_new else None,
                confidence=entity.confidence,
            )
        )
    return IdentityResolutionResult(
        identities=identities,
        resolution_notes="Generated by canonical resolution pipeline.",
    )


async def _embed_semantic_nodes(extraction_result: ExtractionResult) -> None:
    """Embed Decision/Constraint/Insight/Signal/ActionItem nodes concurrently."""
    _EMBEDDABLE_TYPES = {"Decision", "Constraint", "Insight", "Signal", "Preference", "Fact", "ActionItem"}
    embed_tasks: list[tuple[Any, asyncio.Task[list[float]]]] = []
    for node in extraction_result.nodes:
        node_type = node.get("type") if isinstance(node, dict) else getattr(node, "type", None)
        if node_type not in _EMBEDDABLE_TYPES:
            continue
        node_text = _node_embedding_text(node)
        if not node_text:
            _log.warning(
                "Skipping embedding for %s node with no embeddable text: %r",
                node_type, node,
            )
            continue
        task = asyncio.create_task(embed_text(node_text))
        embed_tasks.append((node, task))

    results = await asyncio.gather(*(task for _, task in embed_tasks), return_exceptions=True)
    for (node, _task), result in zip(embed_tasks, results):
        if isinstance(result, BaseException):
            node_type = node.get("type") if isinstance(node, dict) else getattr(node, "type", None)
            _log.warning("Embedding failed for %s node, leaving unembedded: %s", node_type, result)
            continue
        if isinstance(node, dict):
            node["embedding"] = result
        else:
            node.embedding = result


def _node_embedding_text(node: Any) -> str:
    """Produce a compact text representation of a semantic node for embedding."""
    if isinstance(node, dict):
        node_type = node.get("type", "")
        if node_type == "Preference":
            parts = [node.get("category", ""), node.get("preference", ""), node.get("context", "")]
        elif node_type == "Fact":
            parts = [node.get("subject", ""), node.get("predicate", ""), node.get("object", "")]
        elif node_type == "ActionItem":
            parts = [node.get("label", ""), node.get("source", "")]
        else:
            parts = [node.get("label", ""), node.get("description", ""),
                     node.get("body", ""), node.get("rationale", ""),
                     node.get("trigger", ""), node.get("symptom", ""),
                     node.get("observation", ""), node.get("actually_means", "")]
    else:
        node_type = getattr(node, "type", "")
        if node_type == "Preference":
            parts = [getattr(node, "category", ""), getattr(node, "preference", ""), getattr(node, "context", "") or ""]
        elif node_type == "Fact":
            parts = [getattr(node, "subject", ""), getattr(node, "predicate", ""), getattr(node, "object", "")]
        elif node_type == "ActionItem":
            parts = [getattr(node, "label", ""), getattr(node, "source", "") or ""]
        else:
            parts = [getattr(node, "label", ""), getattr(node, "description", ""),
                     getattr(node, "body", ""), getattr(node, "rationale", ""),
                     getattr(node, "trigger", ""), getattr(node, "symptom", ""),
                     getattr(node, "observation", ""), getattr(node, "actually_means", "")]
    return " ".join(p for p in parts if p)


def _derive_project(org_id: str) -> str:
    """Convert 'org:acme-corp' → 'acme-corp'."""
    return org_id.removeprefix("org:")


# Providers that use whole-document (single-shot) extraction instead of the
# global structural_entity chunking strategy. Add new providers here when they
# have similar properties: dense, short content where the LLM benefits from
# seeing everything in one pass.
_WHOLE_DOCUMENT_PROVIDERS: frozenset[str] = frozenset(
    {"fireflies", "github-profile"}
)


def _extraction_strategy_for_provider(source_provider: str) -> str:
    """Return the chunking strategy to use for LLM extraction for a given provider.

    Decoupled from retrieval chunking: retrieval always uses the global strategy
    (small, embeddable chunks for PgVector); extraction may use a different
    strategy when single-shot quality matters more than chunk granularity.
    """
    if source_provider in _WHOLE_DOCUMENT_PROVIDERS:
        return "whole_document"
    return "default"


def _inject_profile_repository(
    doc: DocumentRecord,
    extraction_result: ExtractionResult,
) -> None:
    """Attach every profile claim/entity to a canonical owner/repo Entity."""
    from processor.resolution.entity_ids import slugify_entity_component  # noqa: PLC0415

    repository = doc.parent_ref or doc.source_ref.split("/_profile/", 1)[0]
    owner, _, name = repository.partition("/")
    if not owner or not name:
        return
    repository_id = (
        f"entity:repository:{slugify_entity_component(owner)}:"
        f"{slugify_entity_component(name)}"
    )

    existing_repository = None
    remapped_id = None
    for node in extraction_result.nodes:
        node_type = node.get("type") if isinstance(node, dict) else getattr(node, "type", "")
        entity_type = (
            node.get("entity_type")
            if isinstance(node, dict)
            else getattr(node, "entity_type", "")
        )
        canonical_name = (
            node.get("canonical_name")
            if isinstance(node, dict)
            else getattr(node, "canonical_name", "")
        )
        if node_type == "Entity" and entity_type == "repository" and (
            str(canonical_name).lower() == repository.lower()
        ):
            existing_repository = node
            old_id = node.get("id") if isinstance(node, dict) else node.id
            if old_id != repository_id:
                remapped_id = old_id
                if isinstance(node, dict):
                    node["id"] = repository_id
                else:
                    node.id = repository_id
            break

    if remapped_id:
        for edge in extraction_result.edges:
            if edge.from_id == remapped_id:
                edge.from_id = repository_id
            if edge.to_id == remapped_id:
                edge.to_id = repository_id

    if existing_repository is None:
        extraction_result.nodes.append(
            EntityNode(
                id=repository_id,
                label=repository,
                entity_type="repository",
                canonical_name=repository,
                aliases=json.dumps([name, repository]),
                origin="github-profile",
            )
        )

    existing_about = {
        (edge.from_id, edge.to_id)
        for edge in extraction_result.edges
        if edge.rel == "ABOUT"
    }
    for node in extraction_result.nodes:
        node_id = node.get("id") if isinstance(node, dict) else getattr(node, "id", "")
        if not node_id or node_id == repository_id:
            continue
        if (node_id, repository_id) in existing_about:
            continue
        extraction_result.edges.append(
            SemanticEdge(
                from_id=node_id,
                rel="ABOUT",
                to_id=repository_id,
                edge_context=f"Extracted repository-profile knowledge about {repository}.",
                inferred=False,
            )
        )


async def _inject_document_author(
    doc: DocumentRecord,
    extraction_result: ExtractionResult,
    pool,
) -> None:
    """
    Inject an EntityNode + AUTHORED_BY edge for the document author.

    Mirrors the trace pipeline's entity-stub pattern: looks up the author's
    existing entity via entity_registry.employee_id, falling back to a
    deterministic ID from the employee's canonical_name. persist_entities_from_extraction
    (Step 8) handles the Postgres write; dual_write handles Neo4j.

    Idempotent: no-op if node/edge already present. No-op if author_employee_id is unset.
    """
    if not doc.author_employee_id:
        return

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    emp.canonical_name AS name,
                    er.id              AS entity_id
                FROM employees emp
                LEFT JOIN entity_registry er
                       ON er.employee_id = emp.id
                      AND er.org_id      = emp.org_id
                      AND er.entity_type = 'person'
                      AND er.status      = 'active'
                WHERE emp.id     = $1
                  AND emp.org_id = $2
                LIMIT 1
                """,
                doc.author_employee_id,
                doc.org_id,
            )
    except Exception as exc:
        _log.warning(
            "Author lookup failed: doc=%s employee=%s error=%s",
            doc.id, doc.author_employee_id, exc,
        )
        return

    if row is None:
        _log.warning(
            "Author employee not found: doc=%s employee=%s",
            doc.id, doc.author_employee_id,
        )
        return

    canonical_name = str(row["name"] or doc.author_employee_id).strip()
    entity_id: str = str(row["entity_id"] or "").strip()

    if not entity_id:
        entity_id = _employee_entity_id(doc.org_id, doc.author_employee_id)

    already_has_node = any(
        (node.get("id") if isinstance(node, dict) else getattr(node, "id", None)) == entity_id
        for node in extraction_result.nodes
    )
    if not already_has_node:
        extraction_result.nodes.append(
            EntityNode(
                id=entity_id,
                label=canonical_name,
                entity_type="person",
                canonical_name=canonical_name,
                origin="document_author",
            )
        )

    already_has_edge = any(
        edge.rel == "AUTHORED_BY"
        and edge.from_id == doc.id
        and edge.to_id == entity_id
        for edge in extraction_result.edges
    )
    if not already_has_edge:
        extraction_result.edges.append(
            SemanticEdge(
                from_id=doc.id,
                rel="AUTHORED_BY",
                to_id=entity_id,
                edge_context=f"Document {doc.id} was authored by {canonical_name}.",
                inferred=False,
            )
        )

    _log.info(
        "Author entity injected: doc=%s entity=%s name=%r",
        doc.id, entity_id, canonical_name,
    )


def _inject_attendee_entities(
    doc: DocumentRecord,
    extraction_result: ExtractionResult,
    resolution_map: ResolutionMap,
) -> None:
    """
    Inject EntityNode + ATTENDED_BY edge for every resolved person in the
    resolution map who is not already present in the extraction result.

    Complements _inject_document_author (which covers only the organizer via
    doc.author_employee_id). This guarantees all meeting participants appear in
    the knowledge graph regardless of whether the Phase 3 LLM explicitly named
    them in extracted content.

    Scoped to meeting providers (fireflies) where resolved_entities are attendees.
    Idempotent: skips nodes/edges already present.
    """
    existing_node_ids: set[str] = {
        node_id
        for node in extraction_result.nodes
        for node_id in [node.get("id") if isinstance(node, dict) else getattr(node, "id", None)]
        if isinstance(node_id, str) and node_id
    }
    existing_edge_keys: set[tuple[str, str, str]] = {
        (edge.from_id, edge.rel, edge.to_id)
        for edge in extraction_result.edges
    }
    injected_nodes = 0

    for entity in resolution_map.resolved_entities:
        entity_id = entity.entity_id
        canonical_name = entity.canonical_name
        if not entity_id or not canonical_name:
            continue

        if entity_id not in existing_node_ids:
            extraction_result.nodes.append(
                EntityNode(
                    id=entity_id,
                    label=canonical_name,
                    entity_type=entity.entity_type or "person",
                    canonical_name=canonical_name,
                    origin="meeting_attendee",
                )
            )
            existing_node_ids.add(entity_id)
            injected_nodes += 1

        edge_key = (doc.id, "INVOLVES", entity_id)
        if edge_key not in existing_edge_keys:
            extraction_result.edges.append(
                SemanticEdge(
                    from_id=doc.id,
                    rel="INVOLVES",
                    to_id=entity_id,
                    edge_context=f"{canonical_name} was a participant in the meeting documented in {doc.id}.",
                    inferred=False,
                )
            )
            existing_edge_keys.add(edge_key)

    if injected_nodes:
        _log.info(
            "Attendee entities injected: doc=%s new_nodes=%d",
            doc.id,
            injected_nodes,
        )


async def _inject_gmail_scope_members(
    doc: DocumentRecord,
    extraction_result: ExtractionResult,
    pool,
) -> None:
    """
    Inject Entity nodes + Document INVOLVES edges for Gmail scope members.

    Gmail membership is managed by the crawler in Postgres channel_members.
    This is the authoritative source for query-time RBAC, so graph participation
    edges should mirror that membership rather than inferred body mentions.
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    cm.employee_id,
                    er.id AS entity_id,
                    COALESCE(er.canonical_name, e.canonical_name, cm.external_user_id) AS canonical_name
                FROM channel_members cm
                LEFT JOIN employees e
                       ON e.id = cm.employee_id
                LEFT JOIN entity_registry er
                       ON er.employee_id = cm.employee_id
                      AND er.org_id = e.org_id
                      AND er.entity_type = 'person'
                      AND er.status = 'active'
                WHERE cm.channel_id = $1
                  AND cm.source_provider = 'gmail'
                  AND cm.employee_id IS NOT NULL
                """,
                doc.source_scope,
            )
    except Exception as exc:
        _log.warning(
            "Gmail scope member lookup failed: doc=%s scope=%s error=%s",
            doc.id,
            doc.source_scope,
            exc,
        )
        return

    if not rows:
        return

    existing_node_ids: set[str] = {
        node_id
        for node in extraction_result.nodes
        for node_id in [node.get("id") if isinstance(node, dict) else getattr(node, "id", None)]
        if isinstance(node_id, str) and node_id
    }
    existing_edge_keys: set[tuple[str, str, str]] = {
        (edge.from_id, edge.rel, edge.to_id)
        for edge in extraction_result.edges
    }

    injected_nodes = 0
    injected_edges = 0
    for row in rows:
        entity_id = str(row["entity_id"] or "").strip()
        canonical_name = str(row["canonical_name"] or "").strip()
        if not canonical_name:
            continue
        if not entity_id:
            entity_id = _employee_entity_id(doc.org_id, str(row["employee_id"]))

        if entity_id not in existing_node_ids:
            extraction_result.nodes.append(
                EntityNode(
                    id=entity_id,
                    label=canonical_name,
                    entity_type="person",
                    canonical_name=canonical_name,
                    origin="gmail_scope_member",
                )
            )
            existing_node_ids.add(entity_id)
            injected_nodes += 1

        edge_key = (doc.id, "INVOLVES", entity_id)
        if edge_key not in existing_edge_keys:
            extraction_result.edges.append(
                SemanticEdge(
                    from_id=doc.id,
                    rel="INVOLVES",
                    to_id=entity_id,
                    edge_context=(
                        f"{canonical_name} is a Gmail scope member for message {doc.source_ref}."
                    ),
                    inferred=False,
                )
            )
            existing_edge_keys.add(edge_key)
            injected_edges += 1

    if injected_nodes or injected_edges:
        _log.info(
            "Gmail members injected: doc=%s nodes=%d involves_edges=%d",
            doc.id,
            injected_nodes,
            injected_edges,
        )


def _safe_slug(value: str) -> str:
    chars: list[str] = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_", ":"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "unknown"


async def _inject_jira_participants(
    doc: DocumentRecord,
    content: str,
    extraction_result: ExtractionResult,
    pool,
) -> None:
    """
    Inject Jira reporter/assignee/comment authors as Entity + INVOLVES edges.
    """
    import json  # noqa: PLC0415

    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return
    if not isinstance(payload, dict):
        return

    participants: list[dict[str, str]] = []
    for actor in (payload.get("reporter"), payload.get("assignee")):
        if isinstance(actor, dict):
            account_id = str(actor.get("account_id") or "").strip()
            display_name = str(actor.get("display_name") or "").strip() or "unknown"
            if account_id:
                participants.append({"account_id": account_id, "display_name": display_name})
    comments = payload.get("comments")
    if isinstance(comments, list):
        for row in comments:
            if not isinstance(row, dict):
                continue
            author = row.get("author")
            if not isinstance(author, dict):
                continue
            account_id = str(author.get("account_id") or "").strip()
            display_name = str(author.get("display_name") or "").strip() or "unknown"
            if account_id:
                participants.append({"account_id": account_id, "display_name": display_name})

    if not participants:
        return

    deduped: dict[str, str] = {}
    for participant in participants:
        account_id = participant["account_id"]
        if account_id not in deduped:
            deduped[account_id] = participant["display_name"]

    account_ids = sorted(deduped.keys())
    mapping: dict[str, tuple[str | None, str | None, str]] = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    im.external_id AS account_id,
                    im.employee_id AS employee_id,
                    er.id AS entity_id,
                    COALESCE(er.canonical_name, e.canonical_name) AS canonical_name
                FROM identity_mappings im
                LEFT JOIN employees e
                       ON e.id = im.employee_id
                      AND e.org_id = im.org_id
                LEFT JOIN entity_registry er
                       ON er.employee_id = im.employee_id
                      AND er.org_id = im.org_id
                      AND er.entity_type = 'person'
                      AND er.status = 'active'
                WHERE im.org_id = $1
                  AND im.provider = 'jira'
                  AND im.external_id = ANY($2::text[])
                """,
                doc.org_id,
                account_ids,
            )
    except Exception as exc:
        _log.warning("Jira participant lookup failed: doc=%s err=%s", doc.id, exc)
        return

    for row in rows:
        account_id = str(row["account_id"] or "").strip()
        if not account_id:
            continue
        canonical_name = str(row["canonical_name"] or "").strip() or deduped.get(account_id, "unknown")
        mapping[account_id] = (
            str(row["employee_id"] or "").strip() or None,
            str(row["entity_id"] or "").strip() or None,
            canonical_name,
        )

    existing_node_ids: set[str] = {
        node_id
        for node in extraction_result.nodes
        for node_id in [node.get("id") if isinstance(node, dict) else getattr(node, "id", None)]
        if isinstance(node_id, str) and node_id
    }
    existing_edge_keys: set[tuple[str, str, str]] = {
        (edge.from_id, edge.rel, edge.to_id)
        for edge in extraction_result.edges
    }

    injected_nodes = 0
    injected_edges = 0
    for account_id, display_name in deduped.items():
        mapped = mapping.get(account_id)
        canonical_name = display_name
        entity_id: str
        if mapped is not None:
            _employee_id, mapped_entity_id, mapped_name = mapped
            canonical_name = mapped_name or display_name
            entity_id = mapped_entity_id or f"entity:person:{_safe_slug(doc.org_id)}:{_safe_slug(account_id)}"
        else:
            entity_id = f"entity:person:{_safe_slug(doc.org_id)}:{_safe_slug(account_id)}"

        if entity_id not in existing_node_ids:
            extraction_result.nodes.append(
                EntityNode(
                    id=entity_id,
                    label=canonical_name,
                    entity_type="person",
                    canonical_name=canonical_name,
                    origin="jira_participant",
                )
            )
            existing_node_ids.add(entity_id)
            injected_nodes += 1

        edge_key = (doc.id, "INVOLVES", entity_id)
        if edge_key not in existing_edge_keys:
            extraction_result.edges.append(
                SemanticEdge(
                    from_id=doc.id,
                    rel="INVOLVES",
                    to_id=entity_id,
                    edge_context=(
                        f"{canonical_name} participated in Jira issue {doc.source_ref}."
                    ),
                    inferred=False,
                )
            )
            existing_edge_keys.add(edge_key)
            injected_edges += 1

    if injected_nodes or injected_edges:
        _log.info(
            "Jira participants injected: doc=%s nodes=%d edges=%d",
            doc.id,
            injected_nodes,
            injected_edges,
        )
