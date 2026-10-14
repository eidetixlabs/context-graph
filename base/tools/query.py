"""
tools/query.py — Graph query tools exposed via MCP:
  query_context_vector  — semantic similarity search (Neo4j vector indexes)
  search_memory         — fulltext substring search
  search_entities       — entity name/alias search
  list_topics           — list Topic nodes
  get_node              — fetch a single node by id
  traverse              — graph hop from a node
  query_subgraph        — full subgraph expansion from semantic seeds
  hybrid_search         — parallel PgVector + Neo4j search with RRF merge
  assign_topic          — create MEMBER_OF edge
  list_projects         — list project slugs
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import hashlib
import json
import logging
import re
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as _dc_field
from datetime import datetime, timezone
from typing import Optional

from base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recall profiling — lightweight span tree logged after each recall() call
# #TODO : This is a util class. 
# ---------------------------------------------------------------------------

@dataclass
class _ProfileSpan:
    name: str
    start: float = _dc_field(default_factory=time.perf_counter)
    end: Optional[float] = None
    children: list["_ProfileSpan"] = _dc_field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        if self.end is None:
            return (time.perf_counter() - self.start) * 1000
        return (self.end - self.start) * 1000

    def finish(self) -> None:
        if self.end is None:
            self.end = time.perf_counter()


_recall_span: contextvars.ContextVar[Optional[_ProfileSpan]] = contextvars.ContextVar(
    "_recall_span", default=None
)


@contextlib.asynccontextmanager
async def _pspan(name: str):
    """Async profiling span — records elapsed time and nests under current parent."""
    parent = _recall_span.get()
    span = _ProfileSpan(name=name)
    if parent is not None:
        parent.children.append(span)
    token = _recall_span.set(span)
    try:
        yield span
    finally:
        span.finish()
        _recall_span.reset(token)


@contextlib.contextmanager
def _pspan_sync(name: str):
    """Sync profiling span — safe to use inside asyncio.to_thread workers."""
    parent = _recall_span.get()
    span = _ProfileSpan(name=name)
    if parent is not None:
        parent.children.append(span)
    token = _recall_span.set(span)
    try:
        yield span
    finally:
        span.finish()
        _recall_span.reset(token)


def _log_profile_tree(span: _ProfileSpan, prefix: str = "", is_last: bool = True) -> None:
    connector = "└─ " if is_last else "├─ "
    logger.info("[recall:profile] %s%s%s  %.1fms", prefix, connector, span.name, span.elapsed_ms)
    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(span.children):
        _log_profile_tree(child, child_prefix, is_last=(i == len(span.children) - 1))

# ---------------------------------------------------------------------------
# High-signal relationship set (default for query_context expansion)
# ---------------------------------------------------------------------------

HIGH_SIGNAL_RELS = {
    "SHAPED_BY", "EXPLAINS", "CAUSES", "CONTRADICTS",
    "LIFTED", "SUPERSEDES", "ABANDONED",
}

ALL_RELS = HIGH_SIGNAL_RELS | {
    "MEMBER_OF", "SUBTOPIC_OF", "MENTIONS", "HAS_PREFERENCE",
    "ABOUT", "APPLIES_TO", "AFFECTS", "OWNS", "BLOCKS", "INVOLVES",
    "EXTRACTED_FROM", "AUTHORED_BY",
}

SEMANTIC_LABELS = ["Decision", "Constraint", "Insight", "Signal"]
EXPANDABLE_LABELS = SEMANTIC_LABELS + ["Preference", "Fact"]
ALL_LABELS      = EXPANDABLE_LABELS + ["Topic", "Entity"]

SEARCH_SNIPPET_CHARS = 200
AGENT_METADATA_EXCLUSIONS = {
    "embedding",
    "source_chat",
    "source_s3_url",
    "source_doc_id",
    "source_scope",
    "project",
    "scope",
    "superseded_by",
    "created_at",
    "tag",
    "origin",
}
PRIMARY_CONTENT_FIELDS: dict[str, str] = {
    "Decision": "rationale",
    "Constraint": "description",
    "Insight": "body",
    "Signal": "observation",
    "Preference": "preference",
    "Fact": "statement",
}
SECONDARY_CONTENT_FIELDS: dict[str, list[str]] = {
    "Decision": ["trigger", "alternatives", "confidence"],
    "Constraint": ["hard", "symptom", "source"],
    "Insight": ["failure_without"],
    "Signal": ["actually_means", "appears_to_mean", "conditions", "evidence"],
    "Document": ["source_provider"],
    "Entity": ["entity_type"],
    "Preference": ["context"],
    "Fact": ["category"],
}


# ---------------------------------------------------------------------------
# list_projects
# ---------------------------------------------------------------------------

def list_projects(client: Neo4jClient) -> list[str]:
    rows = client.run("""
        MATCH (n)
        WHERE n.project IS NOT NULL AND n.project <> ''
        RETURN DISTINCT n.project AS project
        ORDER BY project
    """)
    return [r["project"] for r in rows]


# ---------------------------------------------------------------------------
# search_memory  (substring search)
# ---------------------------------------------------------------------------

def search_memory(
    client: Neo4jClient,
    query: str,
    project: str | None = None,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> list[dict]:
    if not query.strip():
        return []
    sanitized_query = _sanitize_fulltext_query(query)
    if not sanitized_query:
        return []
    project_filter = "AND n.project = $project" if project else ""
    params: dict = {"query": sanitized_query}
    if project:
        params["project"] = project
    repo_filter = "AND (n.repo IS NULL OR n.repo = $repo)" if repo else ""
    if repo:
        params["repo"] = repo

    scope_filter = ""
    if allowed_scopes is not None:
        scope_filter = "AND (n:Entity OR n.source_scope IN $allowed_scopes)"
        params["allowed_scopes"] = allowed_scopes

    # Single UNWIND query covers all five content fulltext indexes in one round-trip.
    # preference_fulltext / fact_fulltext  → staleness: valid_until IS NULL
    # semantic_fulltext                    → staleness: superseded_by IS NULL
    # action_item_fulltext / entity_fulltext → no expiry concept; all nodes surface
    # search_entities() still exists for dedicated entity-only lookup by name.
    # Subquery CALL limits rows per index before aggregation and strips embedding server-side
    # (apoc.map.removeKey) so the 384-float vector never crosses the wire.
    unwind_cypher = f"""
        UNWIND $ft_indexes AS idx
        CALL {{
            WITH idx
            CALL db.index.fulltext.queryNodes(idx, $query) YIELD node AS n, score
            WHERE (CASE WHEN idx IN $valid_until_ft_indexes
                        THEN n.valid_until IS NULL
                        WHEN idx IN $no_expiry_ft_indexes
                        THEN true
                        ELSE n.superseded_by IS NULL END)
            {project_filter}
            {repo_filter}
            {scope_filter}
            RETURN n, score
            ORDER BY score DESC
            LIMIT 50
        }}
        RETURN apoc.map.removeKey(properties(n), 'embedding') AS props, labels(n)[0] AS node_type, score
    """
    params["ft_indexes"] = ["semantic_fulltext", "preference_fulltext", "fact_fulltext", "action_item_fulltext", "entity_fulltext"]
    params["valid_until_ft_indexes"] = ["preference_fulltext", "fact_fulltext"]
    params["no_expiry_ft_indexes"] = ["action_item_fulltext", "entity_fulltext"]

    all_rows: list = []
    try:
        with _pspan_sync("neo4j:fulltext_query"):
            all_rows.extend(client.run(unwind_cypher, params))
    except Exception:
        logger.warning("fulltext UNWIND query failed, returning empty results", exc_info=True)

    seen: dict[str, tuple] = {}
    for r in all_rows:
        nid = r["props"].get("id", "")
        if not nid or (nid in seen and seen[nid][1] >= r["score"]):
            continue
        seen[nid] = (r, r["score"])

    sorted_rows = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:50]
    return [_normalise_node(r["props"], r["node_type"]) for r, _ in sorted_rows]


# ---------------------------------------------------------------------------
# search_entities
# ---------------------------------------------------------------------------

def search_entities(
    client: Neo4jClient,
    query: str,
    allowed_scopes: list[str] | None = None,
) -> list[dict]:
    if not query.strip():
        return []
    sanitized_query = _sanitize_fulltext_query(query)
    if not sanitized_query:
        return []
    params: dict = {"query": sanitized_query}
    rows = client.run(f"""
        CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
        YIELD node AS e, score
        RETURN properties(e) AS props, score
        ORDER BY score DESC
        LIMIT 20
    """, params)
    return [r["props"] for r in rows]


def search_entities_batch(
    client: Neo4jClient,
    queries: list[str],
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Batch entity search. Returns one result set per query."""
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        q = query.strip()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(q)

    results: list[dict] = []
    for query in deduped[:50]:
        matches = search_entities(client, query, allowed_scopes=allowed_scopes)
        results.append({
            "query": query,
            "matches": matches,
            "matched": len(matches),
        })

    return {"results": results, "requested": len(queries), "executed": len(results)}


# ---------------------------------------------------------------------------
# list_topics
# ---------------------------------------------------------------------------

def list_topics(
    client: Neo4jClient,
    project: str | None = None,
    allowed_scopes: list[str] | None = None,
) -> list[dict]:
    params: dict = {}
    conditions: list[str] = []
    if project:
        conditions.append("t.project = $project")
        params["project"] = project
    if allowed_scopes is not None:
        conditions.append("t.source_scope IN $allowed_scopes")
        params["allowed_scopes"] = allowed_scopes
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    rows = client.run(f"""
        MATCH (t:Topic)
        {where_clause}
        RETURN properties(t) AS props
        ORDER BY t.label
    """, params)
    return [r["props"] for r in rows]




# ---------------------------------------------------------------------------
# query_context_vector — semantic vector search across all 4 index labels
# ---------------------------------------------------------------------------

_VECTOR_INDEXES = [
    "decision_vector",
    "constraint_vector",
    "insight_vector",
    "signal_vector",
    "preference_vector",
    "fact_vector",
    "entity_vector",
    "action_item_vector",
]

# Preference and Fact use valid_until for expiry; other semantic types use superseded_by.
# Entity and ActionItem have no expiry concept — superseded_by IS NULL evaluates true for them
# (NULL IS NULL = true in Cypher), so the ELSE branch in the vector query naturally passes them.
_INDEX_ACTIVE_FILTER = {
    "preference_vector": "node.valid_until IS NULL",
    "fact_vector":       "node.valid_until IS NULL",
}

# Derived: indexes whose active filter is valid_until (vs superseded_by for the rest)
_VALID_UNTIL_VECTOR_INDEXES = [
    idx for idx, filt in _INDEX_ACTIVE_FILTER.items() if "valid_until" in filt
]


def query_context_vector(
    client: Neo4jClient,
    query_embedding: list[float],
    project: str = "",
    limit: int = 8,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> dict:
    """Union vector search across all semantic node types including Entity and ActionItem.

    Queries all vector indexes in a single UNWIND query, deduplicates and sorts
    by score server-side, and returns the top `limit` nodes.

    Embedding vectors are excluded from the returned projection to avoid
    shipping 384-float payloads over the wire; _normalise_node also drops them
    but excluding here eliminates the transfer cost entirely.
    """
    limit = max(1, min(limit, 20))
    # Over-fetch per index so we have enough candidates after cross-index dedup
    k = min(limit * 2, 15)

    scope_filter = ""
    scope_params: dict = {}
    if allowed_scopes is not None:
        scope_filter = "AND (node:Entity OR node.source_scope IN $allowed_scopes)"
        scope_params["allowed_scopes"] = allowed_scopes

    # Map-projection `node {.*, embedding: null}` strips the 384-float embedding
    # server-side so it is never transferred.  Dedup + sort + limit also pushed
    # to the DB to minimise rows on the wire.
    cypher = f"""
        UNWIND $indexes AS idx
        CALL db.index.vector.queryNodes(idx, $k, $embedding) YIELD node, score
        WHERE ($project = '' OR node.project = $project)
          AND ($repo IS NULL OR node.repo IS NULL OR node.repo = $repo)
          AND (CASE WHEN idx IN $valid_until_indexes
                    THEN node.valid_until IS NULL
                    ELSE node.superseded_by IS NULL END)
          {scope_filter}
        WITH node, max(score) AS score
        RETURN node {{.*, embedding: null}} AS props, labels(node)[0] AS node_type, score
        ORDER BY score DESC
        LIMIT $limit
    """
    with _pspan_sync("neo4j:vector_index_query"):
        rows = client.run(cypher, {
            "indexes": _VECTOR_INDEXES,
            "valid_until_indexes": _VALID_UNTIL_VECTOR_INDEXES,
            "k": k,
            "embedding": query_embedding,
            "project": project,
            "repo": repo,
            "limit": limit,
            **scope_params,
        })

    top_nodes = [_normalise_node(row["props"], row["node_type"]) for row in rows]

    return {
        "nodes":         top_nodes,
        "total_matched": len(top_nodes),
        "returned":      len(top_nodes),
    }


# ---------------------------------------------------------------------------
# traverse — agent-driven single-node graph walk
# ---------------------------------------------------------------------------

def traverse(
    client: Neo4jClient,
    node_id: str,
    project: str = "",
    rel: str | None = None,
    direction: str = "both",
    limit: int = 20,
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Return neighbours of `node_id` reachable via `rel` (or all rels if None).

    direction: "out" | "in" | "both"
    """
    limit = max(1, min(limit, 50))

    # Validate rel against whitelist to prevent injection
    if rel is not None:
        rel_upper = rel.upper()
        if rel_upper not in ALL_RELS:
            return {
                "nodes": [], "edges": [],
                "error": f"Unknown relationship type '{rel}'. Valid types: {', '.join(sorted(ALL_RELS))}.",
            }
        rel_clause = rel_upper
    else:
        rel_clause = "|".join(ALL_RELS)

    params: dict = {"id": node_id, "project": project, "limit": limit}
    # Scope filter applied to BOTH seed and neighbour sides.
    # Without checking the seed, a caller who knows a forbidden node_id could
    # read its full properties via the RETURN clause even if no neighbours pass.
    outgoing_seed_scope_check = "1=1"
    incoming_seed_scope_check = "1=1"
    outgoing_neighbour_scope_filter = ""
    incoming_neighbour_scope_filter = ""
    if allowed_scopes is not None:
        outgoing_seed_scope_check = "(a:Entity OR a.source_scope IN $allowed_scopes)"
        incoming_seed_scope_check = "(b:Entity OR b.source_scope IN $allowed_scopes)"
        outgoing_neighbour_scope_filter = "AND (b:Entity OR b.source_scope IN $allowed_scopes)"
        incoming_neighbour_scope_filter = "AND (a:Entity OR a.source_scope IN $allowed_scopes)"
        relationship_scope_filter = "AND r.source_scope IN $allowed_scopes"
        params["allowed_scopes"] = allowed_scopes
    else:
        relationship_scope_filter = ""

    results_nodes: dict[str, dict] = {}
    results_edges: list[dict] = []

    project_filter = "(n.project = $project OR n.project IS NULL OR n.project = '')"

    if direction in ("out", "both"):
        rows = client.run(f"""
            MATCH (a {{id: $id}})-[r:{rel_clause}]->(b)
            WHERE {outgoing_seed_scope_check}
            AND {project_filter.replace('n.', 'b.')}
            {outgoing_neighbour_scope_filter}
            {relationship_scope_filter}
            RETURN
                a.id AS from_id, labels(a)[0] AS from_type, properties(a) AS from_props,
                b.id AS to_id,   labels(b)[0] AS to_type,   properties(b) AS to_props,
                type(r) AS rel_type, properties(r) AS rel_props
            LIMIT $limit
        """, params)
        for row in rows:
            _collect_row(row, results_nodes, results_edges)

    if direction in ("in", "both"):
        rows = client.run(f"""
            MATCH (a)-[r:{rel_clause}]->(b {{id: $id}})
            WHERE {incoming_seed_scope_check}
            AND {project_filter.replace('n.', 'a.')}
            {incoming_neighbour_scope_filter}
            {relationship_scope_filter}
            RETURN
                a.id AS from_id, labels(a)[0] AS from_type, properties(a) AS from_props,
                b.id AS to_id,   labels(b)[0] AS to_type,   properties(b) AS to_props,
                type(r) AS rel_type, properties(r) AS rel_props
            LIMIT $limit
        """, params)
        for row in rows:
            _collect_row(row, results_nodes, results_edges)

    nodes = list(results_nodes.values())[:limit]
    edges = _dedup_edges(results_edges)

    if not nodes:
        rel_desc = rel if rel else "any relationship"
        return {
            "nodes": [], "edges": [],
            "message": f"No {direction} neighbours found for '{node_id}' via {rel_desc}.",
        }

    return {"nodes": nodes, "edges": edges}


def traverse_batch(
    client: Neo4jClient,
    node_ids: list[str],
    project: str = "",
    rel: str | None = None,
    direction: str = "both",
    limit: int = 20,
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Batch traverse multiple node IDs with shared options."""
    results: list[dict] = []
    seen: set[str] = set()
    for node_id in node_ids:
        nid = node_id.strip()
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out = traverse(
            client=client,
            node_id=nid,
            project=project,
            rel=rel,
            direction=direction,
            limit=limit,
            allowed_scopes=allowed_scopes,
        )
        results.append({
            "node_id": nid,
            "nodes": out.get("nodes", []),
            "edges": out.get("edges", []),
            "message": out.get("message"),
            "error": out.get("error"),
        })

    return {"results": results, "requested": len(node_ids), "executed": len(results)}


def query_subgraph(
    client: Neo4jClient,
    project: str = "",
    query: str = "",
    max_hops: int = 2,
    budget: int = 60,
    node_types: list[str] | None = None,
    rel_types: list[str] | None = None,
    cursor: str | None = None,
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Retrieve a fully-expanded context subgraph for a topic.

    Returns ranked seeds + connected nodes + edges in one call.
    Replaces query_context + repeated traverse loop.
    """
    max_hops = max(1, min(max_hops, 3))
    budget = max(5, min(budget, 150))

    if rel_types is None:
        rel_types = list(HIGH_SIGNAL_RELS)
    else:
        rel_types = [r for r in rel_types if r in ALL_RELS]
        if not rel_types:
            rel_types = list(HIGH_SIGNAL_RELS)

    if node_types is None:
        node_types = list(EXPANDABLE_LABELS)
    else:
        node_types = [n for n in node_types if n in ALL_LABELS]
        if not node_types:
            node_types = list(EXPANDABLE_LABELS)

    label_filter = "+" + "|+".join(node_types)
    rel_filter = "|".join(rel_types)

    frontier_ids: list[str] = []
    seed_ids: list[str] = []
    seed_ranks: dict[str, int] = {}
    cursor_visited: set[str] = set()

    if cursor:
        try:
            cursor_data = json.loads(base64.b64decode(cursor).decode())
            cursor_visited = set(cursor_data.get("visited", []))
            frontier_ids = cursor_data.get("frontier", [])
        except Exception:
            pass

    seeds_found: list[dict] = []
    if frontier_ids:
        seed_ids = list(frontier_ids)
    else:
        seeds_found = search_memory(client, query, project, allowed_scopes=allowed_scopes)
        seed_ids = [n["id"] for n in seeds_found if n.get("id")]
        for rank, nid in enumerate(seed_ids):
            seed_ranks[nid] = rank

    merged_nodes: dict[str, dict] = {}
    merged_edges: list[dict] = []

    try:
        if allowed_scopes is None:
            rows = client.run(
                """
            UNWIND $seed_ids AS sid
            MATCH (seed {id: sid})
            WHERE $project = '' OR seed.project = $project OR seed.project IS NULL OR seed.project = ''
            CALL apoc.path.subgraphAll(seed, {
                maxLevel: $max_hops,
                limit: $budget,
                relationshipFilter: $rel_filter,
                labelFilter: $label_filter,
                bfs: true
            })
            YIELD nodes, relationships
            RETURN nodes, relationships
                """,
                {
                    "seed_ids": seed_ids,
                    "project": project,
                    "max_hops": max_hops,
                    "budget": budget,
                    "rel_filter": rel_filter,
                    "label_filter": label_filter,
                },
            )
        else:
            rows = client.run(
                f"""
                UNWIND $seed_ids AS sid
                MATCH (seed {{id: sid}})
                WHERE ($project = '' OR seed.project = $project OR seed.project IS NULL OR seed.project = '')
                  AND (seed:Entity OR seed.source_scope IN $allowed_scopes)
                MATCH p=(seed)-[:{rel_filter}*0..{max_hops}]-(reached)
                WHERE all(r IN relationships(p) WHERE r.source_scope IN $allowed_scopes)
                  AND all(n IN nodes(p) WHERE
                      (n:Entity OR n.source_scope IN $allowed_scopes)
                      AND ($project = '' OR n.project = $project OR n.project IS NULL OR n.project = '')
                      AND (n = seed OR any(label IN labels(n) WHERE label IN $node_types))
                  )
                RETURN nodes(p) AS nodes, relationships(p) AS relationships
                LIMIT $budget
                """,
                {
                    "seed_ids": seed_ids,
                    "project": project,
                    "allowed_scopes": allowed_scopes,
                    "node_types": node_types,
                    "budget": budget,
                },
            )
    except Exception:
        logger.warning("APOC subgraphAll query failed in query_subgraph, returning empty rows", exc_info=True)
        rows = []

    for row in rows:
        for neo4j_node in row.get("nodes") or []:
            props = dict(neo4j_node)
            node_type = list(neo4j_node.labels)[0] if hasattr(neo4j_node, "labels") else props.get("_type", "")
            normalised = _normalise_node(props, node_type)
            nid = normalised.get("id", "")
            if not nid or nid in cursor_visited:
                continue
            if allowed_scopes is not None:
                node_scope = normalised.get("source_scope")
                if node_type != "Entity" and node_scope not in allowed_scopes:
                    continue
            merged_nodes[nid] = normalised

        for rel in row.get("relationships") or []:
            try:
                from_id = rel.start_node["id"] if hasattr(rel, "start_node") else None
                to_id = rel.end_node["id"] if hasattr(rel, "end_node") else None
                rel_type = rel.type if hasattr(rel, "type") else None
                rel_props = dict(rel)
                if from_id and to_id and rel_type:
                    if (
                        allowed_scopes is not None
                        and rel_props.get("source_scope") not in allowed_scopes
                    ):
                        continue
                    merged_edges.append({
                        "from": from_id,
                        "rel": rel_type,
                        "to": to_id,
                        "created_at": rel_props.get("created_at"),
                    })
            except Exception:
                continue

    merged_edges = _dedup_edges_simple(merged_edges)

    # BFS from all seeds simultaneously to get real hop distances.
    # Uses the edges already returned by APOC so no extra round-trip.
    # Edges are treated as undirected for distance purposes.
    adjacency: dict[str, set[str]] = {nid: set() for nid in merged_nodes}
    for _e in merged_edges:
        _a, _b = _e.get("from"), _e.get("to")
        if _a in adjacency and _b in adjacency:
            adjacency[_a].add(_b)
            adjacency[_b].add(_a)

    hop_distances: dict[str, int] = {}
    _bfs_queue: deque[str] = deque()
    for _sid in seed_ids:
        if _sid in merged_nodes:
            hop_distances[_sid] = 0
            _bfs_queue.append(_sid)
    while _bfs_queue:
        _cur = _bfs_queue.popleft()
        _cur_dist = hop_distances[_cur]
        for _nb in adjacency.get(_cur, ()):
            if _nb not in hop_distances:
                hop_distances[_nb] = _cur_dist + 1
                _bfs_queue.append(_nb)
    # Nodes unreachable via collected edges (disconnected components after
    # scope/label filtering) fall back to max_hops so they score lowest.
    for _nid in merged_nodes:
        if _nid not in hop_distances:
            hop_distances[_nid] = max_hops

    now = datetime.now(tz=timezone.utc)
    total_seeds = max(len(seed_ids), 1)

    def _score_node(nid: str, node: dict) -> float:
        hop_distance = hop_distances.get(nid, max_hops)
        created_at = node.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    ca = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    ca = created_at
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=timezone.utc)
                days_old = (now - ca).days
            except Exception:
                days_old = 180
        else:
            days_old = 180
        recency_decay = 1.0 / (1.0 + days_old / 180)
        rank = seed_ranks.get(nid)
        text_score = 1.0 - (rank / total_seeds) if rank is not None else 0.5
        return round(text_score * (0.7 ** hop_distance) * recency_decay, 4)

    scored = [(nid, node, _score_node(nid, node)) for nid, node in merged_nodes.items()]
    scored.sort(key=lambda x: x[2], reverse=True)

    truncated = len(scored) > budget
    scored = scored[:budget]

    current_page_ids: set[str] = {nid for nid, _, _ in scored}
    visited_ids: set[str] = set(current_page_ids)
    visited_ids |= cursor_visited

    nodes_out = []
    for nid, node, score in scored:
        node_copy = dict(node)
        node_copy["score"] = score
        nodes_out.append(node_copy)

    def _edge_rank_key(e: dict):
        signal = 0 if e.get("rel") in HIGH_SIGNAL_RELS else 1
        raw_ts = e.get("created_at")
        try:
            ts = datetime.fromisoformat(str(raw_ts)).timestamp() if raw_ts is not None else 0.0
        except Exception:
            ts = 0.0
        return (signal, -ts)

    edges_in_set = sorted(
        (e for e in merged_edges if e.get("from") in current_page_ids and e.get("to") in current_page_ids),
        key=_edge_rank_key,
    )

    seed_nodes_out = [
        {"id": nid, "score": score, "type": node.get("_type", "")}
        for nid, node, score in scored
        if nid in seed_ids
    ]

    boundary_params: dict = {"visited_ids": list(current_page_ids), "project": project}
    boundary_scope_filter = ""
    if allowed_scopes is not None:
        boundary_scope_filter = """
        AND (a:Entity OR a.source_scope IN $allowed_scopes)
        AND (b:Entity OR b.source_scope IN $allowed_scopes)
        AND r.source_scope IN $allowed_scopes
        """
        boundary_params["allowed_scopes"] = allowed_scopes
    boundary_rows = client.run(
        f"""
        MATCH (a)-[r]->(b)
        WHERE ((a.id IN $visited_ids AND NOT b.id IN $visited_ids)
            OR (b.id IN $visited_ids AND NOT a.id IN $visited_ids))
        AND ($project = '' OR a.project = $project OR a.project IS NULL OR a.project = '')
        AND ($project = '' OR b.project = $project OR b.project IS NULL OR b.project = '')
        {boundary_scope_filter}
        RETURN a.id AS from_id, type(r) AS rel_type, b.id AS to_id
        LIMIT 50
        """,
        boundary_params,
    )
    boundary_edges = [
        {"from": r["from_id"], "rel": r["rel_type"], "to": r["to_id"], "truncated": True}
        for r in boundary_rows
    ]

    cursor_str: str | None = None
    if truncated:
        outermost = [nid for nid, node, _ in scored if nid not in seed_ids]
        cursor_data = {
            "visited": list(visited_ids),
            "frontier": outermost,
            "qhash": hashlib.sha256(query.encode()).hexdigest()[:8],
        }
        cursor_str = base64.b64encode(json.dumps(cursor_data).encode()).decode()

    return {
        "seeds": seed_nodes_out,
        "nodes": nodes_out,
        "edges": edges_in_set,
        "boundary_edges": boundary_edges,
        "total_matched": len(seeds_found),
        "returned": len(nodes_out),
        "truncated": truncated,
        "cursor": cursor_str,
    }


# ---------------------------------------------------------------------------
# hybrid_search — parallel PgVector + Neo4j search with RRF merge
# ---------------------------------------------------------------------------

async def _pgvector_chunk_search(
    pool,
    embedding: list[float],
    org_id: str,
    limit: int,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> list[dict]:
    """Cosine-similarity search over document_chunks for a given org."""
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    async with _pspan("pg:vector_chunk_search"):
        rows = await pool.fetch(
            """
            SELECT
                c.doc_id,
                c.chunk_index,
                c.char_offset,
                c.char_length,
                1 - (c.embedding <=> $1::vector) AS similarity
            FROM document_chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE d.org_id = $2
              AND ($3::text[] IS NULL OR d.source_scope = ANY($3::text[]))
              AND ($4::text IS NULL OR d.repo IS NULL OR d.repo = $4)
            ORDER BY c.embedding <=> $1::vector
            LIMIT $5
            """,
            vec_str,
            org_id,
            allowed_scopes,
            repo,
            limit,
        )
    return [dict(r) for r in rows]


def _neo4j_hybrid_search(
    client: Neo4jClient,
    query: str,
    embedding: list[float],
    project: str = "",
    limit: int = 20,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> list[dict]:
    """Combine Neo4j vector search + fulltext search with internal RRF.

    Vector and fulltext queries run in parallel threads.  Each thread receives
    a copy of the current contextvars context so that _pspan_sync spans created
    inside query_context_vector / search_memory are correctly nested under the
    parent "_neo4j_hybrid_search" span in the profile tree.
    """
    with _pspan_sync("_neo4j_hybrid_search"):
        def _run_vector() -> dict:
            with _pspan_sync("query_context_vector"):
                return query_context_vector(
                    client, embedding, project, limit, allowed_scopes, repo
                )

        def _run_text() -> list[dict]:
            with _pspan_sync("search_memory"):
                return search_memory(client, query, project, allowed_scopes, repo)

        # Capture parent span context before spawning threads; each worker runs
        # inside a copy so child spans attach to the correct parent.
        ctx_vec = contextvars.copy_context()
        ctx_txt = contextvars.copy_context()

        with ThreadPoolExecutor(max_workers=2) as executor:
            vec_future = executor.submit(
                ctx_vec.run,
                _run_vector,
            )
            txt_future = executor.submit(
                ctx_txt.run,
                _run_text,
            )
            vector_hits_result = vec_future.result()
            text_hits = txt_future.result()

        vector_hits = vector_hits_result.get("nodes", [])

        # Internal RRF (k=60, equal weights) to merge the two Neo4j result sets
        with _pspan_sync("rrf_merge"):
            scores: dict[str, float] = defaultdict(float)
            all_nodes: dict[str, dict] = {}
            k = 60

            for rank, node in enumerate(vector_hits):
                nid = node.get("id", "")
                if nid:
                    scores[nid] += 1.0 / (k + rank + 1)
                    all_nodes[nid] = node

            for rank, node in enumerate(text_hits):
                nid = node.get("id", "")
                if nid:
                    scores[nid] += 1.0 / (k + rank + 1)
                    all_nodes[nid] = node

            result = sorted(all_nodes.values(), key=lambda n: scores.get(n.get("id", ""), 0.0), reverse=True)
            for node in result:
                nid = node.get("id", "")
                node["rrf_score"] = round(scores.get(nid, 0.0), 6)

    return result


def _fetch_document_nodes(
    client: Neo4jClient,
    doc_ids: list[str],
    project: str = "",
    repo: str | None = None,
) -> list[dict]:
    """Fetch fully-hydrated Document nodes from Neo4j for the given doc_ids."""
    if not doc_ids:
        return []
    rows = client.run(
        """
        MATCH (d:Document)
        WHERE d.id IN $doc_ids
          AND ($project = '' OR d.project = $project)
          AND ($repo IS NULL OR d.repo IS NULL OR d.repo = $repo)
        RETURN properties(d) AS props, labels(d)[0] AS node_type
        """,
        {"doc_ids": doc_ids, "project": project, "repo": repo},
    )
    return [_normalise_node(r["props"], r["node_type"]) for r in rows]


def _dedup_against_document_nodes(neo4j_nodes: list[dict], document_node_ids: set[str]) -> list[dict]:
    """Remove Neo4j nodes whose source_doc_id is already in document_nodes."""
    out: list[dict] = []
    for node in neo4j_nodes:
        src = node.get("source_doc_id", "")
        if src and src in document_node_ids:
            continue
        out.append(node)
    return out


def _graph_fanout(
    client: Neo4jClient,
    doc_nodes: list[dict],
    project: str = "",
    max_hops: int = 1,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> list[dict]:
    """Return semantic nodes reachable from the matched Document nodes.

    hop 1: direct EXTRACTED_FROM — nodes extracted from these documents.
    hop 2+: additional semantic-graph hops from those nodes via high-signal rels,
            using APOC variable-depth expansion (APOC must be installed).
    """
    doc_ids = [d["id"] for d in doc_nodes]
    if not doc_ids:
        return []

    hop1_params: dict = {"doc_ids": doc_ids, "project": project, "repo": repo}
    hop1_scope_filter = ""
    if allowed_scopes is not None:
        hop1_scope_filter = """
          AND (n:Entity OR n.source_scope IN $allowed_scopes)
          AND d.source_scope IN $allowed_scopes
          AND extracted.source_scope IN $allowed_scopes
        """
        hop1_params["allowed_scopes"] = allowed_scopes

    # Hop 1 — always: semantic nodes extracted from matched documents
    rows = client.run(
        f"""
        MATCH (n)-[extracted:EXTRACTED_FROM]->(d:Document)
        WHERE d.id IN $doc_ids
          AND ($project = '' OR n.project = $project OR n.project IS NULL)
          AND ($repo IS NULL OR n.repo IS NULL OR n.repo = $repo)
          AND ($repo IS NULL OR d.repo IS NULL OR d.repo = $repo)
          {hop1_scope_filter}
        RETURN DISTINCT properties(n) AS props, labels(n)[0] AS node_type
        """,
        hop1_params,
    )
    hop1_nodes = {
        r["props"].get("id"): _normalise_node(r["props"], r["node_type"])
        for r in rows
        if r["props"].get("id")
    }

    if max_hops == 1 or not hop1_nodes:
        return list(hop1_nodes.values())

    # Hop 2+ — APOC variable-depth expansion from hop-1 nodes
    rel_filter = "|".join(HIGH_SIGNAL_RELS)
    label_filter = "+" + "|+".join(EXPANDABLE_LABELS)
    seed_ids = list(hop1_nodes.keys())

    hop2_scope_filter = ""
    hop2_scope_params: dict = {}
    if allowed_scopes is not None:
        hop2_scope_filter = "AND (n:Entity OR n.source_scope IN $allowed_scopes)"
        hop2_scope_params["allowed_scopes"] = allowed_scopes

    expanded: dict[str, dict] = dict(hop1_nodes)
    try:
        if allowed_scopes is None:
            apoc_rows = client.run(
                f"""
            UNWIND $seed_ids AS sid
            MATCH (seed {{id: sid}})
            CALL apoc.path.subgraphAll(seed, {{
                maxLevel: $max_hops,
                relationshipFilter: $rel_filter,
                labelFilter: $label_filter,
                bfs: true
            }})
            YIELD nodes
            UNWIND nodes AS n
            WHERE ($project = '' OR n.project = $project OR n.project IS NULL)
              AND ($repo IS NULL OR n.repo IS NULL OR n.repo = $repo)
            {hop2_scope_filter}
            RETURN DISTINCT properties(n) AS props, labels(n)[0] AS node_type
                """,
                {
                    "seed_ids": seed_ids,
                    "max_hops": max_hops - 1,
                    "rel_filter": rel_filter,
                    "label_filter": label_filter,
                    "project": project,
                    "repo": repo,
                    **hop2_scope_params,
                },
            )
        else:
            apoc_rows = client.run(
                f"""
                UNWIND $seed_ids AS sid
                MATCH (seed {{id: sid}})
                MATCH p=(seed)-[:{rel_filter}*1..{max_hops - 1}]-(n)
                WHERE all(r IN relationships(p) WHERE r.source_scope IN $allowed_scopes)
                  AND all(path_node IN nodes(p) WHERE
                      (path_node:Entity OR path_node.source_scope IN $allowed_scopes)
                      AND ($project = '' OR path_node.project = $project OR path_node.project IS NULL)
                      AND ($repo IS NULL OR path_node.repo IS NULL OR path_node.repo = $repo)
                  )
                RETURN DISTINCT properties(n) AS props, labels(n)[0] AS node_type
                """,
                {
                    "seed_ids": seed_ids,
                    "allowed_scopes": allowed_scopes,
                    "project": project,
                    "repo": repo,
                },
            )
    except Exception:
        logger.warning("APOC subgraphAll query failed in _graph_fanout, skipping hop 2+ expansion", exc_info=True)
        apoc_rows = []
    for row in apoc_rows:
        nid = row["props"].get("id")
        if nid and nid not in expanded:
            expanded[nid] = _normalise_node(row["props"], row["node_type"])

    return list(expanded.values())


def _rrf_merge(
    document_nodes: list[dict],
    fanout: list[dict],
    similar_nodes: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge document_nodes, fanout nodes, and similar_nodes via RRF.

    document_nodes weight = 1.5x; fanout and similar_nodes weight = 1.0x.
    """
    scores: dict[str, float] = defaultdict(float)
    all_nodes: dict[str, dict] = {}

    for rank, node in enumerate(document_nodes):
        nid = node.get("id", "")
        if nid:
            scores[nid] += 1.5 / (k + rank + 1)
            all_nodes[nid] = node

    for rank, node in enumerate(fanout):
        nid = node.get("id", "")
        if nid:
            scores[nid] += 1.0 / (k + rank + 1)
            all_nodes[nid] = node

    for rank, node in enumerate(similar_nodes):
        nid = node.get("id", "")
        if nid:
            scores[nid] += 1.0 / (k + rank + 1)
            all_nodes[nid] = node

    merged = sorted(all_nodes.values(), key=lambda n: scores.get(n.get("id", ""), 0.0), reverse=True)
    for node in merged:
        nid = node.get("id", "")
        node["rrf_score"] = round(scores.get(nid, 0.0), 6)
    return merged


_DOC_SELECT = "SELECT id, title, s3_key, source_scope, source_provider, current_version, repo, doc_type FROM documents WHERE id = $1"
_DOC_SELECT_BY_S3 = "SELECT id, title, s3_key, source_scope, source_provider, current_version, repo, doc_type FROM documents WHERE s3_key = $1"

# {org_id}/{provider}/raw/{source_ref}/v{version}.json  →  doc:{provider}:{source_ref}
_S3_KEY_DOC_ID_RE = re.compile(
    r"^(?P<org_id>[^/]+)/(?P<provider>[^/]+)/raw/(?P<source_ref>[^/]+)/v\d+\.json$"
)
_S3_KEY_DOC_ID_PARTIAL_RE = re.compile(
    r"^(?P<provider>[^/]+)/raw/(?P<source_ref>[^/]+)/v\d+\.json$"
)


def _strip_s3_bucket_prefix(key: str) -> str:
    """Remove leading bucket name when callers pass a full object URI path."""
    key = key.lstrip("/")
    try:
        from base.tools.s3 import _bucket

        bucket = _bucket()
        prefix = f"{bucket}/"
        if key.startswith(prefix):
            return key[len(prefix):]
    except RuntimeError:
        pass
    return key


def _doc_id_from_s3_key(s3_key: str) -> str | None:
    """Map a raw-object S3 key to documents.id when the path matches ingestion layout."""
    match = _S3_KEY_DOC_ID_RE.match(s3_key) or _S3_KEY_DOC_ID_PARTIAL_RE.match(s3_key)
    if match is None:
        return None
    return f"doc:{match.group('provider')}:{match.group('source_ref')}"


async def _fetch_document_row(pool, raw_doc_id: str):
    """Resolve a document row from canonical id or common s3_key-style aliases."""
    if raw_doc_id.startswith("doc:"):
        row = await pool.fetchrow(_DOC_SELECT, raw_doc_id)
        if row is not None:
            return row

    s3_key = _strip_s3_bucket_prefix(raw_doc_id)
    parsed_id = _doc_id_from_s3_key(s3_key)
    if parsed_id:
        row = await pool.fetchrow(_DOC_SELECT, parsed_id)
        if row is not None:
            return row

    row = await pool.fetchrow(_DOC_SELECT_BY_S3, s3_key)
    if row is not None:
        return row

    if not raw_doc_id.startswith("doc:"):
        return await pool.fetchrow(_DOC_SELECT, raw_doc_id)
    return None


def _optional_row_value(row, key: str):
    try:
        return row[key]
    except (KeyError, TypeError):
        return None


async def get_document_content(
    pool,
    doc_id: str,
    chunk_index: int | None = None,
    employee_id: str | None = None,
) -> dict:
    """Fetch document content from S3.

    Looks up documents.s3_key (and optionally document_chunks char offsets)
    then reads the raw content from S3.  If chunk_index is provided, slices
    the full text to return just that chunk.

    Accepts canonical document ids (doc:{provider}:{source_ref}) or s3_key-style
    paths copied from Neo4j (with or without bucket prefix).
    """
    from base.tools.s3 import read_document_content as _s3_read

    row = await _fetch_document_row(pool, doc_id)
    if row is None:
        return {"error": "Document not found", "doc_id": doc_id}

    resolved_doc_id: str = row["id"]

    # Access check: service accounts bypass. Portal documents are org-visible.
    if employee_id is not None and row.get("source_provider") != "portal":
        member = await pool.fetchval(
            """
            SELECT 1 FROM channel_members
            WHERE channel_id = $1 AND employee_id = $2
            LIMIT 1
            """,
            row["source_scope"],
            employee_id,
        )
        if member is None:
            return {"error": "Access denied", "doc_id": resolved_doc_id}

    s3_key: str = row["s3_key"] or ""
    title: str = row["title"] or ""

    if not s3_key:
        return {"error": f"Document '{resolved_doc_id}' has no s3_key"}

    chunk_row = None
    if chunk_index is not None:
        chunk_row = await pool.fetchrow(
            """
            SELECT chunk_index, char_offset, char_length
            FROM document_chunks
            WHERE doc_id = $1 AND chunk_index = $2
            """,
            resolved_doc_id,
            chunk_index,
        )
        if chunk_row is None:
            return {
                "error": "Chunk not found",
                "doc_id": resolved_doc_id,
                "chunk_index": chunk_index,
            }

        # Chunk offsets may refer to provider-normalized text (for example a
        # rendered repository profile), not the raw provider JSON. Durable
        # index payloads retain the exact chunk text used for embeddings.
        index_payload_key = await pool.fetchval(
            """
            SELECT payload_s3_key
            FROM document_index_jobs
            WHERE doc_id = $1
              AND document_version = $2
              AND vector_state = 'completed'
            ORDER BY created_at
            LIMIT 1
            """,
            resolved_doc_id,
            row["current_version"],
        )
        if index_payload_key:
            try:
                from base.tools.s3 import read_json_object as _s3_read_json

                index_payload = await _s3_read_json(str(index_payload_key))
                chunks = index_payload.get("chunks")
                if isinstance(chunks, list) and 0 <= chunk_index < len(chunks):
                    indexed_chunk = chunks[chunk_index]
                    if isinstance(indexed_chunk, dict) and isinstance(
                        indexed_chunk.get("text"), str
                    ):
                        return {
                            "doc_id": resolved_doc_id,
                            "title": title,
                            "repo": _optional_row_value(row, "repo"),
                            "doc_type": _optional_row_value(row, "doc_type"),
                            "chunk": {
                                "index": chunk_row["chunk_index"],
                                "char_offset": chunk_row["char_offset"],
                                "char_length": chunk_row["char_length"],
                                "text": indexed_chunk["text"],
                            },
                        }
            except RuntimeError:
                _log.warning(
                    "Could not read durable index payload for doc=%s; "
                    "falling back to raw source offsets",
                    resolved_doc_id,
                )

    try:
        content = await _s3_read(s3_key)
    except RuntimeError as exc:
        return {"error": str(exc), "doc_id": resolved_doc_id}

    if chunk_index is not None:
        if chunk_row is None:
            return {
                "error": "Chunk metadata missing",
                "doc_id": resolved_doc_id,
                "chunk_index": chunk_index,
            }
        offset: int = chunk_row["char_offset"]
        length: int = chunk_row["char_length"]
        chunk_text = content[offset: offset + length]
        return {
            "doc_id": resolved_doc_id,
            "title":  title,
            "repo": _optional_row_value(row, "repo"),
            "doc_type": _optional_row_value(row, "doc_type"),
            "chunk": {
                "index":       chunk_row["chunk_index"],
                "char_offset": offset,
                "char_length": length,
                "text":        chunk_text,
            },
        }

    return {
        "doc_id":  resolved_doc_id,
        "title":   title,
        "repo": _optional_row_value(row, "repo"),
        "doc_type": _optional_row_value(row, "doc_type"),
        "content": content,
    }


async def hybrid_search(
    client: Neo4jClient,
    pool,
    query: str,
    chunk_embedding: list[float],
    neo4j_embedding: list[float],
    project: str = "",
    org_id: str = "",
    chunk_limit: int = 10,
    node_limit: int = 20,
    fanout_hops: int = 1,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> dict:
    """
    Parallel hybrid search across PgVector (document chunks) and Neo4j (semantic nodes).

    Uses two different embeddings matched to each store's index:
    - chunk_embedding: bge-large-en-v1.5 (1024-dim) for PgVector
    - neo4j_embedding: all-MiniLM-L6-v2 (384-dim) for Neo4j vector indexes

    Returns:
        {
            "document_nodes": [Fully-hydrated Document nodes from Neo4j matching PgVector chunk hits],
            "fanout":         [Semantic nodes extracted from document_nodes documents],
            "similar_nodes":  [Neo4j semantic nodes not already covered by document_nodes],
            "merged":         [All nodes sorted by RRF score],
        }
    """
    chunk_task = asyncio.create_task(
        _pgvector_chunk_search(
            pool,
            chunk_embedding,
            org_id,
            chunk_limit,
            allowed_scopes=allowed_scopes,
            repo=repo,
        )
    )
    neo4j_task = asyncio.create_task(
        asyncio.to_thread(
            _neo4j_hybrid_search,
            client,
            query,
            neo4j_embedding,
            project,
            node_limit,
            allowed_scopes,
            repo,
        )
    )

    chunk_rows, neo4j_nodes = await asyncio.gather(chunk_task, neo4j_task)

    doc_ids = list({row["doc_id"] for row in chunk_rows if row.get("doc_id")})
    # Wrap the blocking Neo4j call so the event loop stays free while it runs.
    document_nodes = await asyncio.to_thread(
        _fetch_document_nodes, client, doc_ids, project, repo
    )
    document_node_ids = {n["id"] for n in document_nodes}

    # Launch the blocking _graph_fanout in a thread, then compute similar_nodes
    # (pure Python) concurrently while the thread is running.
    fanout_task = asyncio.create_task(
        asyncio.to_thread(
            _graph_fanout,
            client,
            document_nodes,
            project,
            fanout_hops,
            allowed_scopes,
            repo,
        )
    )
    similar_nodes = _dedup_against_document_nodes(neo4j_nodes, document_node_ids)
    fanout = await fanout_task

    merged = _rrf_merge(document_nodes, fanout, similar_nodes)

    return {
        "document_nodes": document_nodes,
        "fanout":         fanout,
        "similar_nodes":  similar_nodes,
        "merged":         merged,
        "chunk_rows":     chunk_rows,
    }


async def search(
    client: Neo4jClient,
    pool,
    query: str,
    chunk_embedding: list[float],
    neo4j_embedding: list[float],
    project: str = "",
    org_id: str = "",
    limit: int = 20,
    chunk_limit: int = 10,
    node_limit: int = 20,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> dict:
    """Unified hybrid search with condensed, agent-facing results."""
    limit = max(1, min(limit, 50))
    raw = await hybrid_search(
        client=client,
        pool=pool,
        query=query,
        chunk_embedding=chunk_embedding,
        neo4j_embedding=neo4j_embedding,
        project=project,
        org_id=org_id,
        chunk_limit=chunk_limit,
        node_limit=node_limit,
        fanout_hops=1,
        allowed_scopes=allowed_scopes,
        repo=repo,
    )
    merged = raw.get("merged", [])
    results: list[dict] = []
    for node in merged[:limit]:
        node_type = str(node.get("_type") or node.get("type") or "Unknown")
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        label = _node_label(node, node_type)
        snippet = _node_snippet(node, node_type)
        score_val = node.get("rrf_score", node.get("score", 0.0))
        try:
            score = round(float(score_val), 6)
        except Exception:
            score = 0.0
        entry = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "score": score,
            "snippet": snippet,
        }
        created_at = node.get("created_at")
        if created_at:
            entry["created_at"] = created_at
        source_doc = node.get("source_doc") or node.get("source_doc_id")
        if source_doc:
            entry["source_doc"] = source_doc
        for metadata_field in ("repo", "doc_type"):
            metadata_value = node.get(metadata_field)
            if metadata_value:
                entry[metadata_field] = metadata_value
        results.append(entry)
    return {
        "results": results,
        "returned": len(results),
        "total_matched": len(merged),
    }


async def _recall_documents(
    pool,
    chunk_rows: list[dict],
    excerpt_cap: int,
    detail: str,
    employee_id: str | None,
) -> list[dict]:
    """Build the documents[] section of recall directly from Postgres chunk rows.

    - Ranks chunks by pgvector similarity (already ordered by _pgvector_chunk_search).
    - Fetches doc metadata (title, s3_key, source_scope) in one batched query.
    - Access-checks once: loads the employee's allowed scopes, then filters docs.
    - Reads S3 and slices char_offset/char_length from chunk_rows (no second PG
      round-trip for offset data).
    """
    from base.tools.s3 import read_document_content as _s3_read

    if not chunk_rows:
        return []

    # One query for all document metadata
    distinct_doc_ids = list({cr["doc_id"] for cr in chunk_rows if cr.get("doc_id")})
    async with _pspan("pg:doc_metadata_fetch"):
        doc_meta_rows = await pool.fetch(
            "SELECT id, title, s3_key, source_scope, repo, doc_type FROM documents WHERE id = ANY($1)",
            distinct_doc_ids,
        )
    doc_meta: dict[str, dict] = {r["id"]: dict(r) for r in doc_meta_rows}

    # Access check: load all scopes the employee belongs to (one query, not N)
    allowed_scopes_for_employee: set[str] | None = None
    if employee_id is not None:
        async with _pspan("pg:employee_scope_fetch"):
            scope_rows = await pool.fetch(
                "SELECT channel_id FROM channel_members WHERE employee_id = $1",
                employee_id,
            )
        allowed_scopes_for_employee = {r["channel_id"] for r in scope_rows}

    # Build per-chunk document entries (preserving similarity order)
    documents: list[dict] = []
    seen_chunks: set[tuple[str, int]] = set()
    seen_title_chunks: set[tuple[str, int]] = set()
    for cr in chunk_rows:
        doc_id = cr.get("doc_id")
        chunk_index = cr.get("chunk_index")
        if not doc_id or chunk_index is None:
            continue
        key = (doc_id, chunk_index)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)

        meta = doc_meta.get(doc_id)
        if not meta:
            continue

        # Access gate
        if allowed_scopes_for_employee is not None:
            if meta.get("source_scope") not in allowed_scopes_for_employee:
                continue

        # Title-level dedup: same (title, chunk_index) across different doc_ids means
        # the same source content was ingested more than once (e.g. archived snapshots).
        # pgvector orders rows by similarity desc, so the first occurrence is the best one.
        title = meta.get("title") or ""
        title_key = (title, chunk_index)
        if title_key in seen_title_chunks:
            continue
        seen_title_chunks.add(title_key)

        entry: dict = {
            "doc_id": doc_id,
            "title": title,
            "chunk_index": chunk_index,
            "similarity": round(float(cr.get("similarity") or 0.0), 6),
        }
        for metadata_field in ("repo", "doc_type"):
            metadata_value = meta.get(metadata_field)
            if metadata_value:
                entry[metadata_field] = metadata_value
        documents.append(entry)

    # Fetch S3 excerpts in parallel for top excerpt_cap docs (detail=full only)
    if detail == "full" and documents:
        excerpt_slice = documents[:excerpt_cap]

        async def _fetch_excerpt(entry: dict, cr_map: dict) -> str | None:
            doc_id = entry["doc_id"]
            chunk_key = (doc_id, entry["chunk_index"])
            cr = cr_map.get(chunk_key)
            meta = doc_meta.get(doc_id)
            if not cr or not meta or not meta.get("s3_key"):
                return None
            try:
                async with _pspan(f"s3:read({entry['doc_id'][:8]}…)"):
                    content = await _s3_read(meta["s3_key"])
            except RuntimeError:
                return None
            offset: int = cr["char_offset"]
            length: int = cr["char_length"]
            return content[offset: offset + length]

        cr_map = {(cr["doc_id"], cr["chunk_index"]): cr for cr in chunk_rows}
        async with _pspan(f"s3:excerpts_parallel[{len(excerpt_slice)}]"):
            excerpt_results = await asyncio.gather(
                *[_fetch_excerpt(e, cr_map) for e in excerpt_slice],
                return_exceptions=True,
            )
        for entry, result in zip(excerpt_slice, excerpt_results):
            if isinstance(result, str) and result:
                entry["source_excerpt"] = result

    return documents


async def _build_nodes_with_neighbors(
    client: "Neo4jClient",
    neo4j_nodes: list,
    limit: int,
    project: str,
    allowed_scopes: list[str] | None,
    repo: str | None = None,
) -> list[dict]:
    """Assemble the nodes[] payload: shape raw Neo4j hits and attach 1-hop neighbors."""
    top_nodes = neo4j_nodes[:limit]
    nodes: list[dict] = []
    for node in top_nodes:
        node_type = str(node.get("_type") or node.get("type") or "Unknown")
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        score_val = node.get("rrf_score", node.get("score", 0.0))
        try:
            score = round(float(score_val), 6)
        except Exception:
            score = 0.0
        entry: dict = {
            "id": node_id,
            "type": node_type,
            "label": _node_label(node, node_type),
            "score": score,
            "snippet": _node_snippet(node, node_type),
        }
        source_doc = node.get("source_doc") or node.get("source_doc_id")
        if source_doc:
            entry["source_doc"] = source_doc
        for metadata_field in ("repo", "doc_type"):
            metadata_value = node.get(metadata_field)
            if metadata_value:
                entry[metadata_field] = metadata_value
        created_at = node.get("created_at")
        if created_at:
            entry["created_at"] = created_at
        nodes.append(entry)

    seed_ids = [n["id"] for n in nodes]
    if not seed_ids:
        for node_entry in nodes:
            node_entry["neighbors"] = []
        return nodes

    try:
        async with _pspan("_expand_graph[1-hop neighbors]"):
            nbr_nodes, nbr_edges = await asyncio.to_thread(
                _expand_graph, client, seed_ids, project, 1, allowed_scopes, repo
            )
        seed_id_set = set(seed_ids)
        neighbors_map: dict[str, list[dict]] = {}
        for edge in nbr_edges:
            from_id = edge.get("from")
            to_id = edge.get("to")
            if from_id in seed_id_set and to_id and to_id != from_id:
                nd = nbr_nodes.get(to_id)
                if nd:
                    nbr_entry: dict = {
                        "id": to_id,
                        "type": nd["type"],
                        "label": _node_label(nd["props"], nd["type"]),
                        "rel": edge.get("rel"),
                    }
                    nbr_src = nd["props"].get("source_doc") or nd["props"].get("source_doc_id")
                    if nbr_src:
                        nbr_entry["source_doc"] = nbr_src
                    neighbors_map.setdefault(str(from_id), []).append(nbr_entry)
            elif to_id in seed_id_set and from_id and from_id != to_id:
                nd = nbr_nodes.get(from_id)
                if nd:
                    nbr_entry = {
                        "id": from_id,
                        "type": nd["type"],
                        "label": _node_label(nd["props"], nd["type"]),
                        "rel": edge.get("rel"),
                    }
                    nbr_src = nd["props"].get("source_doc") or nd["props"].get("source_doc_id")
                    if nbr_src:
                        nbr_entry["source_doc"] = nbr_src
                    neighbors_map.setdefault(str(to_id), []).append(nbr_entry)
        for node_entry in nodes:
            node_entry["neighbors"] = neighbors_map.get(node_entry["id"], [])
    except Exception:
        logger.warning("recall: _expand_graph neighbors failed, continuing without neighbors", exc_info=True)
        for node_entry in nodes:
            node_entry.setdefault("neighbors", [])

    return nodes


async def recall(
    client: Neo4jClient,
    pool,
    query: str,
    chunk_embedding: list[float],
    neo4j_embedding: list[float],
    project: str = "",
    org_id: str = "",
    limit: int = 10,
    chunk_limit: int = 10,
    node_limit: int = 20,
    detail: str = "full",
    excerpt_cap: int = 5,
    allowed_scopes: list[str] | None = None,
    employee_id: str | None = None,
    repo: str | None = None,
) -> dict:
    """Composite one-shot recall: document chunks + semantic graph nodes.

    Returns two independent sections:

    documents[] — ranked by pgvector similarity, built entirely from Postgres +
        S3.  Each entry: doc_id, title, chunk_index, similarity, source_excerpt
        (detail=full).  No Neo4j involvement.

    nodes[] — Neo4j semantic hits (Claim, Entity, Fact, …) ranked by RRF, each
        with snippet, source_doc pointer, and 1-hop neighbors.

    detail=full  (default): includes source_excerpt in documents[].
    detail=light : skips S3 reads.
    """
    limit = max(1, min(limit, 50))
    chunk_limit = max(1, min(chunk_limit, 50))
    excerpt_cap = max(1, min(excerpt_cap, 20))

    # Run Postgres chunk search and Neo4j semantic search in parallel.
    # Each pipeline chains its own downstream work so neither blocks on the other.

    async def _pipeline_documents() -> list:
        async with _pspan("_pipeline_documents"):
            async with _pspan("_pgvector_chunk_search"):
                chunk_rows = await _pgvector_chunk_search(
                    pool,
                    chunk_embedding,
                    org_id,
                    3,
                    allowed_scopes=allowed_scopes,
                    repo=repo,
                )
            logger.info(f"[DEBUG] chunk-rows count={len(chunk_rows)}")
            async with _pspan("_recall_documents"):
                result = await _recall_documents(pool, chunk_rows, 3, detail, employee_id)
        return result

    async def _pipeline_nodes() -> list[dict]:
        async with _pspan("_pipeline_nodes"):
            async with _pspan("to_thread:_neo4j_hybrid_search"):
                neo4j_nodes = await asyncio.to_thread(
                    _neo4j_hybrid_search,
                    client,
                    query,
                    neo4j_embedding,
                    project,
                    node_limit,
                    allowed_scopes,
                    repo,
                )
            async with _pspan("_build_nodes_with_neighbors"):
                result = await _build_nodes_with_neighbors(
                    client,
                    neo4j_nodes,
                    limit,
                    project,
                    allowed_scopes,
                    repo,
                )
        return result

    async with _pspan("recall") as root_span:
        documents, nodes = await asyncio.gather(
            _pipeline_documents(),
            _pipeline_nodes(),
        )

    logger.info("[recall:profile] ── timing tree ──────────────────────────────")
    _log_profile_tree(root_span)
    logger.info(
        "[recall:profile] ── summary: docs_pipeline=%.1fms  nodes_pipeline=%.1fms  total=%.1fms ──",
        next((c.elapsed_ms for c in root_span.children if c.name == "_pipeline_documents"), 0.0),
        next((c.elapsed_ms for c in root_span.children if c.name == "_pipeline_nodes"), 0.0),
        root_span.elapsed_ms,
    )

    return {
        "documents": documents,
        "nodes": nodes,
        "doc_count": len(documents),
        "node_count": len(nodes),
        "detail": detail,
    }


def get_graph(
    client: Neo4jClient,
    node_ids: list[str],
    project: str = "",
    max_hops: int = 2,
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Topology-only graph view: nodes(id,type,label) + edges(from,to,rel)."""
    max_hops = max(1, min(max_hops, 4))
    nodes, edges = _expand_graph(client, node_ids, project, max_hops, allowed_scopes)
    node_rows = []
    for nid, data in sorted(nodes.items()):
        row = {"id": nid, "type": data["type"], "label": _node_label(data["props"], data["type"])}
        created_at = data["props"].get("created_at")
        if created_at:
            row["created_at"] = created_at
        source_doc = data["props"].get("source_doc") or data["props"].get("source_doc_id")
        if source_doc:
            row["source_doc"] = source_doc
        for metadata_field in ("repo", "doc_type"):
            metadata_value = data["props"].get(metadata_field)
            if metadata_value:
                row[metadata_field] = metadata_value
        node_rows.append(row)
    edge_rows = [
        {"from": e["from"], "to": e["to"], "rel": e["rel"]}
        for e in edges
    ]
    return {"nodes": node_rows, "edges": edge_rows, "returned": len(node_rows)}


def get_graph_expanded(
    client: Neo4jClient,
    node_ids: list[str],
    project: str = "",
    max_hops: int = 1,
    allowed_scopes: list[str] | None = None,
) -> dict:
    """Expanded graph view with type-aware content and edge context.

    max_hops=0: fetch only the requested nodes (no traversal — replaces get_node).
    max_hops=1 (default): seed nodes + their immediate neighbours.
    max_hops=2: two levels of connections.
    """
    max_hops = max(0, min(max_hops, 2))
    nodes, edges = _expand_graph(client, node_ids, project, max_hops, allowed_scopes)
    node_rows = [
        _agent_node_view(nid, data["props"], data["type"])
        for nid, data in sorted(nodes.items())
    ]
    edge_rows = []
    for edge in edges:
        row = {"from": edge["from"], "to": edge["to"], "rel": edge["rel"]}
        edge_context = edge.get("edge_context")
        if edge_context:
            row["edge_context"] = edge_context
        edge_rows.append(row)
    return {"nodes": node_rows, "edges": edge_rows, "returned": len(node_rows)}


def _expand_graph(
    client: Neo4jClient,
    node_ids: list[str],
    project: str = "",
    max_hops: int = 2,
    allowed_scopes: list[str] | None = None,
    repo: str | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    _traversal_rels = ALL_RELS - {"EXTRACTED_FROM", "AUTHORED_BY"}
    rel_filter = "|".join(_traversal_rels)
    label_filter = "+" + "|+".join(ALL_LABELS)
    deduped_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw in node_ids:
        nid = str(raw or "").strip()
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)
        deduped_ids.append(nid)
    deduped_ids = deduped_ids[:50]
    if not deduped_ids:
        return {}, []

    merged_nodes: dict[str, dict] = {}
    merged_edges: dict[tuple[str, str, str], dict] = {}
    try:
        if allowed_scopes is None:
            rows = client.run(
                """
            UNWIND $seed_ids AS sid
            MATCH (seed {id: sid})
            WHERE ($project = '' OR seed.project = $project OR seed.project IS NULL OR seed.project = '')
              AND ($repo IS NULL OR seed.repo IS NULL OR seed.repo = $repo)
            CALL apoc.path.subgraphAll(seed, {
                maxLevel: $max_hops,
                relationshipFilter: $rel_filter,
                labelFilter: $label_filter,
                bfs: true
            })
            YIELD nodes, relationships
            RETURN nodes, relationships
                """,
                {
                    "seed_ids": deduped_ids,
                    "project": project,
                    "repo": repo,
                    "max_hops": max_hops,
                    "rel_filter": rel_filter,
                    "label_filter": label_filter,
                },
            )
        else:
            rows = client.run(
                f"""
                UNWIND $seed_ids AS sid
                MATCH (seed {{id: sid}})
                WHERE ($project = '' OR seed.project = $project OR seed.project IS NULL OR seed.project = '')
                  AND ($repo IS NULL OR seed.repo IS NULL OR seed.repo = $repo)
                  AND (seed:Entity OR seed.source_scope IN $allowed_scopes)
                MATCH p=(seed)-[:{rel_filter}*0..{max_hops}]-(reached)
                WHERE all(r IN relationships(p) WHERE r.source_scope IN $allowed_scopes)
                  AND all(n IN nodes(p) WHERE
                      (n:Entity OR n.source_scope IN $allowed_scopes)
                      AND ($project = '' OR n.project = $project OR n.project IS NULL OR n.project = '')
                      AND ($repo IS NULL OR n.repo IS NULL OR n.repo = $repo)
                  )
                RETURN nodes(p) AS nodes, relationships(p) AS relationships
                LIMIT 500
                """,
                {
                    "seed_ids": deduped_ids,
                    "project": project,
                    "repo": repo,
                    "allowed_scopes": allowed_scopes,
                },
            )
    except Exception:
        logger.warning("APOC subgraphAll query failed in _expand_graph, returning empty rows", exc_info=True)
        rows = []

    for row in rows:
        for neo4j_node in row.get("nodes") or []:
            props = dict(neo4j_node)
            node_project = props.get("project")
            if project and node_project not in (None, "", project):
                continue
            if repo and props.get("repo") not in (None, "", repo):
                continue
            node_type = list(neo4j_node.labels)[0] if hasattr(neo4j_node, "labels") else props.get("_type", "")
            normalised = _normalise_node(props, node_type)
            if allowed_scopes is not None:
                node_scope = normalised.get("source_scope")
                if node_type != "Entity" and node_scope not in allowed_scopes:
                    continue
            nid = normalised.get("id")
            if nid:
                merged_nodes[nid] = {"props": normalised, "type": node_type}

        for rel in row.get("relationships") or []:
            try:
                from_id = rel.start_node["id"] if hasattr(rel, "start_node") else None
                to_id = rel.end_node["id"] if hasattr(rel, "end_node") else None
                rel_type = rel.type if hasattr(rel, "type") else None
                if not (from_id and to_id and rel_type):
                    continue
                rel_props: dict = {}
                try:
                    rel_props = dict(rel.items()) if hasattr(rel, "items") else dict(rel)
                except Exception:
                    rel_props = {}
                if (
                    allowed_scopes is not None
                    and rel_props.get("source_scope") not in allowed_scopes
                ):
                    continue
                if from_id not in merged_nodes or to_id not in merged_nodes:
                    continue
                merged_edges[(from_id, rel_type, to_id)] = {
                    "from": from_id,
                    "rel": rel_type,
                    "to": to_id,
                    "edge_context": rel_props.get("edge_context"),
                }
            except Exception:
                continue

    return merged_nodes, list(merged_edges.values())


def _dedup_edges_simple(edges: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in edges:
        key = (e.get("from"), e.get("rel"), e.get("to"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def _collect_row(row: dict, nodes: dict[str, dict], edges: list[dict]) -> None:
    a = _normalise_node(row["from_props"], row["from_type"])
    b = _normalise_node(row["to_props"],   row["to_type"])
    for n in (a, b):
        nid = n.get("id", "")
        if nid:
            nodes[nid] = n
    edges.append(_normalise_edge(row))




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_node(props: dict, node_type: str) -> dict:
    out = dict(props)
    out.pop("embedding", None)  # never send embedding vectors to the LLM
    out["_type"] = node_type
    return out


_LUCENE_SPECIAL_CHARS_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/&|])')


def _sanitize_fulltext_query(query: str) -> str:
    """Sanitize user query for Neo4j fulltext/Lucene parser."""
    cleaned = query.replace("—", " ").replace("–", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    tokens = [
        _LUCENE_SPECIAL_CHARS_RE.sub(r"\\\1", token)
        for token in cleaned.split(" ")
        if token
    ]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " OR ".join(tokens)


def _node_label(node: dict, node_type: str) -> str:
    if node_type == "Entity":
        return str(node.get("canonical_name") or node.get("label") or node.get("id") or "")
    if node_type == "Preference":
        return str(node.get("category") or node.get("label") or node.get("id") or "")
    if node_type == "Fact":
        return str(node.get("statement") or node.get("label") or node.get("id") or "")
    if node_type == "Document":
        return str(node.get("title") or node.get("label") or node.get("id") or "")
    return str(node.get("label") or node.get("id") or "")


def _node_content(node: dict, node_type: str) -> str:
    if node_type == "Entity":
        canonical = str(node.get("canonical_name") or node.get("label") or "")
        aliases = node.get("aliases") or []
        alias_text = ", ".join(str(a) for a in aliases) if isinstance(aliases, list) else str(aliases or "")
        if canonical and alias_text:
            return f"{canonical}; aliases: {alias_text}"
        return canonical or alias_text
    if node_type == "Document":
        title = str(node.get("title") or node.get("label") or node.get("id") or "").strip()
        preview = _document_preview(node)
        if title and preview:
            return f"{title}\n{preview}"
        return title or preview
    primary_key = PRIMARY_CONTENT_FIELDS.get(node_type)
    if primary_key:
        return str(node.get(primary_key) or "")
    for key in ("description", "body", "rationale", "observation"):
        if node.get(key):
            return str(node[key])
    return ""


def _node_snippet(node: dict, node_type: str) -> str:
    content = _node_content(node, node_type)
    if not content:
        return _node_label(node, node_type)[:SEARCH_SNIPPET_CHARS]
    return content[:SEARCH_SNIPPET_CHARS]


def _document_preview(node: dict) -> str:
    """Best-effort short document preview from available summary/text fields."""
    for key in ("summary", "text", "content", "excerpt", "snippet", "body", "description"):
        value = node.get(key)
        if isinstance(value, str):
            cleaned = " ".join(value.split()).strip()
            if cleaned:
                return cleaned[:SEARCH_SNIPPET_CHARS]
    return ""


def _agent_node_view(node_id: str, node: dict, node_type: str) -> dict:
    out: dict = {
        "id": node_id,
        "type": node_type or "Unknown",
        "label": _node_label(node, node_type),
        "content": _node_content(node, node_type),
    }
    for field in SECONDARY_CONTENT_FIELDS.get(node_type, []):
        if field in AGENT_METADATA_EXCLUSIONS:
            continue
        value = node.get(field)
        if value is not None and value != "":
            out[field] = value
    created_at = node.get("created_at")
    if created_at:
        out["created_at"] = created_at
    source_doc = node.get("source_doc") or node.get("source_doc_id")
    if source_doc:
        out["source_doc"] = source_doc
    for metadata_field in ("repo", "doc_type"):
        metadata_value = node.get(metadata_field)
        if metadata_value:
            out[metadata_field] = metadata_value
    return out


def _normalise_edge(r: dict) -> dict:
    rp = dict(r.get("rel_props") or {})
    rp.pop("_rel_id", None)
    return {
        "from":      r["from_id"],
        "from_type": r["from_type"],
        "to":        r["to_id"],
        "to_type":   r["to_type"],
        "rel":       r["rel_type"],
        **rp,
    }


def _dedup_edges(edges: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for e in edges:
        key = (e.get("from"), e.get("rel"), e.get("to"))
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ---------------------------------------------------------------------------
# get_node — fetch a single node by id
# ---------------------------------------------------------------------------

def get_node(
    client: Neo4jClient,
    node_id: str,
    project: str | None = None,
    allowed_scopes: list[str] | None = None,
) -> dict | None:
    """Return all properties of a single node by id, or None if not found.

    If project is supplied the node must belong to that project (prevents
    cross-project information leakage on shared deployments).
    """
    project_filter = "AND (n.project = $project OR n.project IS NULL OR n.project = '')" if project else ""
    params: dict = {"id": node_id}
    if project:
        params["project"] = project
    scope_filter = ""
    if allowed_scopes is not None:
        scope_filter = "AND (n:Entity OR n.source_scope IN $allowed_scopes)"
        params["allowed_scopes"] = allowed_scopes
    rows = client.run(
        f"MATCH (n {{id: $id}}) WHERE true {project_filter} {scope_filter} "
        "RETURN labels(n)[0] AS node_type, properties(n) AS props LIMIT 1",
        params,
    )
    if not rows:
        return None
    r = rows[0]
    return _normalise_node(r["props"], r["node_type"])


# ---------------------------------------------------------------------------
# assign_topic  (MEMBER_OF edge creation)
# ---------------------------------------------------------------------------

_MEMBER_OF_VALID_TYPES = {"Decision", "Constraint", "Insight", "Signal"}


def assign_topic(
    client: Neo4jClient,
    node_id: str,
    node_type: str,
    topic_id: str,
    edge_context: str,
    allowed_scopes: list[str] | None = None,
) -> None:
    """Create a MEMBER_OF edge from a semantic node to a Topic (idempotent).

    Scope-gated: the MATCH on n includes a source_scope check so that employees
    cannot create edges to nodes outside their access scopes. The write is a
    no-op if n is not found or the caller lacks scope access.
    allowed_scopes=None bypasses filtering (service accounts).
    """
    if node_type not in _MEMBER_OF_VALID_TYPES:
        raise ValueError(f"assign_topic: invalid node_type '{node_type}'")

    scope_where = ""
    params: dict = {
        "node_id":      node_id,
        "topic_id":     topic_id,
        "edge_context": edge_context,
        "node_type":    node_type,
    }
    if allowed_scopes is not None:
        scope_where = "AND n.source_scope IN $allowed_scopes"
        params["allowed_scopes"] = allowed_scopes

    cypher = f"""
        MATCH (n:{node_type} {{id: $node_id}})
        WHERE true {scope_where}
        MATCH (t:Topic {{id: $topic_id}})
        MERGE (n)-[r:MEMBER_OF]->(t)
        SET r.edge_context = $edge_context,
            r.inferred     = false,
            r.from_type    = $node_type,
            r.to_type      = 'Topic'
    """
    client.run_write(cypher, params)
