---
name: eidetix-mcp
description: Extended guide for the Eidetix MCP tools (get_repository_context, export_repository_profile, recall, get_graph, get_graph_expanded, get_content, resolve_entities). Use when the server preamble and per-tool descriptions are not enough — worked examples, relationship glossary, decision tree, and edge cases.
disable-model-invocation: true
---

# Eidetix MCP — Extended Guide

Eidetix is the team's long-term memory: past Decisions, Constraints that shaped them, Insights, and Signals, all linked by how they influenced each other.

## Repository task start

At the start of substantive work in a Git repository:

1. Use `git rev-parse --show-toplevel` for the root and `git remote get-url
   origin` for GitHub `owner/repo`. A directory basename is not repository
   identity.
2. Resolve local `HEAD`, the tracked upstream (or `origin/HEAD`), merge-base,
   dirty status, and changed paths.
3. Call `get_repository_context` with the repository, task, and that local git
   state. The tool fetches the canonical profile directly rather than relying
   on vector ranking.
4. Inspect local code with grep/search, reads, LSP, and tests. Preserve command
   status (`verified`, `documented`, or `inferred`) and verify applicable
   commands locally.
5. Dirty files override profile implementation claims and must be inspected.
   Never discard local work or rebuild shared context from unmerged changes.

Use `export_repository_profile` when deterministic, cited Markdown is needed.
It is read-only and includes ref, commit, generated time, and stale warnings.

## The 3-step recipe

1. **recall** a topic — returns document excerpts + semantic graph hits in one call.
2. **get_graph** on promising node ids to see what shaped/caused/relates to them.
3. **get_graph_expanded** or **get_content** to read the detail you need.

---

## Decision tree

| Goal | Tool |
|---|---|
| "What should I know before changing this repository?" | `get_repository_context` |
| "Export the canonical repository brief" | `export_repository_profile` |
| "Do we have anything on X?" | `recall` |
| "What else is connected to this result?" | `get_graph` |
| "I know which items I want — give me full content" | `get_graph_expanded` |
| "Read the original document behind this result" | `get_content` |
| "I have a name (person/service/tool) — find the canonical record" | `resolve_entities` then `recall` |

---

## `recall` response shape

`recall` returns two independent sections:

```json
{
  "documents": [
    {
      "doc_id": "doc:notion:abc123",
      "title": "Q2 Architecture Review",
      "chunk_index": 2,
      "similarity": 0.921,
      "source_excerpt": "...exact chunk text from the document..."
    }
  ],
  "nodes": [
    {
      "id": "decision:use-postgres-for-sessions",
      "type": "Decision",
      "label": "Use Postgres for session storage",
      "score": 0.043,
      "snippet": "Chose Postgres over Redis to reduce operational overhead...",
      "source_doc": "doc:notion:abc123",
      "neighbors": [
        { "id": "constraint:redis-ops-overhead", "type": "Constraint", "rel": "SHAPED_BY" }
      ]
    }
  ]
}
```

- `documents[]` — ranked by vector similarity; each entry is a specific chunk with its excerpt already fetched from S3 (`detail=full`, the default).
- `nodes[]` — semantic hits (Decision, Constraint, Insight, Signal, Entity …) ranked by RRF; each includes 1-hop neighbors.
- Use `source_doc` on a node to call `get_content(doc_id=source_doc)` for the full document.

---

## Worked examples

### Example 1 — research a topic end-to-end

```
recall(query="Postgres session storage")
→ {
    documents: [{ doc_id: "doc:notion:abc", chunk_index: 2, similarity: 0.91,
                  source_excerpt: "We evaluated Redis and Postgres for session storage..." }],
    nodes: [{ id: "decision:use-postgres-for-sessions", type: "Decision",
              neighbors: [{ id: "constraint:redis-ops-overhead", rel: "SHAPED_BY" }] }]
  }

get_graph(node_ids=["decision:use-postgres-for-sessions"], max_hops=2)
→ nodes + edges: "constraint:redis-ops-overhead" SHAPED_BY, "insight:session-expiry-bug" CAUSES

get_graph_expanded(node_ids=["constraint:redis-ops-overhead"], max_hops=0)
→ full content for just that constraint: description, hard=true, symptom
```

### Example 2 — read the source doc

```
recall(query="MCP streamable HTTP transport")
→ {
    documents: [{ doc_id: "doc:slack:thread-2025-04-22", title: "MCP transport decision",
                  source_excerpt: "Switched to streamable HTTP because SSE couldn't..." }],
    nodes: [...]
  }

get_content(doc_id="doc:slack:thread-2025-04-22")
→ full document text
```

### Example 3 — entity-first lookup

```
resolve_entities(names=["Sudesh", "Neo4j"])
→ [{ canonical_name: "Sudesh Kulkarni", id: "entity:sudesh-kulkarni", ... }, ...]

recall(query="Sudesh Kulkarni Neo4j")
→ decisions, constraints, insights mentioning those entities + relevant document chunks
```

---

## Relationship glossary

These are the `rel` values you will see in `get_graph` / `get_graph_expanded` edges and in `nodes[].neighbors`:

| rel | Plain-English meaning |
|---|---|
| `SHAPED_BY` | This decision/insight was directly influenced by this constraint |
| `EXPLAINS` | This node gives the reasoning behind another |
| `CAUSES` | This signal or constraint leads to the connected outcome |
| `CONTRADICTS` | These two nodes are in tension with each other |
| `LIFTED` | A constraint was removed or relaxed by this decision |
| `SUPERSEDES` | Newer record replaces an older one |
| `ABANDONED` | This option was considered and dropped |
| `MEMBER_OF` | This node belongs to a Topic grouping |
| `EXTRACTED_FROM` | This semantic node came from this Document |
| `MENTIONS` | A node references a person, service, or tool (Entity) |
| `ABOUT` | A document is about this topic or entity |

High-signal for context research: `SHAPED_BY`, `EXPLAINS`, `CAUSES`, `CONTRADICTS`, `LIFTED`.

---

## Node content fields by type

Each node in `get_graph_expanded` returns a `content` field — the richest single field for that type:

| Type | `content` field | Other key fields |
|---|---|---|
| Decision | `rationale` | `trigger`, `alternatives`, `confidence` |
| Constraint | `description` | `hard`, `symptom`, `source` |
| Insight | `body` | `failure_without` |
| Signal | `observation` | `actually_means`, `appears_to_mean`, `conditions` |
| Document | title + first ~200 chars | `source_provider` |
| Entity | `canonical_name` + aliases | `entity_type` |

---

## Edge cases

- **Empty recall results**: the topic is not yet in memory. Proceed without context.
- **`documents[]` empty, `nodes[]` non-empty**: semantic nodes exist but no document chunks matched — graph context only.
- **`nodes[]` empty, `documents[]` non-empty**: raw document evidence exists but no structured semantic nodes yet.
- **`max_hops=0` on `get_graph_expanded`**: fetches only the exact nodes requested, no traversal. Use when you already know the ids and only want full content.
- **`get_graph` with `max_hops=4`**: widest map; use only for broad discovery, not for reading content.
- **`resolve_entities` batch**: pass all names in one call as `names=[...]`. Prefer batch over multiple single-name calls.
- **`source_doc` on a node**: call `get_content(doc_id=source_doc)` to read the full original document.
