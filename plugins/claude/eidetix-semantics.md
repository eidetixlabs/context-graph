# Context Graph — Semantic Journaling (Claude Code)

Prefer MCP `ingest_traces` against a local Context Graph server. You may also
append observation traces to `~/.context-graph/projects/<slug>/semantics.json`.
For that storage key, use `basename(git rev-parse --show-toplevel)`,
falling back to `basename(cwd)` only outside Git. This slug is never the GitHub
repository identity. The Context Graph processor handles extraction, entity
resolution, and dedup.

---

## Repository context gate (substantive tasks)

Before substantive implementation, debugging, architecture, or review work:

1. Resolve the root with `git rev-parse --show-toplevel`.
2. Read `git remote get-url origin` and derive GitHub `owner/repo`; never use a
   directory basename as repository identity.
3. Resolve local `HEAD`, the tracked upstream (falling back to `origin/HEAD`),
   merge-base, dirty status, and changed paths.
4. Call `get_repository_context` with `repository`, `task`, `local_head`,
   `merge_base`, `working_tree_dirty`, and `changed_paths`.
5. Use the result as a shared baseline, then inspect local code with grep,
   reads, LSP, and relevant tests. Verify command candidates locally.

Dirty changes override profile implementation claims and must be inspected.
Never discard local work or regenerate shared context from unmerged changes.
Skip this gate for trivial edits, non-Git directories, or repositories without
a usable GitHub remote.

## Recall gate (mandatory!)

Before saying "I don't know", "not familiar", "could you clarify", or asking any
question about a name, term, entity, product, or prior decision, call the
**context-graph** MCP server first:

- `recall(query)` — start here; returns `documents[]` (chunk excerpts ranked by vector similarity) and `nodes[]` (semantic hits with 1-hop neighbors)

Unknown proper nouns and uncertain references are mandatory recall triggers.
Only ask the user after `recall(query)` returns no relevant context.

---

## What to capture

- **Decision** — architecture or approach choice where alternatives existed; includes tradeoff conclusions with durable consequences
- **Constraint** — hard limit discovered (platform/API/runtime/tooling/environment)
- **Insight** — tribal knowledge that would waste 30+ min without it; non-obvious implementation reasoning
- **Signal** — non-obvious indicator of direction, risk, or quality; ruled-out paths count even without a formal decision

**"No signal this session" is valid.** Precision over recall.

### Hard gates — do NOT write when

- Session ended without conclusion and produced no durable findings
- You corrected your reasoning mid-session (discard pre-correction analysis)
- Build/tests failed and were NOT resolved
- A "decision" had no real alternatives — only one obvious path

---

## Trace format

Before writing, read **`node-schema.md`** in this directory for exact node field definitions per type.

One trace per distinct finding. Append to the JSON array; **never mutate existing entries**.

```json
{
  "trace_id": "<uuid>",
  "ts": "<ISO8601>",
  "session_id": "<session_id>",
  "node": {
    "type": "Decision | Constraint | Insight | Signal",
    "label": "Short human-readable label",
    "origin": "claude",
    "tag": "checkpoint:stop",
    "...": "category-specific fields — see node-schema.md"
  },
  "mentions": [
    { "mention_text": "<exact name>", "mention_type": "person | service | org | repo | tool" }
  ],
  "keywords": ["3-8 specific keywords/phrases"]
}
```

**Do NOT set `id` or `source_doc` — the processor assigns these.**

**`tag` values:**
- `checkpoint:stop` — Stop hook checkpoint
- `checkpoint:manual` — manual capture mid-session
- `failure:tool` — tool-failure capture
- `session:start` — session-context entry

**Trace writing rules:**
- **Exhaustive prose** — name every person, service, tool, ticket, and metric in node body fields
- **No graph references** — no node IDs, edge types, or graph internals; traces are self-contained
- **Keywords** — 3–8 specific terms/phrases; avoid generic words
- Traces are append-only; corrections = new trace with updated summary, not mutation

---

## Stop checkpoint

If the Stop hook asks for a checkpoint: write up to **3** high-signal traces from this turn, or nothing. Keep `semantics.json` valid JSON. The hook blocks at most once per turn.
