# Agent Trace Extraction Guide

Eidetix is the team's long-term memory. At the end of every completed action or chat, run semantic extraction as a **hard gate** before finalizing. If there is any durable signal, write one trace per finding and submit via `ingest_traces`. If truly nothing worth capturing, state: `Semantic extraction: no signal in this chat.`

---

## What to capture (and what to skip)

Capture only **durable, high-signal** findings in exactly four categories:

| Category | What it is | Write when… |
|---|---|---|
| **Decision** | A choice where real alternatives existed and the outcome has lasting consequences | Architecture or approach was chosen over other viable options; tradeoffs were resolved |
| **Constraint** | A hard or soft limit imposed by platform, API, policy, budget, environment, or dependency | You discovered something that forbids, requires, or bounds future work |
| **Insight** | Tribal knowledge — non-obvious implementation reasoning that would waste 30+ minutes without it | You learned how something actually works in this codebase/stack, not obvious from docs |
| **Signal** | A non-obvious symptom or indicator where the naive interpretation is wrong | An observable event, error, or trend that misleads unless you know the real cause |

**Precision over recall.** Five clear traces beat twenty noisy ones. `"No signal this session"` is valid.

### Hard gates — do NOT write when

- The session ended without conclusion and produced no durable findings
- You corrected your reasoning mid-session (discard pre-correction analysis)
- Build or tests failed and were **not** resolved
- A "decision" had no real alternatives — only one obvious path
- The finding is speculative, unresolved, or still being investigated

---

## Trace envelope

Each trace is one JSON object. Submit an array of these via `ingest_traces`.

```json
{
  "trace_id": "<uuid>",
  "ts": "<ISO8601>",
  "session_id": "<chat-or-run-uuid>",
  "node": { "...see category sections below..." },
  "mentions": [
    { "mention_text": "<exact name>", "mention_type": "person | service | org | repo | tool" }
  ],
  "keywords": ["3-8", "specific", "searchable", "terms"]
}
```

### Envelope field reference

| Field | Required | Meaning |
|---|---|---|
| `trace_id` | yes | UUID unique to this trace. Generate a new one per finding. |
| `ts` | yes | ISO8601 timestamp when the finding was captured (e.g. `2026-06-15T14:30:00Z`). |
| `session_id` | no | Chat, agent run, or task UUID — links traces from the same session. |
| `node` | yes | The semantic payload. Must include `type`, `label`, and category-specific fields (see below). |
| `mentions` | no | Explicitly named entities referenced in the trace. Used for entity resolution in the knowledge graph. |
| `keywords` | yes | 3–8 specific terms or phrases for search and recall. Avoid generic words like "fix", "update", "issue". |

### Rules that apply to ALL traces

- **One trace per distinct finding.** Do not bundle unrelated decisions into one trace.
- **Append-only mindset.** Corrections = a new trace with updated content, never mutation of prior traces.
- **Exhaustive prose.** Name every person, service, tool, ticket, metric, and file path in node body fields. The server resolves entities from this text.
- **No graph references.** Do not embed node IDs, edge types, or graph internals — traces are self-contained observations.
- **Do NOT set `id` or `source_doc`** inside `node` — the processor assigns these.
- **Do NOT set `embedding`** — server-computed.

Call `get_schema` for machine-readable field constraints and required structured fields per category.

---

## Node fields common to all categories

These live inside the `node` object alongside category-specific fields:

| Field | Required | Meaning |
|---|---|---|
| `type` | yes | `"Decision"` \| `"Constraint"` \| `"Insight"` \| `"Signal"` |
| `label` | yes | Short human-readable title (what you'd scan in a list). |
| `origin` | yes | Runtime that produced this trace: `"cursor"`, `"claude"`, or `"agent"`. |
| `tag` | no | Provenance tag — see tag values below. Default to `"checkpoint:stop"` at end-of-chat extraction. |
| `scope` | no | Defaults to `"system"`. Use `"subsystem:X"` or `"module:X"` for narrower scope. |
| `created_at` | no | Optional; processor falls back to top-level `ts` if absent. |

**Tag values:**

| Tag | When to use |
|---|---|
| `checkpoint:stop` | End-of-chat or end-of-action extraction (default) |
| `checkpoint:manual` | Mid-session manual capture |
| `failure:tool` | A tool failure revealed a Constraint or Signal |
| `session:start` | Session-context entry at conversation start |

---

## Decision

**When:** A choice was made where real alternatives existed and the outcome matters for future work.

| Field | Required | Meaning |
|---|---|---|
| `rationale` | yes | Why this choice was made over the alternatives. |
| `trigger` | no | What forced or prompted the decision (bug, requirement, deadline). |
| `alternatives` | no | Array of `{"option": "...", "rejected_because": "..."}` for each rejected path. |
| `confidence` | no | `"high"` \| `"medium"` \| `"low"` (default: `"medium"`). |
| `uncertainty` | no | Known unknowns that could change the decision later. |

```json
{
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "ts": "2026-06-15T14:30:00Z",
  "session_id": "run-uuid-here",
  "node": {
    "type": "Decision",
    "label": "Use Postgres for session storage",
    "rationale": "Redis adds ops overhead; Postgres is already in the stack and session volume fits within 10k rows.",
    "trigger": "Session expiry bug revealed the in-memory store was not persisted across restarts.",
    "alternatives": [
      {"option": "Redis", "rejected_because": "Adds a new stateful dependency with no other use case."},
      {"option": "In-memory", "rejected_because": "Lost on restart — caused the original bug."}
    ],
    "confidence": "high",
    "scope": "system",
    "origin": "agent",
    "tag": "checkpoint:stop"
  },
  "mentions": [
    {"mention_text": "Postgres", "mention_type": "service"},
    {"mention_text": "Redis", "mention_type": "service"}
  ],
  "keywords": ["session storage", "Postgres", "Redis", "persistence", "stateful dependency"]
}
```

---

## Constraint

**When:** A hard or soft limit was discovered — platform, API, policy, budget, environment, or dependency.

| Field | Required | Meaning |
|---|---|---|
| `description` | yes | What the constraint is and why it exists. Be specific about the limit. |
| `source` | no | What imposes it: `platform`, `api-limitation`, `performance`, `business-rule`, `legal`, `environment`, `dependency`, `organizational`, `observed-failure`, `user-requirement`, `plan`. |
| `hard` | no | `true` if violating causes failure (default: `false`). |
| `symptom` | no | Observable symptom when violated — **include when `hard: true`**. |

```json
{
  "trace_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "ts": "2026-06-15T14:30:00Z",
  "node": {
    "type": "Constraint",
    "label": "Notion API: max 100 blocks per request",
    "description": "Notion's block-children endpoint returns at most 100 items per page. Queries exceeding this silently truncate without error.",
    "source": "api-limitation",
    "hard": true,
    "symptom": "Pages with >100 blocks appear partially crawled with no error in logs.",
    "scope": "subsystem:notion-crawler",
    "origin": "agent",
    "tag": "checkpoint:stop"
  },
  "mentions": [
    {"mention_text": "Notion", "mention_type": "service"}
  ],
  "keywords": ["Notion API", "block limit", "100 blocks", "silent truncation", "crawler"]
}
```

---

## Insight

**When:** Tribal knowledge that would waste 30+ minutes without it — non-obvious implementation reasoning.

| Field | Required | Meaning |
|---|---|---|
| `body` | yes | The knowledge itself — full prose, name everything explicitly. |
| `failure_without` | no | Concrete failure mode if this insight is unknown. |

```json
{
  "trace_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "ts": "2026-06-15T14:30:00Z",
  "node": {
    "type": "Insight",
    "label": "Neo4j MERGE on list properties requires exact array equality",
    "body": "When using MERGE in Neo4j with a node that has a list property (e.g. aliases), the MERGE only matches if the array is identical in order and value. Any difference creates a duplicate node. Use SET after MERGE to update list fields instead of matching on them.",
    "failure_without": "Silent duplicate Entity nodes are created on every upsert, breaking entity resolution.",
    "scope": "subsystem:neo4j",
    "origin": "agent",
    "tag": "checkpoint:stop"
  },
  "mentions": [
    {"mention_text": "Neo4j", "mention_type": "service"}
  ],
  "keywords": ["Neo4j MERGE", "list property", "array equality", "duplicate nodes", "entity resolution"]
}
```

---

## Signal

**When:** A non-obvious symptom or indicator — the naive interpretation is wrong.

| Field | Required | Meaning |
|---|---|---|
| `observation` | yes | The symptom, trend, or observable event you saw. |
| `appears_to_mean` | no | The naive / surface interpretation. |
| `actually_means` | no | The correct interpretation. |
| `conditions` | no | When or where this signal occurs. |
| `evidence` | no | How you determined the correct interpretation. |

```json
{
  "trace_id": "d4e5f6a7-b8c9-0123-def0-234567890123",
  "ts": "2026-06-15T14:30:00Z",
  "node": {
    "type": "Signal",
    "label": "connection refused on MCP socket after cold start",
    "observation": "MCP client receives 'connection refused' immediately after desktop app launch.",
    "appears_to_mean": "The MCP server failed to start.",
    "actually_means": "The server started but the TLS certificate is not yet trusted by the OS. The handshake fails before the TCP connection is established.",
    "conditions": "First launch after install, or after OS keychain reset.",
    "evidence": "Wireshark trace showed TCP RST at TLS ClientHello, not at TCP SYN.",
    "scope": "subsystem:mcp-server",
    "origin": "agent",
    "tag": "failure:tool"
  },
  "mentions": [
    {"mention_text": "MCP", "mention_type": "tool"}
  ],
  "keywords": ["connection refused", "TLS certificate", "cold start", "MCP handshake", "keychain"]
}
```

---

## Mentions

The `mentions` array links named entities to the trace for graph resolution.

| `mention_type` | Use for |
|---|---|
| `person` | Named individuals |
| `service` | External or internal services (Postgres, Neo4j, Notion, S3) |
| `org` | Companies or teams |
| `repo` | Git repositories |
| `tool` | CLI tools, frameworks, MCP servers |

Only these five types are resolved by the processor. Include high-signal, explicitly named entities only — do not mention every noun in the trace.

---

## Workflow checklist

1. **Recall first** — use `recall(query)` before claiming ignorance about prior decisions or entities.
2. **Signal check** — did this session produce a Decision, Constraint, Insight, or Signal?
3. **Hard gate** — if yes, read this guide (or call `get_schema`) and construct trace(s).
4. **Submit** — call `ingest_traces` with the trace array.
5. **Confirm** — state what you wrote, or explicitly say no signal.

A task or chat is **not complete** until extraction is done.
