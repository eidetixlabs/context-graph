---
name: eidetix-node-schema
description: Full Eidetix node schema for writing observation traces. Invoke before writing any trace to get exact field definitions per node type.
---

# Eidetix Node Schema

Use this when writing traces. The `node` field of a trace must match one of these schemas exactly.

## Trace envelope

```json
{
  "trace_id": "<uuid>",
  "ts": "<ISO8601>",
  "session_id": "<chat-uuid>",
  "node": { ... },
  "mentions": [{ "mention_text": "...", "mention_type": "..." }],
  "keywords": ["3-8 specific terms"]
}
```

## Rules that apply to ALL node types

- **`id` — omit or set to anything.** The processor always overwrites it as `{type.lower()}:{trace_id}`.
- **`source_doc` — omit.** The processor clears this.
- **`origin`** — set to `"cursor"` (Cursor) or `"claude"` (Claude Code). Lives inside `node`.
- **`tag`** — lives inside `node`. See tag values below.
- **`created_at`** — optional; processor falls back to top-level `ts` if absent.
- **`scope`** — defaults to `"system"`. Use `"subsystem:X"` or `"module:X"` for narrower scope.
- **`embedding`** — never set; server-computed.

**Tag values:**
- `checkpoint:stop` — Stop hook checkpoint
- `checkpoint:manual` — Manual capture mid-session
- `failure:tool` — Tool-failure capture
- `session:start` — Session-context entry

---

## Decision

**When:** A choice was made where real alternatives existed.

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"Decision"` |
| `label` | yes | Short human-readable label |
| `rationale` | yes | Why this choice was made |
| `trigger` | no | What prompted this decision |
| `alternatives` | no | `[{"option": "...", "rejected_because": "..."}]` |
| `confidence` | no | `"high"` \| `"medium"` \| `"low"` (default: `"medium"`) |
| `uncertainty` | no | Known unknowns |
| `scope` | no | Default: `"system"` |
| `origin` | yes | `"cursor"` or `"claude"` |
| `tag` | no | See tag values |

```json
{
  "type": "Decision",
  "label": "Use Postgres for session storage",
  "rationale": "Redis adds ops overhead; Postgres already in stack and sessions fit within 10k rows.",
  "trigger": "Session expiry bug revealed the in-memory store was not persisted across restarts.",
  "alternatives": [
    {"option": "Redis", "rejected_because": "Adds a new stateful dependency with no other use case."},
    {"option": "In-memory", "rejected_because": "Lost on restart — caused the original bug."}
  ],
  "confidence": "high",
  "scope": "system",
  "origin": "cursor",
  "tag": "checkpoint:stop"
}
```

---

## Constraint

**When:** A hard or soft limit was discovered (platform, API, policy, budget, environment).

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"Constraint"` |
| `label` | yes | Short human-readable label |
| `description` | yes | What the constraint is and why it exists |
| `source` | no | What imposes it: `"platform"`, `"api-limitation"`, `"performance"`, `"business-rule"`, `"legal"`, `"environment"`, `"dependency"`, `"organizational"`, etc. |
| `hard` | no | `true` if violating causes failure (default: `false`) |
| `symptom` | no | Observable symptom when violated — include when `hard: true` |
| `scope` | no | Default: `"system"` |
| `origin` | yes | `"cursor"` or `"claude"` |
| `tag` | no | See tag values |

```json
{
  "type": "Constraint",
  "label": "Notion API: max 100 blocks per request",
  "description": "Notion's block-children endpoint returns at most 100 items per page. Queries exceeding this silently truncate without error.",
  "source": "api-limitation",
  "hard": true,
  "symptom": "Pages with >100 blocks appear partially crawled with no error in logs.",
  "scope": "subsystem:notion-crawler",
  "origin": "cursor",
  "tag": "checkpoint:stop"
}
```

---

## Insight

**When:** Tribal knowledge that would waste 30+ min without it; non-obvious implementation reasoning.

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"Insight"` |
| `label` | yes | Short human-readable label |
| `body` | yes | The knowledge, observation, or learning — full prose, name everything |
| `failure_without` | no | Concrete failure mode if this insight is unknown |
| `scope` | no | Default: `"system"` |
| `origin` | yes | `"cursor"` or `"claude"` |
| `tag` | no | See tag values |

```json
{
  "type": "Insight",
  "label": "Neo4j MERGE on list properties requires exact array equality",
  "body": "When using MERGE in Neo4j with a node that has a list property (e.g. aliases), the MERGE only matches if the array is identical in order and value. Any difference creates a duplicate node. Use SET after MERGE to update list fields instead of matching on them.",
  "failure_without": "Silent duplicate Entity nodes are created on every upsert, breaking entity resolution.",
  "scope": "subsystem:neo4j",
  "origin": "cursor",
  "tag": "checkpoint:stop"
}
```

---

## Signal

**When:** A non-obvious symptom or indicator — the naive interpretation is wrong.

| Field | Required | Description |
|---|---|---|
| `type` | yes | `"Signal"` |
| `label` | yes | Short human-readable label |
| `observation` | yes | The symptom, trend, or observable event |
| `appears_to_mean` | no | The naive / surface interpretation |
| `actually_means` | no | The correct interpretation |
| `conditions` | no | When / where this signal occurs |
| `evidence` | no | How this was determined |
| `scope` | no | Default: `"system"` |
| `origin` | yes | `"cursor"` or `"claude"` |
| `tag` | no | See tag values |

```json
{
  "type": "Signal",
  "label": "connection refused on MCP socket after cold start",
  "observation": "MCP client receives 'connection refused' immediately after desktop app launch.",
  "appears_to_mean": "The MCP server failed to start.",
  "actually_means": "The server started but the TLS certificate is not yet trusted by the OS. The handshake fails before the TCP connection is established.",
  "conditions": "First launch after install, or after OS keychain reset.",
  "evidence": "Wireshark trace showed TCP RST at TLS ClientHello, not at TCP SYN.",
  "scope": "subsystem:mcp-server",
  "origin": "cursor",
  "tag": "failure:tool"
}
```

---

## Mention types

Valid `mention_type` values for the `mentions` array: `person` · `service` · `org` · `repo` · `tool`
