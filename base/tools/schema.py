"""
tools/schema.py — Authoritative semantics.json schema definition.
Served via the get_schema MCP tool (webserver/main.py → get_schema()).

Write format: observation traces (trace_id-keyed).
Server handles graph extraction, entity resolution, and dedup from trace body.
"""

import json

SCHEMA = {
    "trace_format": {
        "description": (
            "Each semantics.json entry is one node-embedded observation trace. "
            "The server assigns graph ids and provenance; clients provide the "
            "semantic node content and must not provide edges."
        ),
        "fields": {
            "trace_id":   "UUID string — unique per trace",
            "ts":         "ISO8601 datetime string",
            "session_id": "Chat/session UUID (optional)",
            "project":    "Organization project slug assigned by the server when omitted",
            "repo":       (
                "Optional canonical GitHub repository in owner/repo form. "
                "Omit for organization-wide context."
            ),
            "node":       (
                "Object with type=Decision|Constraint|Insight|Signal and the "
                "matching fields from structured_fields. Include label, origin, "
                "and tag in this object when applicable; do not include id or edges."
            ),
            "keywords":   "Array of 3-8 specific keywords/phrases, that will be used to search knowledge-base for context — no generic words",
            "mentions":   "Array of {mention_text, mention_type} — see mention_types below",
        },
    },
    "tag_values": {
        "checkpoint:stop":   "Stop hook checkpoint",
        "checkpoint:manual": "Manual capture mid-session",
        "failure:tool":      "Tool-failure capture",
        "session:start":     "Session-context entry",
    },
    "mention_types": [
        "person", "service", "org", "repo", "tool",
        "object", "location", "event", "role_reference", "pronoun",
    ],
    "structured_fields": {
        "Decision": {
            "required": ["rationale"],
            "optional": ["trigger", "alternatives", "confidence", "uncertainty"],
            "fields": {
                "rationale":    "Why this choice was made over alternatives",
                "trigger":      "What forced or prompted this decision",
                "alternatives": "[{option: str, rejected_because: str}]",
                "confidence":   "high | medium | low  (default: medium)",
                "uncertainty":  "Known unknowns, if any",
            },
        },
        "Constraint": {
            "required": ["description"],
            "optional": ["source", "hard", "symptom"],
            "fields": {
                "description": "What is forbidden or required and why",
                "source":      "platform | observed-failure | user-requirement | api-limitation | performance | business-rule | legal | environment | dependency | organizational | plan",
                "hard":        "boolean — true if violating causes failure (default: false)",
                "symptom":     "Observable symptom when violated — include when hard: true",
            },
        },
        "Insight": {
            "required": ["body"],
            "optional": ["failure_without"],
            "fields": {
                "body":            "The tribal knowledge or non-obvious implementation reasoning",
                "failure_without": "Concrete failure mode if this insight is unknown",
            },
        },
        "Signal": {
            "required": ["observation"],
            "optional": ["appears_to_mean", "actually_means", "conditions", "evidence"],
            "fields": {
                "observation":    "The symptom, trend, or observable event",
                "appears_to_mean": "Naive / surface interpretation",
                "actually_means": "Correct interpretation",
                "conditions":     "When / where this signal occurs",
                "evidence":       "How this was determined",
            },
        },
    },
}


def get_schema() -> str:
    return json.dumps(SCHEMA, indent=2)
