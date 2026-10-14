"""processor/providers/jira.py — Jira issue discourse processor worker."""

from __future__ import annotations

import json
from typing import Any

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker


def _actor_label(actor: Any) -> str:
    if not isinstance(actor, dict):
        return "unknown"
    display_name = str(actor.get("display_name") or "unknown").strip() or "unknown"
    account_id = str(actor.get("account_id") or "").strip()
    if account_id:
        return f"{display_name} ({account_id})"
    return display_name


def _jira_discourse_to_text(doc: DocumentRecord, content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        if doc.title and not content.startswith(doc.title) and not content.startswith(f"# {doc.title}"):
            return f"# {doc.title}\n\n{content}"
        return content

    if not isinstance(payload, dict):
        return content

    issue_key = str(payload.get("issue_key") or "").strip()
    title = str(payload.get("title") or doc.title or issue_key or "Jira issue").strip()
    description = str(payload.get("description") or "").strip()
    labels = payload.get("labels")
    comments = payload.get("comments")

    lines: list[str] = [f"# {title}"]
    if issue_key:
        lines.append(f"Issue: {issue_key}")
    if payload.get("issue_type"):
        lines.append(f"Type: {payload.get('issue_type')}")
    if payload.get("status"):
        lines.append(f"Status: {payload.get('status')}")
    if payload.get("status_category"):
        lines.append(f"Status category: {payload.get('status_category')}")
    if payload.get("priority"):
        lines.append(f"Priority: {payload.get('priority')}")
    if payload.get("url"):
        lines.append(f"URL: {payload.get('url')}")
    if payload.get("qualification_rule"):
        lines.append(f"Qualification rule: {payload.get('qualification_rule')}")

    reporter = payload.get("reporter")
    assignee = payload.get("assignee")
    if reporter:
        lines.append(f"Reporter: {_actor_label(reporter)}")
    if assignee:
        lines.append(f"Assignee: {_actor_label(assignee)}")

    for field_name, label in (
        ("created_at", "Created"),
        ("updated_at", "Updated"),
        ("resolved_at", "Resolved"),
    ):
        field_value = payload.get(field_name)
        if field_value:
            lines.append(f"{label}: {field_value}")

    if isinstance(labels, list):
        label_values = [str(label).strip() for label in labels if str(label).strip()]
        if label_values:
            lines.append(f"Labels: {', '.join(label_values)}")

    components = payload.get("components")
    if isinstance(components, list) and components:
        lines.append(f"Components: {', '.join(str(item) for item in components if str(item).strip())}")
    fix_versions = payload.get("fix_versions")
    if isinstance(fix_versions, list) and fix_versions:
        lines.append(
            f"Fix versions: {', '.join(str(item) for item in fix_versions if str(item).strip())}"
        )

    linked_issues = payload.get("linked_issues")
    if isinstance(linked_issues, list) and linked_issues:
        lines.append("")
        lines.append("## Linked Issues")
        for linked in linked_issues:
            if not isinstance(linked, dict):
                continue
            key = str(linked.get("key") or "").strip()
            relation = str(linked.get("relation") or "linked").strip()
            summary = str(linked.get("summary") or "").strip()
            entry = f"- {relation}: {key}" if key else f"- {relation}"
            if summary:
                entry += f" — {summary}"
            lines.append(entry)

    if description:
        lines.append("")
        lines.append("## Description")
        lines.append(description)

    if isinstance(comments, list) and comments:
        lines.append("")
        lines.append("## Discussion")
        sorted_comments = sorted(
            [item for item in comments if isinstance(item, dict)],
            key=lambda row: (
                str(row.get("created_at") or ""),
                str(row.get("entry_id") or ""),
            ),
        )
        for row in sorted_comments:
            timestamp = str(row.get("created_at") or "")
            author = _actor_label(row.get("author"))
            body = str(row.get("body") or "").strip()
            lines.append(f"- [{timestamp}] comment by {author}")
            if body:
                lines.append(f"  {body}")

    rendered = "\n".join(lines).strip()
    return rendered or content


class JiraProcessorWorker(BaseProcessorWorker):
    """Processor worker for Jira issue documents."""

    PROVIDER = "jira"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        return _jira_discourse_to_text(doc, content)

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        return _jira_discourse_to_text(doc, content)
