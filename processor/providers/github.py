"""
processor/providers/github.py — GitHub discourse processor worker.

Processes documents with source_provider='github'.

GitHub discourse payloads are structured JSON (issue/PR body + threaded
discussion metadata). This worker renders them into extraction-friendly text so
the LLM sees chronology, participants, and thread context in one place.
"""

from __future__ import annotations

import json
from typing import Any

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker


def _actor_label(actor: Any) -> str:
    if not isinstance(actor, dict):
        return "unknown"
    login = str(actor.get("login") or "unknown").strip()
    actor_type = str(actor.get("type") or "").strip()
    if actor_type:
        return f"{login} ({actor_type})"
    return login


def _entry_created_at(entry: dict[str, Any]) -> str:
    return str(entry.get("created_at") or "")


def _entry_sort_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        _entry_created_at(entry),
        str(entry.get("entry_type") or ""),
        str(entry.get("entry_id") or ""),
    )


def _render_thread_entries(entries: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for entry in sorted(entries, key=_entry_sort_key):
        entry_type = str(entry.get("entry_type") or "entry")
        created_at = str(entry.get("created_at") or "")
        author = _actor_label(entry.get("author"))
        body = str(entry.get("body") or "").strip()
        prefix = f"- [{created_at}] {entry_type} by {author}"

        path = entry.get("thread_path")
        line = entry.get("thread_line")
        if path:
            location = str(path)
            if line is not None:
                location = f"{location}:{line}"
            prefix += f" ({location})"

        parent_id = entry.get("parent_review_comment_id")
        if parent_id:
            prefix += f" [reply_to={parent_id}]"

        lines.append(prefix)
        if body:
            lines.append(f"  {body}")
    return lines


def _render_issue_comments(comments: list[dict[str, Any]]) -> list[str]:
    entries: list[dict[str, Any]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        entries.append({
            "entry_type": "issue_comment",
            "entry_id": comment.get("entry_id"),
            "created_at": comment.get("created_at"),
            "author": comment.get("author"),
            "body": comment.get("body"),
        })
    return _render_thread_entries(entries)


def _github_discourse_to_text(doc: DocumentRecord, content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        if doc.title and not content.startswith(doc.title) and not content.startswith(f"# {doc.title}"):
            return f"# {doc.title}\n\n{content}"
        return content

    if not isinstance(payload, dict):
        return content

    kind = str(payload.get("kind") or "github_item")
    title = str(payload.get("title") or doc.title or "GitHub item").strip()
    repository = str(payload.get("repository") or "").strip()
    number = payload.get("number")
    url = str(payload.get("url") or "").strip()
    state = str(payload.get("state") or "").strip()
    state_reason = str(payload.get("state_reason") or "").strip()
    qualification = str(payload.get("qualification_rule") or "").strip()
    human_comment_count = payload.get("human_comment_count")

    lines: list[str] = [f"# {title}"]
    lines.append(f"Type: {kind}")
    if repository:
        lines.append(f"Repository: {repository}")
    if number is not None:
        lines.append(f"Number: {number}")
    if url:
        lines.append(f"URL: {url}")
    if state:
        lines.append(f"State: {state}")
    if state_reason:
        lines.append(f"State reason: {state_reason}")
    if human_comment_count is not None:
        lines.append(f"Human comment count: {human_comment_count}")
    if qualification:
        lines.append(f"Qualification rule: {qualification}")

    labels = payload.get("labels")
    if isinstance(labels, list):
        names = [
            str(label.get("name")).strip()
            for label in labels
            if isinstance(label, dict) and str(label.get("name") or "").strip()
        ]
        if names:
            lines.append(f"Labels: {', '.join(names)}")

    milestone = payload.get("milestone")
    if isinstance(milestone, dict) and str(milestone.get("title") or "").strip():
        lines.append(f"Milestone: {milestone.get('title')}")

    linked_issues = payload.get("linked_issues")
    if isinstance(linked_issues, list) and linked_issues:
        lines.append("")
        lines.append("## Linked Issues")
        for issue in linked_issues:
            if not isinstance(issue, dict):
                continue
            number_value = issue.get("number")
            title_value = str(issue.get("title") or "").strip()
            url_value = str(issue.get("url") or "").strip()
            descriptor = f"- #{number_value}" if number_value is not None else "- Issue"
            if title_value:
                descriptor += f": {title_value}"
            if url_value:
                descriptor += f" ({url_value})"
            lines.append(descriptor)

    check_runs = payload.get("check_runs")
    if isinstance(check_runs, list) and check_runs:
        lines.append("")
        lines.append("## Check Runs")
        for run in check_runs:
            if not isinstance(run, dict):
                continue
            name = str(run.get("name") or "check").strip()
            status = str(run.get("status") or "unknown").strip()
            conclusion = str(run.get("conclusion") or "").strip()
            url = str(run.get("html_url") or "").strip()
            line = f"- {name}: status={status}"
            if conclusion:
                line += f", conclusion={conclusion}"
            if url:
                line += f" ({url})"
            lines.append(line)

    workflow_runs = payload.get("workflow_runs")
    if isinstance(workflow_runs, list) and workflow_runs:
        lines.append("")
        lines.append("## Workflow Runs")
        for run in workflow_runs:
            if not isinstance(run, dict):
                continue
            name = str(run.get("name") or "workflow").strip()
            status = str(run.get("status") or "unknown").strip()
            conclusion = str(run.get("conclusion") or "").strip()
            run_number = run.get("run_number")
            url = str(run.get("html_url") or "").strip()
            line = f"- {name}"
            if run_number is not None:
                line += f" #{run_number}"
            line += f": status={status}"
            if conclusion:
                line += f", conclusion={conclusion}"
            if url:
                line += f" ({url})"
            lines.append(line)

    author = payload.get("author")
    if author:
        lines.append(f"Author: {_actor_label(author)}")

    for field_name, label in (
        ("created_at", "Created"),
        ("updated_at", "Updated"),
        ("closed_at", "Closed"),
        ("merged_at", "Merged"),
    ):
        value = payload.get(field_name)
        if value:
            lines.append(f"{label}: {value}")

    body = str(payload.get("body") or "").strip()
    if body:
        lines.append("")
        lines.append("## Body")
        lines.append(body)

    thread_entries_raw = payload.get("thread_entries")
    if isinstance(thread_entries_raw, list) and thread_entries_raw:
        rendered = _render_thread_entries(
            [e for e in thread_entries_raw if isinstance(e, dict)]
        )
    else:
        comments = payload.get("comments")
        rendered = _render_issue_comments(comments if isinstance(comments, list) else [])

    if rendered:
        lines.append("")
        lines.append("## Discussion")
        lines.extend(rendered)

    rendered_text = "\n".join(lines).strip()
    return rendered_text or content


class GithubDiscourseProcessorWorker(BaseProcessorWorker):
    """Processor worker for GitHub discourse documents."""

    PROVIDER = "github"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        return _github_discourse_to_text(doc, content)

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        return _github_discourse_to_text(doc, content)
