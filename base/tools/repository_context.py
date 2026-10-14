"""Canonical, repository-scoped context retrieval for MCP clients."""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any, Iterable

from base.tools.repository_context_notes import normalize_repository
from base.tools.s3 import read_json_object

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
_SEMANTIC_TYPES = {"Decision", "Constraint", "Insight"}
_STATUS_WEIGHT = {"verified": 3, "documented": 2, "inferred": 1}


def _normalize_repository_input(repository: str) -> tuple[str, str]:
    if not isinstance(repository, str):
        raise ValueError("repository must be a string in owner/repo form")
    parts = repository.strip().split("/")
    if len(parts) != 2:
        raise ValueError("repository must be in owner/repo form")
    _, _, normalized, scope = normalize_repository(parts[0], parts[1])
    return normalized, scope


def _clean_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def assess_commit_policy(
    *,
    profile_sha: str,
    local_head: str | None = None,
    merge_base: str | None = None,
    working_tree_dirty: bool = False,
) -> dict[str, Any]:
    """Classify a checkout relative to the canonical profile commit."""
    profile = _clean_sha(profile_sha) or ""
    head = _clean_sha(local_head)
    base = _clean_sha(merge_base)
    relationship_guidance = ""

    if head is not None and head == profile:
        relationship = "current"
        explanation = "Local HEAD exactly matches the canonical profile commit."
        stale_warning = None
    elif head is not None and head != profile and base == profile:
        relationship = "feature_branch_ahead"
        explanation = (
            "The feature branch is ahead of the profile commit; the profile "
            "remains a valid baseline for unchanged code."
        )
        stale_warning = None
    elif head is not None and head == base and head != profile:
        relationship = "local_behind"
        explanation = "The local checkout is behind the canonical profile commit."
        stale_warning = (
            "Local checkout is behind the canonical profile commit; update or "
            "compare locally before relying on implementation details."
        )
    elif head is not None and head != profile and base is None:
        relationship = "diverged_or_unknown"
        explanation = (
            "Local HEAD differs from the profile commit, but no merge-base was "
            "supplied, so the relationship cannot yet be classified."
        )
        stale_warning = (
            "Profile freshness against this checkout is unknown until the "
            "merge-base is computed."
        )
        relationship_guidance = (
            "Run or compute git merge-base between local HEAD and the profiled "
            "upstream ref, then reassess. Do not discard local feature-branch "
            "commits while reconciling the profile baseline."
        )
    else:
        relationship = "diverged_or_unknown"
        explanation = (
            "The checkout relationship to the profile commit is diverged or "
            "cannot be established from the supplied git state."
        )
        stale_warning = (
            "Profile freshness against this checkout is unknown; inspect commit "
            "history and implementation locally."
        )

    dirty_override = bool(working_tree_dirty)
    if dirty_override:
        guidance = (
            "Dirty working-tree changes override profile implementation claims. "
            "Inspect the local diff and changed files before acting; preserve "
            "uncommitted work and verify it locally."
        )
    else:
        guidance = (
            "Treat the profile as shared baseline context and verify task-relevant "
            "implementation and commands in the local checkout."
        )
    if relationship_guidance:
        guidance = f"{guidance} {relationship_guidance}"

    return {
        "relationship": relationship,
        "profile_commit": profile,
        "local_head": head,
        "merge_base": base,
        "dirty_override": dirty_override,
        "explanation": explanation,
        "stale_warning": stale_warning,
        "guidance": guidance,
        "shared_context_policy": (
            "Do not derive or replace shared repository context from unmerged "
            "local changes."
        ),
    }


def _tokens(*values: Any) -> set[str]:
    text = " ".join(str(value or "") for value in values)
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 1}


def _entity_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "unknown"


def _citation_paths(citations: Any) -> list[str]:
    if not isinstance(citations, list):
        return []
    return [
        str(citation["path"])
        for citation in citations
        if isinstance(citation, dict) and citation.get("path")
    ]


def _command_score(
    command: dict[str, Any],
    *,
    task_tokens: set[str],
    changed_paths: list[str],
) -> tuple[int, str]:
    citations = command.get("citations") or []
    citation_paths = _citation_paths(citations)
    command_tokens = _tokens(
        command.get("command"),
        command.get("purpose"),
        *citation_paths,
    )
    overlap = len(task_tokens & command_tokens)
    path_score = 0
    for changed in changed_paths:
        changed_lower = changed.lower()
        changed_root = changed_lower.split("/", 1)[0]
        for cited in citation_paths:
            cited_lower = cited.lower()
            if cited_lower == changed_lower:
                path_score += 8
            elif changed_root and cited_lower.startswith(f"{changed_root}/"):
                path_score += 4
        command_text = str(command.get("command") or "").lower()
        if changed_root and changed_root in command_text:
            path_score += 3
    status_score = _STATUS_WEIGHT.get(str(command.get("status") or ""), 0)
    return (path_score + overlap * 2 + status_score, str(command.get("command") or ""))


def _command_candidates(
    profile: dict[str, Any],
    section: str,
    *,
    task: str,
    changed_paths: list[str],
    limit: int = 6,
) -> list[dict[str, Any]]:
    practices = profile.get("practices")
    if not isinstance(practices, dict):
        return []
    raw_section = practices.get(section)
    if not isinstance(raw_section, dict):
        return []
    commands = raw_section.get("commands")
    if not isinstance(commands, list):
        return []
    candidates = [dict(command) for command in commands if isinstance(command, dict)]
    task_tokens = _tokens(task, *changed_paths)
    candidates.sort(
        key=lambda command: _command_score(
            command,
            task_tokens=task_tokens,
            changed_paths=changed_paths,
        ),
        reverse=True,
    )
    return candidates[:limit]


def _semantic_content(props: dict[str, Any], node_type: str) -> str:
    field = {
        "Decision": "rationale",
        "Constraint": "description",
        "Insight": "body",
    }.get(node_type)
    return str(props.get(field) or props.get("label") or "").strip()


def _semantic_context(
    client: Any,
    *,
    repository: str,
    source_scope: str,
    task: str,
    changed_paths: list[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    owner, repo = repository.split("/", 1)
    repository_id = f"entity:repository:{_entity_slug(owner)}:{_entity_slug(repo)}"
    rows = client.run(
        """
        MATCH (n)-[about:ABOUT]->(repository:Entity {id: $repository_id})
        WHERE labels(n)[0] IN ['Decision', 'Constraint', 'Insight']
          AND n.source_scope = $source_scope
          AND about.source_scope = $source_scope
          AND n.superseded_by IS NULL
        RETURN properties(n) AS props, labels(n)[0] AS node_type
        LIMIT 100
        """,
        {"repository_id": repository_id, "source_scope": source_scope},
    )
    task_tokens = _tokens(task, *changed_paths)
    output: list[tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        node_type = str(row.get("node_type") or "")
        props = dict(row.get("props") or {})
        if node_type not in _SEMANTIC_TYPES:
            continue
        content = _semantic_content(props, node_type)
        item = {
            "id": str(props.get("id") or ""),
            "type": node_type,
            "label": str(props.get("label") or props.get("id") or ""),
            "content": content,
        }
        source_doc = props.get("source_doc") or props.get("source_doc_id")
        if source_doc:
            item["source_doc"] = str(source_doc)
        score = len(task_tokens & _tokens(item["label"], content)) * 2
        output.append((score, item["id"], item))
    output.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _, _, item in output[:limit]]


def _as_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _collect_cited_paths(parts: Iterable[Any]) -> list[str]:
    paths: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("source_type") == "file" and value.get("path"):
                paths.add(str(value["path"]))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for part in parts:
        visit(part)
    return sorted(paths)


async def get_repository_context(
    *,
    pool: Any,
    client: Any,
    org_id: str,
    allowed_scopes: list[str] | None,
    repository: str,
    task: str,
    local_head: str | None = None,
    merge_base: str | None = None,
    working_tree_dirty: bool = False,
    changed_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Load the canonical profile without vector ranking and tailor it locally."""
    normalized_repository, requested_scope = _normalize_repository_input(repository)
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if not isinstance(working_tree_dirty, bool):
        raise ValueError("working_tree_dirty must be a boolean")
    if changed_paths is not None and not isinstance(changed_paths, list):
        raise ValueError("changed_paths must be an array of strings")
    if changed_paths is not None and any(
        not isinstance(path, str) for path in changed_paths
    ):
        raise ValueError("changed_paths must contain only strings")
    normalized_paths = sorted(
        {
            str(path).strip()
            for path in (changed_paths or [])
            if isinstance(path, str) and path.strip()
        }
    )

    scoped_values: list[str] | None = None
    if allowed_scopes is not None:
        matching_scope = next(
            (
                str(scope)
                for scope in allowed_scopes
                if str(scope).lower() == requested_scope.lower()
            ),
            None,
        )
        if matching_scope is None:
            raise PermissionError(
                f"caller does not have access to repository {normalized_repository}"
            )
        scoped_values = [matching_scope]

    row = await pool.fetchrow(
        """
        SELECT
            profile.repository,
            profile.ref,
            profile.commit_sha,
            profile.source_scope,
            profile.last_generated_at,
            document.s3_key
        FROM github_repository_profiles AS profile
        JOIN documents AS document ON document.id = profile.profile_doc_id
        WHERE profile.org_id = $1
          AND lower(profile.repository) = lower($2)
          AND profile.source_scope LIKE 'scope:github:%'
          AND ($3::text[] IS NULL OR profile.source_scope = ANY($3::text[]))
          AND profile.generation_state = 'completed'
          AND profile.profile_doc_id IS NOT NULL
        ORDER BY profile.last_generated_at DESC NULLS LAST, profile.ref, profile.id
        LIMIT 1
        """,
        org_id,
        normalized_repository,
        scoped_values,
    )
    if row is None:
        raise LookupError(
            f"no completed repository profile found for {normalized_repository}"
        )

    source_scope = str(row["source_scope"])
    if not source_scope.lower().startswith("scope:github:"):
        raise PermissionError("repository profile is not in an allowed GitHub scope")
    if allowed_scopes is not None and all(
        source_scope.lower() != str(scope).lower() for scope in allowed_scopes
    ):
        raise PermissionError(
            f"caller does not have access to repository {normalized_repository}"
        )

    raw_profile = await read_json_object(str(row["s3_key"]))
    nested_profile = raw_profile.get("profile")
    profile = (
        dict(nested_profile)
        if isinstance(nested_profile, dict)
        else dict(raw_profile)
    )
    profile_repository = str(profile.get("repository") or "").lower()
    if profile_repository != normalized_repository:
        raise ValueError("stored profile repository does not match canonical index")
    if str(profile.get("ref") or "") != str(row["ref"]):
        raise ValueError("stored profile ref does not match canonical index")
    profile_commit = str(profile.get("commit_sha") or row["commit_sha"] or "")
    if profile_commit != str(row["commit_sha"]):
        raise ValueError("stored profile commit does not match canonical index")

    purpose = dict(profile.get("purpose") or {})
    architecture = dict(profile.get("architecture") or {})
    components = [
        {
            "name": str(component.get("name") or ""),
            "description": str(component.get("description") or ""),
            "responsibilities": list(component.get("responsibilities") or []),
            "technologies": list(component.get("technologies") or []),
            "citations": list(component.get("citations") or []),
        }
        for component in architecture.get("components") or []
        if isinstance(component, dict)
    ][:8]
    test_candidates = _command_candidates(
        profile,
        "testing",
        task=task,
        changed_paths=normalized_paths,
    )
    ci_candidates = _command_candidates(
        profile,
        "ci",
        task=task,
        changed_paths=normalized_paths,
    )
    qa_candidates = _command_candidates(
        profile,
        "qa",
        task=task,
        changed_paths=normalized_paths,
    )
    semantic_context = _semantic_context(
        client,
        repository=normalized_repository,
        source_scope=source_scope,
        task=task,
        changed_paths=normalized_paths,
    )
    commit_policy = assess_commit_policy(
        profile_sha=profile_commit,
        local_head=local_head,
        merge_base=merge_base,
        working_tree_dirty=working_tree_dirty,
    )
    freshness = dict(profile.get("freshness") or {})
    freshness["generated_at"] = _as_iso(row["last_generated_at"])

    result = {
        "repository": normalized_repository,
        "ref": str(profile.get("ref") or row["ref"]),
        "profile_commit": profile_commit,
        "generated_at": _as_iso(row["last_generated_at"]),
        "task": task.strip(),
        "changed_paths": normalized_paths,
        "purpose": {
            "summary": str(purpose.get("summary") or ""),
            "citations": list(purpose.get("citations") or []),
        },
        "architecture": {
            "overview": str(architecture.get("overview") or ""),
            "citations": list(architecture.get("citations") or []),
        },
        "components": components,
        "semantic_context": semantic_context,
        "test_candidates": test_candidates,
        "ci_candidates": ci_candidates,
        "qa_candidates": qa_candidates,
        "freshness": freshness,
        "commit_policy": commit_policy,
        "local_verification": (
            "Inspect task-relevant files and any dirty changes, then run the "
            "applicable candidates locally. Command status describes profile "
            "evidence, not the current checkout."
        ),
    }
    result["cited_paths"] = _collect_cited_paths(
        [
            result["purpose"],
            result["architecture"],
            components,
            test_candidates,
            ci_candidates,
            qa_candidates,
        ]
    )
    return result


def _format_citation(citation: dict[str, Any]) -> str:
    source_type = str(citation.get("source_type") or "")
    if source_type == "file" and citation.get("path"):
        path = str(citation["path"])
        start = citation.get("line_start")
        end = citation.get("line_end")
        blob = citation.get("blob_sha")
        line_part = f":{start}-{end}" if start is not None and end is not None else ""
        blob_part = f"@{blob}" if blob else ""
        return _markdown_code(f"{path}{line_part}{blob_part}")
    if source_type in {"issue", "pull_request"} and citation.get("number"):
        return (
            f"{_markdown_text(source_type)} "
            f"#{_markdown_text(citation['number'])}"
        )
    if source_type == "context_note" and citation.get("note_id"):
        return f"context note {_markdown_code(citation['note_id'])}"
    if citation.get("url"):
        return _markdown_text(citation["url"])
    return _markdown_text(source_type or "repository evidence")


def _citation_suffix(item: dict[str, Any]) -> str:
    citations = item.get("citations")
    if not isinstance(citations, list) or not citations:
        return ""
    rendered = [
        _format_citation(citation)
        for citation in citations
        if isinstance(citation, dict)
    ]
    return f" ({'; '.join(rendered)})" if rendered else ""


def _markdown_text(value: Any) -> str:
    """Render untrusted content as one inert Markdown text line."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    escaped = html.escape(text, quote=True).replace("\\", "\\\\")
    return re.sub(r"([`*_[\]{}()#!>|])", r"\\\1", escaped)


def _markdown_code(value: Any) -> str:
    """Render an untrusted scalar in a delimiter-safe inline code span."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    escaped = html.escape(text, quote=True)
    escaped = (
        escaped.replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
    )
    longest = max((len(run) for run in re.findall(r"`+", escaped)), default=0)
    delimiter = "`" * (longest + 1)
    padding = " " if escaped.startswith("`") or escaped.endswith("`") else ""
    return f"{delimiter}{padding}{escaped}{padding}{delimiter}"


def render_repository_profile_markdown(context: dict[str, Any]) -> str:
    """Render deterministic, citation-bearing Markdown without repository writes."""
    repository = _markdown_text(
        context.get("repository") or "Unknown repository"
    )
    policy = dict(context.get("commit_policy") or {})
    lines = [
        f"# {repository} Repository Context",
        "",
        f"Generated: {_markdown_text(context.get('generated_at') or 'unknown')}",
        f"Ref: {_markdown_code(context.get('ref') or 'unknown')}",
        f"Commit: {_markdown_code(context.get('profile_commit') or 'unknown')}",
    ]
    stale_warning = policy.get("stale_warning")
    if stale_warning:
        lines.extend(["", f"> **STALE WARNING:** {_markdown_text(stale_warning)}"])
    if policy.get("dirty_override"):
        lines.extend(
            [
                "",
                "> **LOCAL OVERRIDE:** Dirty changes override profile implementation "
                "claims and must be inspected locally.",
            ]
        )

    purpose = dict(context.get("purpose") or {})
    architecture = dict(context.get("architecture") or {})
    lines.extend(
        [
            "",
            "## Purpose",
            f"{_markdown_text(purpose.get('summary') or 'Not documented.')}"
            f"{_citation_suffix(purpose)}",
            "",
            "## Architecture",
            f"{_markdown_text(architecture.get('overview') or 'Not documented.')}"
            f"{_citation_suffix(architecture)}",
            "",
            "## Components",
        ]
    )
    components = context.get("components") or []
    if components:
        for component in components:
            if not isinstance(component, dict):
                continue
            lines.append(
                f"- **{_markdown_text(component.get('name') or 'component')}:** "
                f"{_markdown_text(component.get('description') or '')}"
                f"{_citation_suffix(component)}"
            )
    else:
        lines.append("- None documented.")

    for title, key in (
        ("Test Candidates", "test_candidates"),
        ("CI Candidates", "ci_candidates"),
        ("QA Candidates", "qa_candidates"),
    ):
        lines.extend(["", f"## {title}"])
        candidates = context.get(key) or []
        if not candidates:
            lines.append("- None documented.")
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            lines.append(
                f"- {_markdown_code(candidate.get('command') or '')} "
                f"[{_markdown_text(candidate.get('status') or 'unknown')}] — "
                f"{_markdown_text(candidate.get('purpose') or '')}"
                f"{_citation_suffix(candidate)}"
            )

    lines.extend(["", "## Repository Decisions, Constraints, and Insights"])
    semantic = context.get("semantic_context") or []
    if semantic:
        for item in semantic:
            if not isinstance(item, dict):
                continue
            source = (
                f" (source {_markdown_code(item['source_doc'])})"
                if item.get("source_doc")
                else ""
            )
            lines.append(
                f"- **{_markdown_text(item.get('type'))}: "
                f"{_markdown_text(item.get('label'))}** — "
                f"{_markdown_text(item.get('content') or '')}{source}"
            )
    else:
        lines.append("- No task-relevant repository-scoped items found.")

    lines.extend(
        [
            "",
            "## Freshness and Commit Policy",
            f"- Relationship: "
            f"{_markdown_code(policy.get('relationship') or 'diverged_or_unknown')}",
            f"- {_markdown_text(policy.get('explanation') or '')}",
            f"- {_markdown_text(policy.get('guidance') or '')}",
            f"- {_markdown_text(policy.get('shared_context_policy') or '')}",
            f"- {_markdown_text(context.get('local_verification') or '')}",
            "",
            "## Cited Paths",
        ]
    )
    cited_paths = context.get("cited_paths") or []
    lines.extend(
        [f"- {_markdown_code(path)}" for path in cited_paths]
        if cited_paths
        else ["- No file paths cited."]
    )
    return "\n".join(lines).rstrip() + "\n"
