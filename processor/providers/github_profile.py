"""GitHub repository-profile document processor."""

from __future__ import annotations

import html
import json
import re
from typing import Iterable

from processor.schemas.repository_profile import CitationV1, RepoProfileV1
from processor.schemas.scoring import ScoringResult
from processor.state_machine import DocumentRecord


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+.!|>~-])")


def _prose(value: object) -> str:
    """Render untrusted text without allowing Markdown or HTML structure."""
    flattened = " ".join(str(value).replace("\r", "\n").splitlines())
    return _MARKDOWN_SPECIAL.sub(r"\\\1", html.escape(flattened, quote=False))


def _code(value: object) -> str:
    """Render an untrusted value safely inside a backtick code span."""
    flattened = " ".join(str(value).replace("\r", "\n").splitlines())
    return html.escape(flattened, quote=False).replace("`", "&#96;")


def _citation_label(citation: CitationV1) -> str:
    if citation.source_type == "file":
        lines = (
            f"#L{citation.line_start}-L{citation.line_end}"
            if citation.line_start is not None
            else ""
        )
        return f"`{_code(citation.path)}{lines}`"
    if citation.source_type in {"issue", "pull_request"}:
        label = "issue" if citation.source_type == "issue" else "pull request"
        return f"{label} #{citation.number}"
    if citation.source_type == "context_note":
        return f"context note `{_code(citation.note_id)}`"
    return citation.source_type.replace("_", " ")


def _citations(citations: Iterable[CitationV1]) -> str:
    labels = [_citation_label(citation) for citation in citations]
    return f" Sources: {', '.join(labels)}." if labels else ""


def _statement_lines(items) -> list[str]:
    return [
        f"- {_prose(item.summary)}{_citations(item.citations)}"
        for item in items
    ]


def render_repository_profile_markdown(profile: RepoProfileV1) -> str:
    """Render canonical profile JSON into extraction- and retrieval-friendly markdown."""
    lines = [
        f"# {_prose(profile.repository)} Repository Profile",
        "",
        f"Ref: `{_code(profile.ref)}`",
        f"Commit: `{_code(profile.commit_sha)}`",
        "",
        "## Purpose",
        _prose(profile.purpose.summary) + _citations(profile.purpose.citations),
    ]
    if profile.purpose.audiences:
        lines.extend(
            ["", "### Audiences", *[f"- {_prose(item)}" for item in profile.purpose.audiences]]
        )
    if profile.purpose.capabilities:
        lines.extend(["", "### Capabilities", *_statement_lines(profile.purpose.capabilities)])

    if profile.boundaries:
        lines.extend(["", "## Boundaries", *_statement_lines(profile.boundaries)])

    lines.extend(
        [
            "",
            "## Architecture",
            _prose(profile.architecture.overview)
            + _citations(profile.architecture.citations),
        ]
    )
    for component in profile.architecture.components:
        lines.append(
            f"- **{_prose(component.name)}:** {_prose(component.description)}"
            f"{_citations(component.citations)}"
        )
        if component.responsibilities:
            lines.append(
                "  Responsibilities: "
                + ", ".join(_prose(item) for item in component.responsibilities)
            )
        if component.technologies:
            lines.append(
                "  Technologies: "
                + ", ".join(_prose(item) for item in component.technologies)
            )
    if profile.architecture.data_flows:
        lines.append("")
        lines.append("### Data Flows")
        for flow in profile.architecture.data_flows:
            lines.append(
                f"- **{_prose(flow.source)} → {_prose(flow.destination)}:** "
                f"{_prose(flow.description)}"
                f"{_citations(flow.citations)}"
            )

    interface_groups = (
        ("Interfaces", profile.interfaces.interfaces),
        ("External Services", profile.interfaces.external_services),
        ("Datastores", profile.interfaces.datastores),
        ("Queues", profile.interfaces.queues),
    )
    lines.extend(["", "## Interfaces and Integrations"])
    for heading, items in interface_groups:
        if not items:
            continue
        lines.append(f"### {heading}")
        for item in items:
            kind = (
                f" ({_prose(item.protocol_or_kind)})"
                if item.protocol_or_kind
                else ""
            )
            lines.append(
                f"- **{_prose(item.name)}{kind}:** {_prose(item.description)}"
                f"{_citations(item.citations)}"
            )

    lines.extend(["", "## Runtime and Operations"])
    for heading, items in (
        ("Runtime", profile.runtime.runtime),
        ("Deployment", profile.runtime.deployment),
        ("Ownership", profile.runtime.ownership),
    ):
        if items:
            lines.extend([f"### {heading}", *_statement_lines(items)])

    latest_release = profile.release.latest_release
    latest_deployment = profile.release.latest_successful_deployment
    lines.extend(["", "## Release and Production"])
    if latest_release is None and latest_deployment is None:
        lines.append("- No release or successful production deployment was collected.")
    if latest_release is not None:
        label = latest_release.tag_name or latest_release.name or "unnamed release"
        published = latest_release.published_at or latest_release.created_at or "unknown date"
        lines.append(
            f"- Latest release: **{_prose(label)}**, published {_prose(published)}."
            f"{_citations(profile.release.citations)}"
        )
    if latest_deployment is not None:
        environment = latest_deployment.environment or "unspecified environment"
        deployed_at = (
            latest_deployment.updated_at
            or latest_deployment.created_at
            or "unknown date"
        )
        lines.append(
            f"- Latest successful deployment: **{_prose(environment)}** at "
            f"{_prose(deployed_at)} "
            f"(commit `{_code(latest_deployment.sha or 'unknown')}`)."
            f"{_citations(profile.release.citations)}"
        )

    for heading, section in (
        ("Local Development", profile.practices.local_development),
        ("Contributing", profile.practices.contributing),
        ("Testing", profile.practices.testing),
        ("CI", profile.practices.ci),
        ("QA", profile.practices.qa),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                _prose(section.summary) + _citations(section.citations),
            ]
        )
        for command in section.commands:
            lines.append(
                f"- `{_code(command.command)}` — {_prose(command.purpose)} "
                f"[{_prose(command.status)}]"
                f"{_citations(command.citations)}"
            )

    if profile.rationale:
        lines.extend(["", "## Rationale", *_statement_lines(profile.rationale)])
    if profile.contradictions:
        lines.extend(["", "## Contradictions"])
        for contradiction in profile.contradictions:
            lines.append(
                f"- **{_prose(contradiction.topic)}:** "
                f"{'; '.join(_prose(claim) for claim in contradiction.claims)}"
                f"{_citations(contradiction.citations)}"
            )
    if profile.unknowns:
        lines.extend(["", "## Unknowns"])
        lines.extend(
            f"- **{_prose(unknown.question)}** Impact: {_prose(unknown.impact)}"
            for unknown in profile.unknowns
        )

    lines.extend(
        [
            "",
            "## Confidence and Freshness",
            (
                f"Overall confidence: {profile.confidence.overall:.2f}. "
                f"{_prose(profile.confidence.rationale)}"
            ),
            (
                f"Freshness: {_prose(profile.freshness.assessment)} "
                f"(evidence `{_code(profile.freshness.evidence_hash)}`)."
            ),
        ]
    )
    return "\n".join(lines).strip()


def _parse_profile(content: str) -> RepoProfileV1:
    payload = json.loads(content)
    return RepoProfileV1.model_validate(payload)


def __getattr__(name: str):
    """Load the processor worker only when indexing, not when rendering markdown."""
    if name != "GithubProfileProcessorWorker":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from processor.worker import BaseProcessorWorker

    class GithubProfileProcessorWorker(BaseProcessorWorker):
        """Index synthesized repository profiles without generic admission scoring."""

        PROVIDER = "github-profile"

        async def score_content(
            self,
            doc: DocumentRecord,
            content: str,
        ) -> ScoringResult:
            repository = doc.parent_ref or doc.source_ref.split("/_profile/", 1)[0]
            return ScoringResult(
                signal_score=1.0,
                signal_types=["decision", "constraint", "insight", "signal"],
                reasoning=(
                    "Canonical repository profile is deterministically admitted for indexing."
                ),
                suggested_topics=[
                    repository,
                    "repository architecture",
                    "development practices",
                    "operations",
                ],
            )

        async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
            return render_repository_profile_markdown(_parse_profile(content))

        async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
            return render_repository_profile_markdown(_parse_profile(content))

    globals()["GithubProfileProcessorWorker"] = GithubProfileProcessorWorker
    return GithubProfileProcessorWorker
