"""Strict, citation-bearing repository profile schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


CitationSource = Literal[
    "file",
    "issue",
    "pull_request",
    "context_note",
    "repository_metadata",
    "tree",
    "release",
    "deployment",
]


class CitationV1(_StrictModel):
    source_type: CitationSource
    path: str | None = None
    blob_sha: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    url: str | None = None
    number: int | None = Field(default=None, ge=1)
    note_id: str | None = None

    @model_validator(mode="after")
    def validate_source_locator(self) -> CitationV1:
        if self.source_type == "file":
            if not (
                self.path
                and self.blob_sha
                and self.line_start is not None
                and self.line_end is not None
            ):
                raise ValueError(
                    "file citations require path, blob_sha, line_start, and line_end"
                )
            if self.line_end < self.line_start:
                raise ValueError("citation line_end must be >= line_start")
        elif self.source_type in {"issue", "pull_request"} and self.number is None:
            raise ValueError(f"{self.source_type} citations require number")
        elif self.source_type == "context_note" and not self.note_id:
            raise ValueError("context_note citations require note_id")
        return self


class EvidenceStatementV1(_StrictModel):
    summary: str = Field(min_length=1)
    citations: list[CitationV1] = Field(min_length=1)


class PurposeV1(_StrictModel):
    summary: str = Field(min_length=1)
    audiences: list[str]
    capabilities: list[EvidenceStatementV1]
    citations: list[CitationV1] = Field(min_length=1)


class ComponentV1(_StrictModel):
    name: str = Field(
        min_length=1,
        description=(
            "Concrete module, package, entrypoint, or process name from evidence "
            "(e.g. bot.py, src/storage.py). Do not use generic folder labels."
        ),
    )
    description: str = Field(
        min_length=1,
        description=(
            "How this part works at runtime: main types or functions, what it "
            "owns, and which other components it calls. Two to four sentences "
            "grounded in cited files."
        ),
    )
    responsibilities: list[str]
    technologies: list[str]
    citations: list[CitationV1] = Field(min_length=1)


class DataFlowV1(_StrictModel):
    source: str = Field(
        min_length=1,
        description="Component, actor, or system that initiates the handoff.",
    )
    destination: str = Field(
        min_length=1,
        description="Component, actor, or system that receives it.",
    )
    description: str = Field(
        min_length=1,
        description=(
            "What is passed (command, event, record, HTTP call) and when this "
            "path runs. Cite the call site or producer/consumer files."
        ),
    )
    citations: list[CitationV1] = Field(min_length=1)


class ArchitectureV1(_StrictModel):
    overview: str = Field(
        min_length=1,
        description=(
            "A navigation briefing: process model, where execution starts, major "
            "packages/modules, and how a typical request or job flows. Use "
            "specific file names. Do not summarize as 'N files in src/'."
        ),
    )
    components: list[ComponentV1]
    data_flows: list[DataFlowV1]
    citations: list[CitationV1] = Field(min_length=1)


class IntegrationV1(_StrictModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    protocol_or_kind: str | None
    citations: list[CitationV1] = Field(min_length=1)


class InterfacesV1(_StrictModel):
    interfaces: list[IntegrationV1]
    external_services: list[IntegrationV1]
    datastores: list[IntegrationV1]
    queues: list[IntegrationV1]


class RuntimeV1(_StrictModel):
    runtime: list[EvidenceStatementV1]
    deployment: list[EvidenceStatementV1]
    ownership: list[EvidenceStatementV1]


CommandStatus = Literal["verified", "documented", "inferred"]


class RepositoryCommandV1(_StrictModel):
    command: str = Field(min_length=1)
    status: CommandStatus
    purpose: str = Field(min_length=1)
    citations: list[CitationV1] = Field(min_length=1)


class CommandSectionV1(_StrictModel):
    summary: str = Field(min_length=1)
    commands: list[RepositoryCommandV1]
    citations: list[CitationV1] = Field(min_length=1)


class PracticesV1(_StrictModel):
    local_development: CommandSectionV1
    contributing: CommandSectionV1
    testing: CommandSectionV1
    ci: CommandSectionV1
    qa: CommandSectionV1


class ReleaseV1(_StrictModel):
    id: int | str | None
    tag_name: str | None
    name: str | None
    draft: bool
    prerelease: bool
    target_commitish: str | None
    html_url: str | None
    created_at: str | None
    published_at: str | None


class DeploymentV1(_StrictModel):
    id: int | str | None
    environment: str | None
    sha: str | None
    ref: str | None
    task: str | None
    created_at: str | None
    updated_at: str | None
    status: str
    environment_url: str | None
    log_url: str | None


class ReleaseMetadataV1(_StrictModel):
    latest_release: ReleaseV1 | None
    latest_successful_deployment: DeploymentV1 | None
    citations: list[CitationV1]


class ContradictionV1(_StrictModel):
    topic: str = Field(min_length=1)
    claims: list[str] = Field(min_length=2)
    citations: list[CitationV1] = Field(min_length=2)


class UnknownV1(_StrictModel):
    question: str = Field(min_length=1)
    impact: str = Field(min_length=1)


class ConfidenceDimensionsV1(_StrictModel):
    purpose: float = Field(ge=0.0, le=1.0)
    architecture: float = Field(ge=0.0, le=1.0)
    interfaces: float = Field(ge=0.0, le=1.0)
    runtime: float = Field(ge=0.0, le=1.0)
    practices: float = Field(ge=0.0, le=1.0)
    rationale: float = Field(ge=0.0, le=1.0)


class ConfidenceV1(_StrictModel):
    overall: float = Field(ge=0.0, le=1.0)
    dimensions: ConfidenceDimensionsV1
    rationale: str = Field(min_length=1)


class FreshnessV1(_StrictModel):
    commit_sha: str = Field(min_length=1)
    repository_updated_at: str | None
    repository_pushed_at: str | None
    release_published_at: str | None
    evidence_hash: str = Field(min_length=1)
    assessment: str = Field(min_length=1)


class RepoProfileV1(_StrictModel):
    schema_version: Literal["github_repository_profile.v1"]
    repository: str = Field(min_length=1)
    ref: str = Field(min_length=1)
    commit_sha: str = Field(min_length=1)
    evidence_hash: str = Field(min_length=1)
    purpose: PurposeV1
    boundaries: list[EvidenceStatementV1]
    architecture: ArchitectureV1
    interfaces: InterfacesV1
    runtime: RuntimeV1
    practices: PracticesV1
    release: ReleaseMetadataV1
    rationale: list[EvidenceStatementV1]
    contradictions: list[ContradictionV1]
    unknowns: list[UnknownV1]
    confidence: ConfidenceV1
    freshness: FreshnessV1
    citations: list[CitationV1] = Field(min_length=1)
