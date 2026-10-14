from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from processor.agents.repository_profile import (
    _SYSTEM_PROMPT,
    _prepare_evidence_for_synthesis,
    _validate_profile_provenance,
    build_fallback_profile,
    synthesize_repository_profile,
)
from processor.repository_profile_worker import (
    RepositoryProfileQueueRow,
    RepositoryProfileWorker,
    claim_repository_profiles,
    persist_repository_profile,
    process_repository_profile,
)
from processor.providers.github_profile import render_repository_profile_markdown
from processor.schemas.repository_profile import ArchitectureV1, CitationV1, RepoProfileV1
from processor.schemas.extraction import DecisionNode, ExtractionResult
from processor.state_machine import DocumentRecord


CLAIMED_AT = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def test_architecture_prompt_asks_for_module_level_briefing() -> None:
    prompt = _SYSTEM_PROMPT.lower()
    assert "architecture is the section coding agents use to navigate" in prompt
    assert "bot.py" in prompt or "entrypoint" in prompt
    assert "data flows" in prompt
    assert "generic" in prompt
    overview = ArchitectureV1.model_fields["overview"].description or ""
    assert "navigation briefing" in overview.lower()
    assert "n files" in overview.lower() or "file names" in overview.lower()


def _evidence() -> dict:
    return {
        "schema_version": "github_repository_evidence.v1",
        "repository": "acme/service",
        "ref": "main",
        "commit_sha": "abc123",
        "evidence_hash": "evidence-hash",
        "repository_metadata": {
            "name": "service",
            "full_name": "acme/service",
            "owner": "acme",
            "description": "Processes customer events.",
            "html_url": "https://github.com/acme/service",
            "homepage": None,
            "private": True,
            "archived": False,
            "fork": False,
            "default_branch": "main",
            "license": None,
            "topics": ["events"],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2026-08-20T00:00:00Z",
            "pushed_at": "2026-08-23T00:00:00Z",
        },
        "languages": {"Python": 1000},
        "latest_release": {
            "id": 1,
            "tag_name": "v1.2.3",
            "name": "August",
            "draft": False,
            "prerelease": False,
            "target_commitish": "main",
            "html_url": "https://github.com/acme/service/releases/v1.2.3",
            "created_at": "2026-08-01T00:00:00Z",
            "published_at": "2026-08-01T00:00:00Z",
        },
        "latest_successful_deployment": {
            "id": 9,
            "environment": "production",
            "sha": "abc123",
            "ref": "main",
            "task": "deploy",
            "created_at": "2026-08-20T00:00:00Z",
            "updated_at": "2026-08-20T00:05:00Z",
            "status": "success",
            "environment_url": None,
            "log_url": None,
        },
        "required_checks": ["lint", "test"],
        "tree_summary": {
            "total_files": 3,
            "top_directories": [{"directory": "src", "file_count": 2}],
            "top_extensions": [{"extension": "py", "file_count": 2}],
        },
        "tree_inventory": [],
        "evidence": [
            {
                "category": "purpose",
                "extraction_reason": "repository readme",
                "rank": 1,
                "path": "README.md",
                "blob_sha": "readme-sha",
                "byte_count": 100,
                "content": "# Service\nConsumes events.",
                "citation": {
                    "path": "README.md",
                    "blob_sha": "readme-sha",
                    "line_start": 1,
                    "line_end": 2,
                },
            },
            {
                "category": "ci",
                "extraction_reason": "continuous integration or release workflow",
                "rank": 2,
                "path": ".github/workflows/ci.yml",
                "blob_sha": "ci-sha",
                "byte_count": 100,
                "content": "run: pytest",
                "citation": {
                    "path": ".github/workflows/ci.yml",
                    "blob_sha": "ci-sha",
                    "line_start": 1,
                    "line_end": 1,
                },
            },
        ],
        "commands": [
            {
                "command": "pytest",
                "status": "verified",
                "reason": "exact CI run step",
                "citation": {
                    "path": ".github/workflows/ci.yml",
                    "blob_sha": "ci-sha",
                    "line_start": 1,
                    "line_end": 1,
                },
            }
        ],
        "discourse": {"issues": [], "pull_requests": []},
        "selection": {},
        "omissions": [],
    }


def test_repo_profile_schema_is_strict_and_citations_are_typed() -> None:
    with pytest.raises(ValidationError):
        CitationV1.model_validate(
            {
                "source_type": "file",
                "path": "README.md",
                "blob_sha": "sha",
                "line_start": 1,
                "line_end": 2,
                "unexpected": True,
            }
        )

    with pytest.raises(ValidationError):
        CitationV1.model_validate({"source_type": "file", "path": "README.md"})


def test_synthesis_prompt_is_bounded_and_preserves_auditable_identity() -> None:
    evidence = _evidence()
    evidence["tree_inventory"] = [
        {"path": f"src/file_{index}.py", "sha": str(index), "size": 1}
        for index in range(20_000)
    ]
    evidence["evidence"][0]["content"] = "evidence line\n" * 50_000
    evidence["evidence"][0]["citation"]["line_end"] = 50_000

    projected = _prepare_evidence_for_synthesis(evidence)

    assert projected["tree_inventory"] == []
    assert projected["evidence_hash"] == evidence["evidence_hash"]
    assert len(json.dumps(projected)) < 500_000
    assert projected["evidence"][0]["citation"]["line_end"] < 50_000
    assert any(
        omission.get("reason") == "file_excerpt"
        for omission in projected["omissions"]
    )


def test_profile_provenance_accepts_only_cited_file_subranges() -> None:
    evidence = _evidence()
    profile = build_fallback_profile(evidence, [])
    readme = next(
        citation
        for citation in profile.purpose.citations
        if citation.source_type == "file"
    )
    readme.line_end = readme.line_start
    _validate_profile_provenance(profile, evidence, [])

    readme.line_end = 999
    with pytest.raises(ValueError, match="unsupported citation"):
        _validate_profile_provenance(profile, evidence, [])


def test_profile_provenance_accepts_known_path_when_model_mangles_blob_sha() -> None:
    evidence = _evidence()
    profile = build_fallback_profile(evidence, [])
    readme = next(
        citation
        for citation in profile.purpose.citations
        if citation.source_type == "file"
    )
    readme.blob_sha = "deadbeef" * 4
    _validate_profile_provenance(profile, evidence, [])


@pytest.mark.asyncio
async def test_model_citation_to_truncated_unseen_lines_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    evidence["evidence"][0]["content"] = "visible\n" * 50_000
    evidence["evidence"][0]["citation"]["line_end"] = 50_000
    model_profile = build_fallback_profile(
        _prepare_evidence_for_synthesis(evidence),
        [],
    )
    model_profile.purpose.summary = "Unseen model claim."
    readme = next(
        citation
        for citation in model_profile.purpose.citations
        if citation.source_type == "file"
    )
    readme.line_start = 49_999
    readme.line_end = 50_000

    async def fake_run_agent(*args, **kwargs):
        return SimpleNamespace(output=model_profile.model_dump(mode="json"))

    monkeypatch.setattr(
        "processor.agents.repository_profile._configured_model",
        lambda: "test:model",
    )
    monkeypatch.setattr(
        "processor.agents.repository_profile._get_profile_agent",
        lambda model: object(),
    )
    monkeypatch.setattr(
        "processor.agents.repository_profile.run_agent",
        fake_run_agent,
    )

    result = await synthesize_repository_profile(evidence, [])

    assert result.purpose.summary != "Unseen model claim."


def test_profile_provenance_rejects_uncited_factual_sections() -> None:
    evidence = _evidence()
    profile = build_fallback_profile(evidence, [])
    profile.purpose.citations = []

    with pytest.raises(ValueError, match="purpose.*citation"):
        _validate_profile_provenance(profile, evidence, [])


def test_fallback_profile_is_deterministic_complete_and_evidence_backed() -> None:
    notes = [
        {
            "id": "note:1",
            "category": "architecture",
            "body": "The event boundary is intentionally isolated.",
            "author_employee_id": "emp:alice",
            "created_at": "2026-08-22T00:00:00Z",
            "updated_at": "2026-08-22T00:00:00Z",
        }
    ]

    first = build_fallback_profile(_evidence(), notes)
    second = build_fallback_profile(deepcopy(_evidence()), deepcopy(notes))

    assert isinstance(first, RepoProfileV1)
    assert first == second
    assert first.repository == "acme/service"
    assert first.purpose.summary
    assert first.architecture.components
    assert first.practices.testing.commands[0].command == "pytest"
    assert first.practices.testing.commands[0].citations[0].path == (
        ".github/workflows/ci.yml"
    )
    assert any(item.citations[0].note_id == "note:1" for item in first.rationale)
    assert first.release.latest_release.tag_name == "v1.2.3"
    assert first.confidence.overall >= 0
    assert first.freshness.commit_sha == "abc123"
    assert all(
        section.citations
        for section in (
            first.practices.local_development,
            first.practices.contributing,
            first.practices.testing,
            first.practices.ci,
            first.practices.qa,
        )
    )


def test_profile_worker_renders_useful_markdown_and_bypasses_llm_scoring() -> None:
    profile = build_fallback_profile(_evidence(), [])
    markdown = render_repository_profile_markdown(profile)

    assert "# acme/service Repository Profile" in markdown
    assert "## Purpose" in markdown
    assert "## Architecture" in markdown
    assert "## Testing" in markdown
    assert "pytest" in markdown
    assert "README.md" in markdown

    from processor.providers.github_profile import GithubProfileProcessorWorker
    from processor.worker import BaseProcessorWorker

    worker = GithubProfileProcessorWorker()
    assert isinstance(worker, BaseProcessorWorker)


def test_profile_markdown_escapes_untrusted_prose_and_code_spans() -> None:
    profile = build_fallback_profile(_evidence(), [])
    injection = "safe\n## Decision\n<script>x</script> [click](https://evil) `directive`"
    profile.repository = injection
    profile.ref = "main` BAD"
    profile.purpose.summary = injection
    profile.purpose.audiences = [injection]
    profile.architecture.components[0].name = injection
    profile.practices.testing.commands[0].command = "pytest` \n## Constraint"
    profile.rationale = [
        profile.purpose.capabilities[0].model_copy(update={"summary": injection})
    ]
    next(
        citation
        for citation in profile.purpose.citations
        if citation.source_type == "file"
    ).path = "README.md` \n## Insight"

    markdown = render_repository_profile_markdown(profile)

    assert markdown.count("## Purpose") == 1
    assert "## Decision" not in markdown
    assert "\n## Constraint" not in markdown
    assert "\n## Insight" not in markdown
    assert "<script>" not in markdown
    assert "[click](https://evil)" not in markdown
    assert "`directive`" not in markdown
    assert "&#96;" in markdown
    assert "\\[click\\]\\(https://evil\\)" in markdown


@pytest.mark.asyncio
async def test_profile_worker_uses_deterministic_admission_score() -> None:
    from processor.providers.github_profile import GithubProfileProcessorWorker

    worker = GithubProfileProcessorWorker()
    doc = DocumentRecord(
        id="doc:github-profile:acme/service/_profile/main",
        org_id="org:acme",
        source_provider="github-profile",
        source_scope="scope:github:acme/service",
        source_ref="acme/service/_profile/main",
        title="acme/service repository profile",
        s3_key="profile.json",
        processing_state="scoring",
        signal_score=None,
        char_count=100,
        current_version=1,
        parent_ref="acme/service",
    )

    result = await worker.score_content(doc, "profile markdown")

    assert result.signal_score == 1.0
    assert result.signal_types == ["decision", "constraint", "insight", "signal"]
    assert "repository profile" in result.reasoning.lower()


def test_profile_extraction_is_whole_document_and_repository_anchored() -> None:
    from processor.agents.extraction.pipeline import (
        _extraction_strategy_for_provider,
        _inject_profile_repository,
    )

    doc = DocumentRecord(
        id="doc:github-profile:acme/service/_profile/main",
        org_id="org:acme",
        source_provider="github-profile",
        source_scope="scope:github:acme/service",
        source_ref="acme/service/_profile/main",
        title="Profile",
        s3_key="profile.json",
        processing_state="extracting",
        signal_score=1.0,
        char_count=100,
        current_version=1,
        parent_ref="acme/service",
    )
    result = ExtractionResult(
        nodes=[
            DecisionNode(
                id="decision:storage",
                label="Storage",
                rationale="Durability",
            )
        ]
    )

    _inject_profile_repository(doc, result)

    assert _extraction_strategy_for_provider("github-profile") == "whole_document"
    repository = next(
        node
        for node in result.nodes
        if getattr(node, "entity_type", None) == "repository"
    )
    assert repository.id == "entity:repository:acme:service"
    assert any(
        edge.from_id == "decision:storage"
        and edge.rel == "ABOUT"
        and edge.to_id == repository.id
        for edge in result.edges
    )


@pytest.mark.asyncio
async def test_incomplete_indexing_does_not_transition_document_to_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processor.dual_write import IndexJob, IndexingIncompleteError
    from processor.providers.github_profile import GithubProfileProcessorWorker

    worker = GithubProfileProcessorWorker()
    profile = build_fallback_profile(_evidence(), [])
    doc = DocumentRecord(
        id="doc:github-profile:acme/service/_profile/main",
        org_id="org:acme",
        source_provider="github-profile",
        source_scope="scope:github:acme/service",
        source_ref="acme/service/_profile/main",
        title="Profile",
        s3_key="profile.json",
        processing_state="extracting",
        signal_score=1.0,
        char_count=100,
        current_version=1,
        parent_ref="acme/service",
    )
    worker._scoring_cache[doc.id] = await worker.score_content(doc, "markdown")
    failed: list[str] = []

    async def fake_read(key):
        return profile.model_dump_json()

    async def fake_pipeline(**kwargs):
        raise IndexingIncompleteError(
            IndexJob(
                doc_id=doc.id,
                document_version=1,
                payload_hash="hash",
                payload_s3_key="index.json",
                vector_state="completed",
                graph_state="failed",
            )
        )

    async def fake_transition_failed(doc_id, error):
        failed.append(doc_id)

    monkeypatch.setattr("processor.worker.read_document_content", fake_read)
    monkeypatch.setattr("processor.worker.run_extraction_pipeline", fake_pipeline)
    monkeypatch.setattr("processor.worker.transition_failed", fake_transition_failed)
    monkeypatch.setattr("processor.worker._neo4j_client", MagicMock())

    assert not await worker.extract_document(doc)
    assert failed == []


@pytest.mark.asyncio
async def test_invalid_llm_profile_uses_schema_valid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_agent(*args, **kwargs):
        return SimpleNamespace(output={"schema_version": "wrong"})

    monkeypatch.setattr(
        "processor.agents.repository_profile._configured_model",
        lambda: "test:model",
    )
    monkeypatch.setattr(
        "processor.agents.repository_profile._get_profile_agent",
        lambda model: object(),
    )
    monkeypatch.setattr(
        "processor.agents.repository_profile.run_agent",
        fake_run_agent,
    )

    profile = await synthesize_repository_profile(_evidence(), [])

    assert isinstance(profile, RepoProfileV1)
    assert profile.evidence_hash == "evidence-hash"


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Connection:
    def __init__(self, rows=None) -> None:
        self.rows = list(rows or [])
        self.fetchrow_result = None
        self.queue_row = {
            "id": "profile:1",
            "generation_state": "in_progress",
            "evidence_hash": "evidence-hash",
            "commit_sha": "abc123",
            "evidence_s3_key": "evidence.json",
            "updated_at": CLAIMED_AT,
        }
        self.fetch_queries: list[str] = []
        self.executions: list[tuple[str, tuple[object, ...]]] = []

    def transaction(self):
        return _Transaction()

    async def fetch(self, query: str, *args):
        self.fetch_queries.append(query)
        return self.rows

    async def fetchval(self, query: str, *args):
        return "repository_context_notes"

    async def fetchrow(self, query: str, *args):
        self.fetch_queries.append(query)
        if "FROM github_repository_profiles" in query:
            return self.queue_row
        return self.fetchrow_result

    async def execute(self, query: str, *args):
        self.executions.append((query, args))
        return "UPDATE 1"


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


class _ConcurrentTransaction:
    def __init__(self, pool) -> None:
        self.pool = pool

    async def __aenter__(self):
        await self.pool.lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.pool.lock.release()
        return False


class _ConcurrentConnection:
    def __init__(self, pool) -> None:
        self.pool = pool

    def transaction(self):
        return _ConcurrentTransaction(self.pool)

    async def fetchrow(self, query: str, *args):
        if "FROM github_repository_profiles" in query:
            return self.pool.claims[args[0]]
        if "FROM documents" in query:
            return dict(self.pool.document) if self.pool.document else None
        return None

    async def execute(self, query: str, *args):
        if "INSERT INTO documents" in query:
            self.pool.document = {
                "id": args[0],
                "current_version": args[9],
                "content_hash": args[6],
                "s3_key": args[8],
            }
        elif "INSERT INTO document_versions" in query:
            self.pool.versions.append(int(args[1]))
        return "UPDATE 1"


class _ConcurrentAcquire:
    def __init__(self, pool) -> None:
        self.connection = _ConcurrentConnection(pool)

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _ConcurrentPool:
    def __init__(self, rows: list[RepositoryProfileQueueRow]) -> None:
        self.lock = asyncio.Lock()
        self.document: dict | None = None
        self.versions: list[int] = []
        self.claims = {
            row.id: {
                "id": row.id,
                "generation_state": "in_progress",
                "evidence_hash": row.evidence_hash,
                "commit_sha": row.commit_sha,
                "evidence_s3_key": row.evidence_s3_key,
                "updated_at": row.claimed_at,
            }
            for row in rows
        }

    def acquire(self):
        return _ConcurrentAcquire(self)


@pytest.mark.asyncio
async def test_queue_claim_is_atomic_and_reclaims_stale_work() -> None:
    connection = _Connection(
        rows=[
            {
                "id": "profile:1",
                "org_id": "org:acme",
                "connector_id": "conn:github:acme",
                "source_scope": "scope:github:acme/service",
                "repository": "acme/service",
                "ref": "main",
                "commit_sha": "abc123",
                "evidence_hash": "evidence-hash",
                "evidence_s3_key": "evidence.json",
                "trigger_reason": "scheduled",
                "release_metadata": {},
                "updated_at": CLAIMED_AT,
            }
        ]
    )

    claimed = await claim_repository_profiles(_Pool(connection), batch_size=5)

    assert claimed[0].id == "profile:1"
    query = connection.fetch_queries[0]
    assert "FOR UPDATE SKIP LOCKED" in query
    assert "generation_state = 'in_progress'" in query
    assert "interval '30 minutes'" in query


@pytest.mark.asyncio
async def test_worker_includes_notes_and_uses_atomic_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    connection = _Connection(
        rows=[
            {
                "id": "note:1",
                "category": "testing",
                "body": "Use the hermetic suite.",
                "author_employee_id": "emp:alice",
                "created_at": "2026-08-22T00:00:00Z",
                "updated_at": "2026-08-22T00:00:00Z",
            }
        ]
    )
    pool = _Pool(connection)
    received: dict[str, object] = {}
    fallback = build_fallback_profile(_evidence(), [])

    async def fake_read_json(key: str):
        assert key == "evidence.json"
        return _evidence()

    async def fake_synthesize(evidence: dict, notes: list[dict]):
        received["notes"] = notes
        return fallback

    async def fake_persist(pool_arg, queue_row, profile):
        assert pool_arg is pool
        assert queue_row is row
        assert profile is fallback
        return "doc:github-profile:acme/service/_profile/main"

    monkeypatch.setattr(
        "processor.repository_profile_worker.read_json_object", fake_read_json
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.synthesize_repository_profile",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.persist_repository_profile",
        fake_persist,
    )

    assert await process_repository_profile(pool, row)
    assert received["notes"][0]["id"] == "note:1"
    assert received["notes"][0]["author_employee_id"] == "emp:alice"
    assert received["notes"][0]["created_at"] == "2026-08-22T00:00:00Z"


@pytest.mark.asyncio
async def test_profile_document_persistence_uses_scope_hash_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    profile = build_fallback_profile(_evidence(), [])
    connection = _Connection()
    pool = _Pool(connection)
    writes: list[tuple[str, bytes]] = []

    async def fake_write(key: str, content: bytes) -> None:
        writes.append((key, content))

    monkeypatch.setattr(
        "processor.repository_profile_worker.write_json_object",
        fake_write,
    )

    doc_id = await persist_repository_profile(pool, row, profile)

    assert doc_id == "doc:github-profile:acme/service/_profile/main"
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert writes[0][1] == canonical
    payload_hash = hashlib.sha256(canonical).hexdigest()
    assert writes[0][0].endswith(f"/github-profile/raw/{payload_hash}.json")
    document_query, document_args = next(
        execution
        for execution in connection.executions
        if "INSERT INTO documents" in execution[0]
    )
    assert "processing_state" in document_query
    assert document_args[3] == "scope:github:acme/service"
    assert document_args[6] == payload_hash
    assert document_args[9] == 1
    assert not any(
        "INSERT INTO document_index_jobs" in query for query, _ in connection.executions
    )
    completion_query, completion_args = next(
        execution
        for execution in connection.executions
        if "generation_state = 'completed'" in execution[0]
    )
    assert "commit_sha = $4" in completion_query
    assert "evidence_s3_key = $5" in completion_query
    assert completion_args[0] == row.id

    connection.fetchrow_result = {
        "id": doc_id,
        "current_version": 1,
        "content_hash": payload_hash,
        "s3_key": writes[0][0],
    }
    writes.clear()
    assert await persist_repository_profile(pool, row, profile) == doc_id
    assert writes == []


@pytest.mark.asyncio
async def test_stale_queue_refresh_cannot_publish_profile_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    connection = _Connection()
    connection.queue_row = {
        "id": row.id,
        "generation_state": "pending",
        "evidence_hash": "new-evidence-hash",
        "commit_sha": "new-sha",
        "evidence_s3_key": "new-evidence.json",
        "updated_at": CLAIMED_AT,
    }
    pool = _Pool(connection)

    async def fake_read_json(key: str):
        return _evidence()

    async def fake_synthesize(evidence: dict, notes: list[dict]):
        return build_fallback_profile(evidence, notes)

    writes: list[tuple[str, bytes]] = []

    async def fake_write(key: str, content: bytes) -> None:
        writes.append((key, content))

    monkeypatch.setattr(
        "processor.repository_profile_worker.read_json_object", fake_read_json
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.synthesize_repository_profile",
        fake_synthesize,
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.write_json_object",
        fake_write,
    )

    assert not await process_repository_profile(pool, row)
    assert writes == []
    assert not any(
        "INSERT INTO documents" in query for query, _ in connection.executions
    )


@pytest.mark.asyncio
async def test_reclaimed_identical_profile_claim_invalidates_old_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    connection = _Connection()
    connection.queue_row = {
        "id": row.id,
        "generation_state": "in_progress",
        "evidence_hash": row.evidence_hash,
        "commit_sha": row.commit_sha,
        "evidence_s3_key": row.evidence_s3_key,
        "updated_at": datetime(2026, 8, 24, 12, 31, tzinfo=timezone.utc),
    }
    writes: list[str] = []

    async def fake_write(key: str, content: bytes) -> None:
        writes.append(key)

    monkeypatch.setattr(
        "processor.repository_profile_worker.write_json_object",
        fake_write,
    )

    result = await persist_repository_profile(
        _Pool(connection),
        row,
        build_fallback_profile(_evidence(), []),
    )

    assert result is None
    assert writes == []
    assert not any(
        "INSERT INTO documents" in query for query, _ in connection.executions
    )


@pytest.mark.asyncio
async def test_profile_version_allocation_locks_document_and_uses_content_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    profile = build_fallback_profile(_evidence(), [])
    connection = _Connection()
    connection.fetchrow_result = {
        "id": "doc:github-profile:acme/service/_profile/main",
        "current_version": 7,
        "content_hash": "old-hash",
        "s3_key": "old.json",
    }
    writes: list[str] = []

    async def fake_write(key: str, content: bytes) -> None:
        writes.append(key)

    monkeypatch.setattr(
        "processor.repository_profile_worker.write_json_object", fake_write
    )

    await persist_repository_profile(_Pool(connection), row, profile)

    assert any("FOR UPDATE" in query for query in connection.fetch_queries)
    canonical = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload_hash = hashlib.sha256(canonical).hexdigest()
    assert writes == [f"org:acme/github-profile/raw/{payload_hash}.json"]
    version_insert = next(
        args
        for query, args in connection.executions
        if "INSERT INTO document_versions" in query
    )
    assert version_insert[1] == 8


@pytest.mark.asyncio
async def test_concurrent_profile_publications_allocate_distinct_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_row = RepositoryProfileQueueRow(
        id="profile:1",
        org_id="org:acme",
        connector_id="conn:github:acme",
        source_scope="scope:github:acme/service",
        repository="acme/service",
        ref="main",
        commit_sha="abc123",
        evidence_hash="evidence-hash",
        evidence_s3_key="evidence-1.json",
        trigger_reason="scheduled",
        release_metadata={},
        claimed_at=CLAIMED_AT,
    )
    second_row = RepositoryProfileQueueRow(
        **{
            **first_row.__dict__,
            "id": "profile:2",
            "evidence_s3_key": "evidence-2.json",
        }
    )
    first_profile = build_fallback_profile(_evidence(), [])
    second_profile = first_profile.model_copy(deep=True)
    second_profile.purpose.summary = "A newer cited repository purpose."
    pool = _ConcurrentPool([first_row, second_row])
    keys: list[str] = []

    async def fake_write(key: str, content: bytes) -> None:
        keys.append(key)
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "processor.repository_profile_worker.write_json_object", fake_write
    )

    results = await asyncio.gather(
        persist_repository_profile(pool, first_row, first_profile),
        persist_repository_profile(pool, second_row, second_profile),
    )

    assert results[0] == results[1]
    assert pool.versions == [1, 2]
    assert len(set(keys)) == 2


@pytest.mark.asyncio
async def test_profile_pipeline_cycle_runs_synthesis_before_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _IndexWorker:
        async def _run_cycle(self):
            events.append("index")
            return SimpleNamespace()

    async def fake_claim(pool, *, batch_size):
        events.append("claim")
        return [object()]

    async def fake_process(pool, row):
        events.append("synthesize")
        return True

    monkeypatch.setattr(
        "processor.repository_profile_worker.claim_repository_profiles", fake_claim
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.process_repository_profile", fake_process
    )
    monkeypatch.setattr(
        "processor.repository_profile_worker.db_pool.get", lambda: object()
    )

    worker = RepositoryProfileWorker(index_worker=_IndexWorker())
    await worker._run_cycle()

    assert events == ["claim", "synthesize", "index"]
