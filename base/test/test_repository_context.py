from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from base.mcp.handler import AgentMcpHandler, DesktopMcpHandler, McpContext
from base.tools.repository_context import (
    assess_commit_policy,
    get_repository_context,
    render_repository_profile_markdown,
)


def _citation(path: str) -> dict[str, object]:
    return {
        "source_type": "file",
        "path": path,
        "blob_sha": f"{path}-sha",
        "line_start": 1,
        "line_end": 4,
    }


def _profile() -> dict[str, object]:
    readme = _citation("README.md")
    pipeline = _citation("processor/pipeline.py")
    ci = _citation(".github/workflows/ci.yml")
    qa = _citation("docs/qa.md")
    return {
        "schema_version": "github_repository_profile.v1",
        "repository": "acme/service",
        "ref": "main",
        "commit_sha": "profile-sha",
        "evidence_hash": "evidence-hash",
        "purpose": {
            "summary": "Processes customer events.",
            "audiences": ["platform"],
            "capabilities": [],
            "citations": [readme],
        },
        "boundaries": [],
        "architecture": {
            "overview": "An API feeds a background processor.",
            "components": [
                {
                    "name": "processor",
                    "description": "Processes queued events.",
                    "responsibilities": ["event extraction"],
                    "technologies": ["Python"],
                    "citations": [pipeline],
                }
            ],
            "data_flows": [],
            "citations": [pipeline],
        },
        "interfaces": {
            "interfaces": [],
            "external_services": [],
            "datastores": [],
            "queues": [],
        },
        "runtime": {"runtime": [], "deployment": [], "ownership": []},
        "practices": {
            "local_development": {
                "summary": "Local setup",
                "commands": [],
                "citations": [],
            },
            "contributing": {
                "summary": "Contributing",
                "commands": [],
                "citations": [],
            },
            "testing": {
                "summary": "Tests",
                "commands": [
                    {
                        "command": "pytest processor/test -q",
                        "status": "documented",
                        "purpose": "Run processor tests",
                        "citations": [pipeline],
                    },
                    {
                        "command": "pytest webserver/test -q",
                        "status": "inferred",
                        "purpose": "Run API tests",
                        "citations": [readme],
                    },
                ],
                "citations": [pipeline],
            },
            "ci": {
                "summary": "CI",
                "commands": [
                    {
                        "command": "ruff check processor",
                        "status": "verified",
                        "purpose": "Lint processor code",
                        "citations": [ci],
                    }
                ],
                "citations": [ci],
            },
            "qa": {
                "summary": "QA",
                "commands": [
                    {
                        "command": "python scripts/smoke_processor.py",
                        "status": "documented",
                        "purpose": "Smoke test event processing",
                        "citations": [qa],
                    }
                ],
                "citations": [qa],
            },
        },
        "release": {
            "latest_release": None,
            "latest_successful_deployment": None,
            "citations": [],
        },
        "rationale": [],
        "contradictions": [],
        "unknowns": [],
        "confidence": {
            "overall": 0.8,
            "dimensions": {
                "purpose": 0.9,
                "architecture": 0.8,
                "interfaces": 0.5,
                "runtime": 0.5,
                "practices": 0.8,
                "rationale": 0.5,
            },
            "rationale": "Evidence-backed.",
        },
        "freshness": {
            "commit_sha": "profile-sha",
            "repository_updated_at": "2026-08-23T00:00:00Z",
            "repository_pushed_at": "2026-08-23T00:00:00Z",
            "release_published_at": None,
            "evidence_hash": "evidence-hash",
            "assessment": "Current when generated.",
        },
        "citations": [readme, pipeline, ci, qa],
    }


@pytest.mark.parametrize(
    ("local_head", "merge_base", "expected"),
    [
        ("profile-sha", "profile-sha", "current"),
        ("feature-sha", "profile-sha", "feature_branch_ahead"),
        ("old-sha", "old-sha", "local_behind"),
        ("feature-sha", "other-sha", "diverged_or_unknown"),
        (None, None, "diverged_or_unknown"),
    ],
)
def test_commit_policy_classifies_local_relationship(
    local_head: str | None,
    merge_base: str | None,
    expected: str,
) -> None:
    result = assess_commit_policy(
        profile_sha="profile-sha",
        local_head=local_head,
        merge_base=merge_base,
        working_tree_dirty=False,
    )
    assert result["relationship"] == expected


def test_dirty_tree_overrides_profile_claims_without_destructive_advice() -> None:
    result = assess_commit_policy(
        profile_sha="profile-sha",
        local_head="profile-sha",
        merge_base="profile-sha",
        working_tree_dirty=True,
    )
    rendered = json.dumps(result).lower()
    assert result["relationship"] == "current"
    assert result["dirty_override"] is True
    assert "inspect" in rendered
    assert "discard" not in rendered
    assert "regenerat" not in rendered


def test_missing_merge_base_tells_agent_to_compute_it_and_preserve_commits() -> None:
    result = assess_commit_policy(
        profile_sha="profile-sha",
        local_head="feature-sha",
        merge_base=None,
        working_tree_dirty=False,
    )
    rendered = json.dumps(result).lower()

    assert result["relationship"] == "diverged_or_unknown"
    assert "merge-base" in rendered
    assert "run" in rendered or "compute" in rendered
    assert "do not discard" in rendered
    assert "feature-branch commits" in rendered


def test_feature_branch_with_profile_as_merge_base_is_ahead() -> None:
    result = assess_commit_policy(
        profile_sha="profile-sha",
        local_head="feature-sha",
        merge_base="profile-sha",
        working_tree_dirty=False,
    )

    assert result["relationship"] == "feature_branch_ahead"
    assert result["stale_warning"] is None
    assert "ahead" in result["explanation"].lower()


class _Pool:
    def __init__(self, row: dict[str, object] | None) -> None:
        self.row = row
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object):
        self.calls.append((query, args))
        return self.row


class _Client:
    def run(self, query: str, params: dict[str, object]):
        assert params["repository_id"] == "entity:repository:acme:service"
        assert params["source_scope"] == "scope:github:acme/service"
        return [
            {
                "props": {
                    "id": "decision:processor-retries",
                    "label": "Retry processor jobs",
                    "rationale": "Retries protect event processing.",
                    "source_doc_id": "doc:decision",
                    "source_scope": "scope:github:acme/service",
                },
                "node_type": "Decision",
            }
        ]


@pytest.mark.asyncio
async def test_context_fetches_canonical_profile_directly_and_is_task_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool(
        {
            "repository": "acme/service",
            "ref": "main",
            "commit_sha": "profile-sha",
            "source_scope": "scope:github:acme/service",
            "last_generated_at": "2026-08-24T10:00:00+00:00",
            "s3_key": "profile.json",
        }
    )

    async def fake_read_json(key: str) -> dict[str, object]:
        assert key == "profile.json"
        return _profile()

    monkeypatch.setattr(
        "base.tools.repository_context.read_json_object",
        fake_read_json,
    )

    result = await get_repository_context(
        pool=pool,
        client=_Client(),
        org_id="org:acme",
        allowed_scopes=["scope:github:acme/service"],
        repository="Acme/Service",
        task="fix processor retries",
        local_head="feature-sha",
        merge_base="profile-sha",
        working_tree_dirty=True,
        changed_paths=["processor/pipeline.py"],
    )

    assert result["repository"] == "acme/service"
    assert result["commit_policy"]["relationship"] == "feature_branch_ahead"
    assert result["commit_policy"]["dirty_override"] is True
    assert result["components"][0]["name"] == "processor"
    assert result["semantic_context"][0]["type"] == "Decision"
    assert result["test_candidates"][0]["command"] == "pytest processor/test -q"
    assert result["test_candidates"][0]["status"] == "documented"
    assert result["test_candidates"][0]["citations"][0]["path"] == (
        "processor/pipeline.py"
    )
    query, args = pool.calls[0]
    assert "github_repository_profiles" in query
    assert "document_chunks" not in query
    assert "source_scope = ANY" in query
    assert args[0:2] == ("org:acme", "acme/service")


@pytest.mark.asyncio
async def test_context_rejects_repository_outside_allowed_github_scope() -> None:
    pool = _Pool(None)
    with pytest.raises(PermissionError, match="acme/service"):
        await get_repository_context(
            pool=pool,
            client=MagicMock(),
            org_id="org:acme",
            allowed_scopes=["scope:github:acme/other"],
            repository="acme/service",
            task="debug service",
        )
    assert pool.calls == []


def test_export_markdown_is_deterministic_cited_and_warns_when_stale() -> None:
    context = {
        "repository": "acme/service",
        "ref": "main",
        "profile_commit": "profile-sha",
        "generated_at": "2026-08-24T10:00:00+00:00",
        "purpose": {
            "summary": "Processes customer events.",
            "citations": [_citation("README.md")],
        },
        "architecture": {
            "overview": "An API feeds a background processor.",
            "citations": [_citation("processor/pipeline.py")],
        },
        "components": [
            {
                "name": "processor",
                "description": "Processes queued events.",
                "citations": [_citation("processor/pipeline.py")],
            }
        ],
        "test_candidates": [],
        "ci_candidates": [],
        "qa_candidates": [],
        "semantic_context": [],
        "cited_paths": ["README.md", "processor/pipeline.py"],
        "freshness": {"assessment": "Current when generated."},
        "commit_policy": {
            "relationship": "local_behind",
            "stale_warning": "Local checkout is behind the canonical profile commit.",
            "dirty_override": False,
            "guidance": "Verify locally before relying on implementation details.",
        },
        "local_verification": "Run relevant commands locally before finalizing.",
    }
    first = render_repository_profile_markdown(context)
    second = render_repository_profile_markdown(context)
    assert first == second
    assert "Generated: 2026-08-24T10:00:00+00:00" in first
    assert "Ref: `main`" in first
    assert "Commit: `profile-sha`" in first
    assert "STALE WARNING" in first
    assert "`README.md:1-4@README.md-sha`" in first


def test_export_markdown_escapes_all_untrusted_profile_fields() -> None:
    context = {
        "repository": "acme/service\n# Injected heading",
        "ref": "main` [click](https://evil.example)",
        "profile_commit": "sha\n!include dangerous",
        "generated_at": "<script>alert(1)</script>",
        "purpose": {
            "summary": "# Heading\n[link](https://evil.example)\n<div>html</div>",
            "citations": [
                {
                    "source_type": "context_note",
                    "note_id": "note`](https://evil.example)",
                }
            ],
        },
        "architecture": {"overview": "> directive\n!include file", "citations": []},
        "components": [
            {
                "name": "name **bold**",
                "description": "<img src=x onerror=alert(1)>",
                "citations": [],
            }
        ],
        "test_candidates": [
            {
                "command": "pytest`\n# injected",
                "status": "[unsafe](https://evil.example)",
                "purpose": "<b>purpose</b>",
                "citations": [],
            }
        ],
        "ci_candidates": [],
        "qa_candidates": [],
        "semantic_context": [
            {
                "type": "Decision",
                "label": "# label",
                "content": "[click](https://evil.example)",
                "source_doc": "`bad`",
            }
        ],
        "cited_paths": ["README.md`\n## injected"],
        "freshness": {},
        "commit_policy": {
            "relationship": "current\n# injected",
            "stale_warning": "# stale\n<div>bad</div>",
            "dirty_override": False,
            "explanation": "[explain](https://evil.example)",
            "guidance": "!include bad",
            "shared_context_policy": "<span>bad</span>",
        },
        "local_verification": "# verify",
    }

    rendered = render_repository_profile_markdown(context)

    assert rendered.count("\n# ") == 0
    assert "\n## injected" not in rendered
    assert "](https://evil.example)" not in rendered
    assert "<script>" not in rendered
    assert "<div>" not in rendered
    assert "<img " not in rendered
    assert "\n!include" not in rendered
    assert "Repository Context" in rendered


def _mcp_context() -> McpContext:
    return McpContext(
        client=MagicMock(),
        employee={"id": "emp:alice", "role": "member"},
        db_pool=MagicMock(),
        org_id="org:acme",
        inference_client_getter=MagicMock(),
    )


def test_repository_read_tools_are_available_to_desktop_and_agent_handlers() -> None:
    for handler_type in (DesktopMcpHandler, AgentMcpHandler):
        tools = {
            tool["name"]: tool
            for tool in handler_type(_mcp_context()).tool_list()
        }
        assert "get_repository_context" in tools
        assert "export_repository_profile" in tools
        assert "refresh_repository_profile" in tools
        schema = tools["get_repository_context"]["inputSchema"]
        assert schema["required"] == ["repository", "task"]
        assert schema["properties"]["working_tree_dirty"]["type"] == "boolean"
        assert schema["properties"]["changed_paths"]["items"]["type"] == "string"
        refresh_tool = tools["refresh_repository_profile"]
        connector_schema = refresh_tool["inputSchema"]["properties"]["connector_id"]
        assert connector_schema["type"] == "string"
        assert connector_schema["minLength"] == 1
        assert "multiple" in refresh_tool["description"].lower()


@pytest.mark.asyncio
async def test_mcp_repository_tools_dispatch_json_and_markdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _mcp_context()
    context.employee["role"] = "service"
    received: list[dict[str, object]] = []
    result = {
        "repository": "acme/service",
        "ref": "main",
        "profile_commit": "profile-sha",
        "generated_at": "2026-08-24T10:00:00+00:00",
        "purpose": {"summary": "Service", "citations": []},
        "architecture": {"overview": "Worker", "citations": []},
        "components": [],
        "test_candidates": [],
        "ci_candidates": [],
        "qa_candidates": [],
        "semantic_context": [],
        "cited_paths": [],
        "commit_policy": {
            "relationship": "current",
            "dirty_override": False,
            "explanation": "Current.",
            "guidance": "Verify locally.",
            "shared_context_policy": "Keep shared context canonical.",
            "stale_warning": None,
        },
        "local_verification": "Run tests locally.",
    }

    async def fake_get_repository_context(**kwargs):
        received.append(kwargs)
        return result

    monkeypatch.setattr(
        "base.mcp.handler.get_repository_context",
        fake_get_repository_context,
    )
    handler = AgentMcpHandler(context)
    response = json.loads(
        await handler.call_tool(
            "get_repository_context",
            {
                "repository": "acme/service",
                "task": "fix retries",
                "local_head": "profile-sha",
                "merge_base": "profile-sha",
                "working_tree_dirty": False,
                "changed_paths": ["processor/pipeline.py"],
            },
        )
    )
    markdown = await handler.call_tool(
        "export_repository_profile",
        {"repository": "acme/service"},
    )

    assert response["profile_commit"] == "profile-sha"
    assert markdown.startswith("# acme/service Repository Context")
    assert received[0]["task"] == "fix retries"
    assert received[0]["changed_paths"] == ["processor/pipeline.py"]
    assert received[1]["task"] == "export repository profile"


@pytest.mark.asyncio
async def test_authenticated_manual_refresh_uses_caller_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _mcp_context()
    received: dict[str, object] = {}

    async def fake_allowed_scopes(pool, employee_id):
        assert employee_id == "emp:alice"
        return ["scope:github:acme/service"]

    async def fake_request_repository_refresh(**kwargs):
        received.update(kwargs)
        return {
            "repository": "acme/service",
            "reason": "manual",
            "requested_profiles": 1,
        }

    monkeypatch.setattr(
        "base.mcp.handler.get_allowed_scopes",
        fake_allowed_scopes,
    )
    monkeypatch.setattr(
        "base.mcp.handler.request_repository_refresh",
        fake_request_repository_refresh,
    )

    result = json.loads(
        await DesktopMcpHandler(context).call_tool(
            "refresh_repository_profile",
            {
                "repository": "acme/service",
                "connector_id": "github-secondary",
            },
        )
    )

    assert result["requested_profiles"] == 1
    assert received["org_id"] == "org:acme"
    assert received["connector_id"] == "github-secondary"
    assert received["allowed_scopes"] == ["scope:github:acme/service"]
    assert received["force_generation"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_type", [DesktopMcpHandler, AgentMcpHandler])
async def test_manual_refresh_rejects_empty_connector_id(
    handler_type,
) -> None:
    context = _mcp_context()
    context.employee["role"] = "service"

    with pytest.raises(ValueError, match="connector_id must be a non-empty string"):
        await handler_type(context).call_tool(
            "refresh_repository_profile",
            {
                "repository": "acme/service",
                "connector_id": "   ",
            },
        )
