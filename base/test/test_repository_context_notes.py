from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base.db.access import _cache
from base.mcp.handler import AgentMcpHandler, DesktopMcpHandler, McpContext
from base.tools.repository_context_notes import (
    NOTE_BODY_MAX_LENGTH,
    NOTE_TITLE_MAX_LENGTH,
    RepositoryNoteAccessError,
    RepositoryNoteConflictError,
    add_repository_context_note,
    normalize_repository,
    supersede_repository_context_note,
)


class _Transaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self):
        self.events.append("transaction:start")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append("transaction:rollback" if exc_type else "transaction:commit")
        return False


class _Connection:
    def __init__(self, prior_note: dict[str, object] | None = None) -> None:
        self.prior_note = prior_note
        self.events: list[str] = []
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []

    def transaction(self):
        return _Transaction(self.events)

    async def fetchrow(self, query: str, *args):
        if query.lstrip().startswith("SELECT"):
            self.events.append("select:prior")
            self.calls.append(("fetchrow", query, args))
            return self.prior_note
        self.events.append("insert:note")
        self.calls.append(("fetchrow", query, args))
        return {
            "id": args[0],
            "repository": args[3],
            "category": args[5],
            "body": args[6],
            "status": "active",
            "supersedes_id": args[7],
            "author_employee_id": args[4],
            "created_at": "2026-08-24T12:00:00+00:00",
        }

    async def execute(self, query: str, *args):
        event = "update:prior" if "repository_context_notes" in query else "enqueue:profiles"
        self.events.append(event)
        self.calls.append(("execute", query, args))
        return "UPDATE 2" if event == "enqueue:profiles" else "UPDATE 1"


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Pool:
    def __init__(
        self,
        connection: _Connection,
        scopes: list[str] | None = None,
    ) -> None:
        self.connection = connection
        self.fetch = AsyncMock(
            return_value=[
                {"channel_id": scope}
                for scope in (scopes or ["scope:github:acme/service"])
            ]
        )
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        return _Acquire(self.connection)


@pytest.fixture(autouse=True)
def _clear_scope_cache() -> None:
    _cache.clear()


def test_repository_and_note_validation() -> None:
    assert normalize_repository(" Acme ", " Service.API ") == (
        "acme",
        "service.api",
        "acme/service.api",
        "scope:github:acme/service.api",
    )

    with pytest.raises(ValueError, match="owner"):
        normalize_repository("", "service")
    with pytest.raises(ValueError, match="repo"):
        normalize_repository("acme", "../service")


@pytest.mark.asyncio
async def test_add_rejects_invalid_category_and_body_before_db_access() -> None:
    pool = _Pool(_Connection())

    with pytest.raises(ValueError, match="category"):
        await add_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="emp:alice",
            owner="acme",
            repo="service",
            category="secret",
            body="Useful context",
        )
    with pytest.raises(ValueError, match="body"):
        await add_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="emp:alice",
            owner="acme",
            repo="service",
            category="insight",
            body=" \n ",
        )

    pool.fetch.assert_not_awaited()
    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_note_title_and_body_limits_are_enforced_before_db_access() -> None:
    pool = _Pool(_Connection())

    assert max(len(category) for category in ("architecture", "contributing")) <= (
        NOTE_TITLE_MAX_LENGTH
    )
    with pytest.raises(ValueError, match="body.*at most"):
        await add_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="emp:alice",
            owner="acme",
            repo="service",
            category="insight",
            body="x" * (NOTE_BODY_MAX_LENGTH + 1),
        )

    pool.fetch.assert_not_awaited()
    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_add_requires_exact_repository_scope_access() -> None:
    pool = _Pool(_Connection(), scopes=["scope:github:acme/other"])

    with pytest.raises(RepositoryNoteAccessError, match="acme/service"):
        await add_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="emp:alice",
            owner="Acme",
            repo="Service",
            category="architecture",
            body="Uses an event-driven boundary.",
        )

    pool.fetch.assert_awaited_once_with(
        "SELECT channel_id FROM channel_members WHERE employee_id = $1",
        "emp:alice",
    )
    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_add_requires_an_authenticated_employee() -> None:
    pool = _Pool(_Connection())

    with pytest.raises(RepositoryNoteAccessError, match="authenticated employee"):
        await add_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="",
            owner="acme",
            repo="service",
            category="insight",
            body="Useful context",
        )

    pool.fetch.assert_not_awaited()
    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_add_attributes_note_and_requeues_profiles_without_replacing_evidence() -> None:
    connection = _Connection()
    pool = _Pool(connection, scopes=["scope:github:Acme/Service"])

    result = await add_repository_context_note(
        pool,
        org_id="org:acme",
        employee_id="emp:alice",
        owner="Acme",
        repo="Service",
        category=" testing ",
        body="  Run the hermetic suite.  ",
    )

    assert result["repository"] == "acme/service"
    assert result["author_employee_id"] == "emp:alice"
    assert result["category"] == "testing"
    assert result["body"] == "Run the hermetic suite."
    assert result["enqueued_profiles"] == 2
    assert connection.events == [
        "transaction:start",
        "insert:note",
        "enqueue:profiles",
        "transaction:commit",
    ]
    enqueue_query = next(
        query
        for kind, query, _ in connection.calls
        if kind == "execute" and "github_repository_profiles" in query
    )
    assert "generation_state = 'pending'" in enqueue_query
    assert "trigger_reason = 'context_note'" in enqueue_query
    assert "evidence_hash" not in enqueue_query
    assert "evidence_s3_key" not in enqueue_query
    insert_args = next(
        args
        for kind, query, args in connection.calls
        if kind == "fetchrow" and "INSERT INTO repository_context_notes" in query
    )
    assert insert_args[2] == "scope:github:Acme/Service"
    insert_query = next(
        query
        for kind, query, _ in connection.calls
        if kind == "fetchrow" and "INSERT INTO repository_context_notes" in query
    )
    assert "length($6)" in insert_query
    assert "length($7)" in insert_query


@pytest.mark.asyncio
async def test_supersession_and_enqueue_are_one_transaction() -> None:
    prior = {
        "id": "note:old",
        "org_id": "org:acme",
        "source_scope": "scope:github:acme/service",
        "repository": "acme/service",
        "status": "active",
    }
    connection = _Connection(prior)
    pool = _Pool(connection)

    result = await supersede_repository_context_note(
        pool,
        org_id="org:acme",
        employee_id="emp:alice",
        owner="acme",
        repo="service",
        supersedes_id="note:old",
        category="architecture",
        body="The worker now owns profile synthesis.",
    )

    assert result["supersedes_id"] == "note:old"
    assert connection.events == [
        "transaction:start",
        "select:prior",
        "update:prior",
        "insert:note",
        "enqueue:profiles",
        "transaction:commit",
    ]
    select_query = connection.calls[0][1]
    assert "FOR UPDATE" in select_query
    prior_update = connection.calls[1][1]
    assert "status = 'active'" in prior_update


@pytest.mark.asyncio
async def test_cannot_supersede_note_from_another_repository() -> None:
    prior = {
        "id": "note:other",
        "org_id": "org:acme",
        "source_scope": "scope:github:acme/other",
        "repository": "acme/other",
        "status": "active",
    }
    connection = _Connection(prior)
    pool = _Pool(connection)

    with pytest.raises(RepositoryNoteConflictError, match="same repository"):
        await supersede_repository_context_note(
            pool,
            org_id="org:acme",
            employee_id="emp:alice",
            owner="acme",
            repo="service",
            supersedes_id="note:other",
            category="insight",
            body="Do not allow cross-repository supersession.",
        )

    assert connection.events == [
        "transaction:start",
        "select:prior",
        "transaction:rollback",
    ]


def _mcp_context(pool: _Pool) -> McpContext:
    return McpContext(
        client=MagicMock(),
        employee={
            "id": "emp:alice",
            "org_id": "org:acme",
            "canonical_name": "Alice",
            "role": "member",
        },
        db_pool=pool,  # type: ignore[arg-type]
        org_id="org:acme",
        inference_client_getter=MagicMock(),
    )


@pytest.mark.asyncio
async def test_desktop_mcp_exposes_authenticated_note_actions() -> None:
    context = _mcp_context(_Pool(_Connection()))
    handler = DesktopMcpHandler(context)
    tools = {tool["name"]: tool for tool in handler.tool_list()}
    agent_tools = {tool["name"] for tool in AgentMcpHandler(context).tool_list()}

    assert tools["add_repository_context_note"]["inputSchema"]["required"] == [
        "owner",
        "repo",
        "category",
        "body",
    ]
    assert tools["supersede_repository_context_note"]["inputSchema"]["required"] == [
        "owner",
        "repo",
        "supersedes_id",
        "category",
        "body",
    ]
    assert "add_repository_context_note" not in agent_tools
    assert "supersede_repository_context_note" not in agent_tools

    with patch(
        "base.mcp.handler.add_repository_context_note",
        AsyncMock(
            return_value={
                "id": "note:new",
                "repository": "acme/service",
                "category": "insight",
                "author_employee_id": "emp:alice",
                "created_at": "2026-08-24T12:00:00+00:00",
                "enqueued_profiles": 1,
            }
        ),
    ):
        response = json.loads(
            await handler.call_tool(
                "add_repository_context_note",
                {
                    "owner": "acme",
                    "repo": "service",
                    "category": "insight",
                    "body": "Owned by the platform team.",
                },
            )
        )

    assert response == {
        "id": "note:new",
        "repository": "acme/service",
        "category": "insight",
        "author_employee_id": "emp:alice",
        "created_at": "2026-08-24T12:00:00+00:00",
        "enqueued_profiles": 1,
    }
