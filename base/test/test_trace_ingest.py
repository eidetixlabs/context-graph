from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from base.tools import trace_ingest
from base.tools import s3 as tool_s3


def _trace_to_node():
    transformer_path = (
        Path(__file__).resolve().parents[2] / "processor" / "trace" / "transformer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "context_graph_trace_transformer",
        transformer_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trace_to_node


class _AcquireContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    async def __aenter__(self) -> object:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _TxnContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConn:
    def __init__(self) -> None:
        self.execute = AsyncMock()

    def transaction(self) -> _TxnContext:
        return _TxnContext()


class _FakePool:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


def test_normalize_trace_entries_canonicalizes_repo_metadata() -> None:
    [trace] = trace_ingest.normalize_trace_entries(
        [
            {
                "trace_id": "trace-1",
                "ts": "2026-09-16T00:00:00Z",
                "repo": "Acme/Widgets",
                "node": {"type": "Insight", "label": "Gateway", "body": "Owns auth"},
            }
        ],
        project="acme",
        org_id="org:acme",
        author_name="",
        author_employee_id="emp:1",
    )
    assert trace["repo"] == "acme/widgets"


@pytest.mark.asyncio
async def test_top_level_trace_repo_survives_storage_and_becomes_node_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "test-bucket")
    monkeypatch.setattr(tool_s3, "ensure_bucket_async", AsyncMock())
    monkeypatch.setattr(
        trace_ingest,
        "_ensure_trace_scope_membership",
        AsyncMock(),
    )
    stored: list[dict] = []

    async def capture_trace(_key: str, payload: dict) -> None:
        stored.append(dict(payload))

    monkeypatch.setattr(tool_s3, "write_json_async", capture_trace)
    traces = trace_ingest.normalize_trace_entries(
        [
            {
                "trace_id": "trace-1",
                "ts": "2026-09-16T00:00:00Z",
                "repo": "Acme/Widgets",
                "node": {
                    "type": "Insight",
                    "label": "Gateway",
                    "body": "Owns auth",
                },
            }
        ],
        project="acme",
        org_id="org:acme",
        author_name="",
        author_employee_id="emp:1",
    )

    result = await trace_ingest.store_traces_s3(
        project="acme",
        org_id="org:acme",
        traces=traces,
        employee_id="emp:1",
    )
    node = _trace_to_node()(stored[0])

    assert result == {"stored": 1, "failed": 0}
    assert "repo" not in stored[0]["node"]
    assert stored[0]["repo"] == "acme/widgets"
    assert node.repo == "acme/widgets"


@pytest.mark.asyncio
async def test_ensure_trace_scope_membership_upserts_scope_and_member(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn()
    monkeypatch.setattr(trace_ingest.db_pool, "get", lambda: _FakePool(conn))
    invalidated: list[str] = []
    monkeypatch.setattr(trace_ingest, "invalidate_scope_cache", lambda employee_id: invalidated.append(employee_id))

    await trace_ingest._ensure_trace_scope_membership(
        org_id="org:test",
        employee_id="emp:test",
    )

    assert conn.execute.await_count == 2
    first_call = conn.execute.await_args_list[0].args
    second_call = conn.execute.await_args_list[1].args

    assert "INSERT INTO access_scopes" in first_call[0]
    assert first_call[1] == "scope:trace:emp:test"
    assert first_call[2] == "org:test"

    assert "INSERT INTO channel_members" in second_call[0]
    assert second_call[1] == "scope:trace:emp:test"
    assert second_call[2] == "emp:test"
    assert second_call[3] == "emp:test"
    assert invalidated == ["emp:test"]
