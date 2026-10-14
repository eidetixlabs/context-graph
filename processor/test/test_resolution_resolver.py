"""Unit tests for deterministic resolver behavior changes."""

from __future__ import annotations

from typing import Any

import pytest

from processor.resolution.models import MentionCandidate
from processor.resolution.resolver import _OrgData, _resolve_single, resolve_mentions, resolve_trace_mentions


class _Conn:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> None:
        self.execute_calls.append((query, args))


@pytest.mark.asyncio
async def test_resolve_single_skips_llm_when_no_fuzzy_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    called = {"llm": False}

    async def _fake_llm(*args: Any, **kwargs: Any) -> str | None:
        called["llm"] = True
        return None

    monkeypatch.setattr("processor.resolution.resolver._llm_disambiguate", _fake_llm)

    entry = await _resolve_single(
        mention="Completely New Name",
        entity_type="person",
        org_data=_OrgData(),
        org_id="org:test",
        conn=conn,
        mention_confidence=0.77,
    )

    assert entry is not None
    assert called["llm"] is False
    assert entry.resolution_method == "new"
    assert entry.entity_id is None


@pytest.mark.asyncio
async def test_resolve_single_runs_llm_for_single_fuzzy_candidate_and_writes_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn()
    seen = {"candidate_count": 0}

    async def _fake_llm(mention: str, candidates: list[Any], _conn: Any) -> str | None:
        assert mention == "Alex K"
        seen["candidate_count"] = len(candidates)
        return "entity:person:test:alex-kim"

    monkeypatch.setattr("processor.resolution.resolver._llm_disambiguate", _fake_llm)
    org_data = _OrgData(
        fuzzy_index={
            "person": [
                ("entity:person:test:alex-kim", ("alex", "alex kim"), "alex kim"),
            ],
        },
    )

    entry = await _resolve_single(
        mention="Alex K",
        entity_type="person",
        org_data=org_data,
        org_id="org:test",
        conn=conn,
        mention_confidence=0.8,
        existing_entity_ids={"entity:person:test:alex-kim"},
    )

    assert entry is not None
    assert seen["candidate_count"] == 1
    assert entry.entity_id == "entity:person:test:alex-kim"
    assert entry.resolution_method == "llm_disambiguation"
    assert any("INSERT INTO entity_aliases" in query for query, _ in conn.execute_calls)


@pytest.mark.asyncio
async def test_resolve_mentions_updates_intra_run_indexes_for_new_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _Conn()
    consistent_flags: list[bool] = []

    async def _fake_fetch_org_data(_conn: Any, _org_id: str, *, consistent: bool = False) -> _OrgData:
        consistent_flags.append(consistent)
        return _OrgData()

    async def _fake_llm(_mention: str, candidates: list[Any], _conn: Any) -> str | None:
        return candidates[0].entity_id if candidates else None

    monkeypatch.setattr("processor.resolution.resolver._fetch_org_data", _fake_fetch_org_data)
    monkeypatch.setattr("processor.resolution.resolver._llm_disambiguate", _fake_llm)

    mentions = [
        MentionCandidate(
            mention_text="Naman Kabra",
            mention_type="person",
            chunk_index=0,
            confidence=0.95,
        ),
        MentionCandidate(
            mention_text="Naman K",
            mention_type="person",
            chunk_index=1,
            confidence=0.94,
        ),
    ]

    result = await resolve_mentions(mentions=mentions, org_id="org:test", conn=conn)

    assert consistent_flags == [True]
    first_id = result.entries["Naman Kabra"].entity_id
    second_id = result.entries["Naman K"].entity_id
    assert first_id is not None
    assert second_id == first_id
    assert result.entries["Naman Kabra"].resolution_method == "new"
    assert result.entries["Naman K"].resolution_method == "llm_disambiguation"


@pytest.mark.asyncio
async def test_resolve_trace_mentions_uses_consistent_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _Conn()
    consistent_flags: list[bool] = []

    async def _fake_fetch_org_data(_conn: Any, _org_id: str, *, consistent: bool = False) -> _OrgData:
        consistent_flags.append(consistent)
        return _OrgData()

    monkeypatch.setattr("processor.resolution.resolver._fetch_org_data", _fake_fetch_org_data)

    mentions = [
        MentionCandidate(
            mention_text="Trace Person",
            mention_type="person",
            chunk_index=0,
            confidence=0.9,
        ),
    ]

    await resolve_trace_mentions(mentions=mentions, org_id="org:test", conn=conn)
    assert consistent_flags == [True]
