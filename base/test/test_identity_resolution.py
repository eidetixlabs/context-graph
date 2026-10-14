"""Tests for resolve_or_create_identity email collision handling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from base.db.identity import is_identity_stub, resolve_or_create_identity


def test_identity_stub_detection() -> None:
    assert is_identity_stub(
        employee_id="emp:placeholder:fireflies:org-eidetix:vinod",
        email="vinod@coheso.ai",
    )
    assert is_identity_stub(
        email="placeholder+notion+org-eidetix+abc@placeholder.internal",
    )
    assert not is_identity_stub(
        employee_id="emp:placeholder:fireflies:org-eidetix:vinod",
        email="vinod@coheso.ai",
        auth_provider="local",
    )
    assert not is_identity_stub(
        employee_id="emp:abhinav",
        email="abhinav@eidetix.net",
        auth_provider="local",
    )


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_resolve_reuses_existing_employee_on_email_conflict() -> None:
    """Seeded employee row wins when placeholder insert shares org email."""
    conn = AsyncMock()
    conn.transaction = lambda: _FakeTransaction()

    seeded_id = "emp:abhinav-eidetix"
    calls: list[str] = []

    async def fetchrow(query: str, *args):
        calls.append(query)
        if "FROM identity_mappings" in query:
            return None
        if "FROM employees" in query and "lower(email)" in query:
            return None
        if "INSERT INTO employees" in query:
            assert args[2] == "abhinav@eidetix.net"
            assert "deactivated" in query
            assert "'active'" not in query
            return {"id": seeded_id}
        if "FROM identity_mappings" in query and "authoritative" in query:
            return {"employee_id": seeded_id}
        if "FROM entity_registry" in query and "employee_id = $1" in query:
            return {"id": "entity:person:org-eidetix:employee:emp-abhinav-eidetix-deadbeef"}
        return None

    conn.fetchrow = fetchrow
    conn.execute = AsyncMock()

    employee_id, entity_id = await resolve_or_create_identity(
        conn,
        org_id="org:eidetix",
        provider="notion",
        external_user_id="notion-user-uuid",
        name="Abhinav Eidetix",
        email="abhinav@eidetix.net",
    )

    assert employee_id == seeded_id
    assert entity_id.startswith("entity:person:")
