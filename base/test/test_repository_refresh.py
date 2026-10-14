from __future__ import annotations

import pytest

from base.tools.repository_refresh import request_repository_refresh


class _Connection:
    def __init__(self) -> None:
        self.updated_profile_ids: list[str] = []

    async def fetch(self, query, *args):
        if "FROM connector_instances" in query:
            connector_id = args[1]
            connectors = [
                {
                    "id": "github-primary",
                    "org_id": "org:acme",
                    "config": {"repo": "acme/service"},
                },
                {
                    "id": "github-secondary",
                    "org_id": "org:acme",
                    "config": {"repo": "acme/service"},
                },
            ]
            if connector_id is not None:
                connectors = [
                    connector
                    for connector in connectors
                    if connector["id"] == connector_id
                ]
            return connectors
        if "FROM github_repository_profiles" in query:
            connector_id = args[0]
            return [
                {
                    "id": f"profile:{connector_id}",
                    "ref": "main",
                    "source_scope": "scope:github:acme/service",
                }
            ]
        raise AssertionError(query)

    async def execute(self, query, *args):
        assert "UPDATE github_repository_profiles" in query
        self.updated_profile_ids = list(args[0])
        return "UPDATE 1"


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


@pytest.mark.asyncio
async def test_ambiguous_repository_can_refresh_selected_connector_with_rbac() -> None:
    connection = _Connection()
    pool = _Pool(connection)

    with pytest.raises(LookupError, match="specify connector_id"):
        await request_repository_refresh(
            pool=pool,
            org_id="org:acme",
            repository="acme/service",
            reason="manual",
            allowed_scopes=["scope:github:acme/service"],
        )

    result = await request_repository_refresh(
        pool=pool,
        org_id="org:acme",
        repository="acme/service",
        reason="manual",
        connector_id="github-secondary",
        allowed_scopes=["scope:github:acme/service"],
    )

    assert result["connector_id"] == "github-secondary"
    assert result["requested_profiles"] == 1
    assert connection.updated_profile_ids == ["profile:github-secondary"]

    with pytest.raises(PermissionError, match="does not have access"):
        await request_repository_refresh(
            pool=pool,
            org_id="org:acme",
            repository="acme/service",
            reason="manual",
            connector_id="github-secondary",
            allowed_scopes=["scope:github:acme/other"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_id", ["", "   ", 42])
async def test_refresh_rejects_invalid_connector_id(connector_id) -> None:
    with pytest.raises(ValueError, match="connector_id must be a non-empty string"):
        await request_repository_refresh(
            pool=_Pool(_Connection()),
            org_id="org:acme",
            repository="acme/service",
            reason="manual",
            connector_id=connector_id,
            allowed_scopes=["scope:github:acme/service"],
        )
