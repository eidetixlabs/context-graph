from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from base import routing


@pytest.mark.asyncio
async def test_resolve_org_from_header_requires_header_by_default(monkeypatch):
    monkeypatch.delenv("EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK", raising=False)

    with pytest.raises(RuntimeError, match="X-Eidetix-Org header missing"):
        await routing.resolve_org_from_header(
            central_db=object(),  # not used in this branch
            org_header=None,
            default_org_id="org:dev",
        )


@pytest.mark.asyncio
async def test_resolve_org_from_header_fallback_allowed_when_enabled(monkeypatch):
    monkeypatch.setenv("EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK", "1")
    fake_ctx = routing.OrgCtx(org_id="org:dev", pool=object(), neo4j=object())
    monkeypatch.setattr(routing, "_resolve_to_ctx", AsyncMock(return_value=fake_ctx))

    ctx = await routing.resolve_org_from_header(
        central_db=object(),
        org_header=None,
        default_org_id="org:dev",
    )

    assert ctx.org_id == "org:dev"


@pytest.mark.asyncio
async def test_resolve_org_from_token_requires_token_org_mapping(monkeypatch):
    monkeypatch.delenv("EIDETIX_ALLOW_DEFAULT_ORG_FALLBACK", raising=False)
    monkeypatch.setattr(routing, "_lookup_token_org", AsyncMock(return_value=None))

    with pytest.raises(RuntimeError, match="token_org row missing"):
        await routing.resolve_org_from_token(
            central_db=object(),
            raw_token="token",
            default_org_id="org:dev",
        )
