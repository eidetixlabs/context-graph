from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from base import routing


@pytest.mark.asyncio
async def test_resolve_org_id_from_email_domain():
    central = AsyncMock()
    central.fetchrow = AsyncMock(return_value={"org_id": "org:acme"})

    org_id = await routing.resolve_org_id_from_email_domain(central, "ada@acme.com")

    assert org_id == "org:acme"
    central.fetchrow.assert_awaited()
    assert central.fetchrow.await_args.args[1] == "acme.com"


@pytest.mark.asyncio
async def test_resolve_org_id_from_email_domain_falls_back_to_slug():
    central = AsyncMock()
    central.fetchrow = AsyncMock(side_effect=[None, {"org_id": "org:acme"}])

    org_id = await routing.resolve_org_id_from_email_domain(central, "ada@acme.com")

    assert org_id == "org:acme"
    assert central.fetchrow.await_count == 2
    assert central.fetchrow.await_args_list[1].args[1] == "acme"


@pytest.mark.asyncio
async def test_resolve_org_id_from_email_domain_ignores_invalid():
    central = AsyncMock()
    assert await routing.resolve_org_id_from_email_domain(central, "not-an-email") is None
    central.fetchrow.assert_not_called()
