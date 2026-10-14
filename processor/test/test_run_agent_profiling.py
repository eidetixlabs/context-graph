"""Tests for run_agent profiling guardrails."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from processor.agents.base import run_agent


@pytest.mark.asyncio
async def test_run_agent_warns_on_unknown_call_site(caplog):
    caplog.set_level(logging.WARNING)
    agent = MagicMock()
    agent.run = AsyncMock(return_value="ok")

    with patch("processor.agents.base.llm_rate_limiter") as limiter:
        limiter.return_value.__aenter__ = AsyncMock(return_value=None)
        limiter.return_value.__aexit__ = AsyncMock(return_value=None)
        await run_agent(agent, "prompt")

    assert any("without call_site" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_run_agent_strips_system_prompt_from_agent_run_kwargs():
    agent = MagicMock()
    agent.run = AsyncMock(return_value="ok")

    with patch("processor.agents.base.llm_rate_limiter") as limiter:
        limiter.return_value.__aenter__ = AsyncMock(return_value=None)
        limiter.return_value.__aexit__ = AsyncMock(return_value=None)
        await run_agent(
            agent,
            "prompt",
            call_site="test",
            system_prompt="should not forward",
        )

    agent.run.assert_awaited_once_with("prompt")
