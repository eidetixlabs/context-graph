"""
processor/agents/scoring.py — Scoring agent.

Assesses whether a document contains semantic signal worth extracting.
Uses the configured scoring_model (default: Haiku / gpt-4o-mini — fast, cheap).
No tool use; purely a classification + scoring task.
"""

from __future__ import annotations

import asyncio
import logging

import pydantic_ai.exceptions
from pydantic_ai import Agent

from processor.agents.base import create_agent, run_agent
from processor.agents.extraction.utils import trim_content
from processor.config import config
from processor.prompts import get_prompt
from processor.schemas.scoring import ScoringResult
from processor.workspace_keys import current_credentials_id

_log = logging.getLogger(__name__)

_scoring_agents: dict[str, Agent[None, ScoringResult]] = {}

_MAX_HALLUCINATION_RETRIES = 2
_HALLUCINATION_RETRY_DELAY = 2.0


def _get_scoring_agent() -> Agent[None, ScoringResult]:
    cache_key = f"{config.scoring_model}:{current_credentials_id()}"
    agent = _scoring_agents.get(cache_key)
    if agent is None:
        agent = create_agent(
            config.scoring_model,
            output_type=ScoringResult,
            system_prompt=get_prompt("scoring"),
            output_retries=2,
            retries=3,
        )
        _scoring_agents[cache_key] = agent
    return agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_scoring(content: str, title: str | None = None) -> ScoringResult:
    """
    Score a document for semantic signal.

    Args:
        content: The full text content of the document (read from S3).
        title:   Optional document title, prepended for context.

    Returns:
        ScoringResult with signal_score (0.0–1.0), signal_types, reasoning,
        and suggested_topics.
    """
    user_prompt, prompt_sections = _build_user_prompt(content, title)

    _log.debug(
        "Scoring document (model=%s, chars=%d)",
        config.scoring_model,
        len(content),
    )

    for attempt in range(_MAX_HALLUCINATION_RETRIES + 1):
        try:
            result = await run_agent(
                _get_scoring_agent(),
                user_prompt,
                estimated_tokens=len(user_prompt) // 3 + 1500,
                call_site="scoring",
                prompt_sections=prompt_sections,
                profile_system_prompt=get_prompt("scoring"),
            )
            break
        except pydantic_ai.exceptions.UnexpectedModelBehavior as exc:
            if attempt >= _MAX_HALLUCINATION_RETRIES:
                raise
            _log.warning(
                "Scoring agent hallucinated tool call (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                _MAX_HALLUCINATION_RETRIES,
                exc,
                _HALLUCINATION_RETRY_DELAY,
            )
            await asyncio.sleep(_HALLUCINATION_RETRY_DELAY)
    else:
        raise AssertionError("unreachable: final retry always re-raises")

    score = result.output

    _log.info(
        "Scoring complete: score=%.3f types=%s model=%s",
        score.signal_score,
        score.signal_types,
        config.scoring_model,
    )

    return score


def _build_user_prompt(content: str, title: str | None) -> tuple[str, dict[str, str]]:
    content_text = trim_content(content)
    title_line = f"<title>{title}</title>\n" if title else ""
    sections: dict[str, str] = {"content": content_text}
    if title:
        sections["title"] = title_line
    prompt = f"{title_line}<document>\n{content_text}\n</document>"
    return prompt, sections
