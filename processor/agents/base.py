"""
processor/agents/base.py — Model-agnostic agent factory.

Creates PydanticAI Agent instances from the provider:model-name config strings.
Handles provider-specific setup (e.g., Anthropic prompt caching) transparently
so all other code just calls create_agent() and stays provider-agnostic.

All LLM calls should go through run_agent() to enforce the global rate limit.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any, TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings

from base.llm_rate_limiter import llm_rate_limiter
from processor.config import config
from processor.prompt_profiler import log_llm_prompt_profile

_log = logging.getLogger(__name__)

_DepsT = TypeVar("_DepsT")
_OutputT = TypeVar("_OutputT")

_MAX_RATE_LIMIT_RETRIES = 5
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def create_agent(model_str: str, **kwargs: Any) -> Agent:
    """
    Create a PydanticAI Agent from a provider:model-name string.

    PydanticAI natively understands the "provider:model" format, so switching
    providers requires only a config change — no code changes.

    Supported formats (examples):
      "anthropic:claude-3-5-haiku-20241022"
      "anthropic:claude-sonnet-4-20250514"
      "openai:gpt-4o-mini"
      "openai:gpt-4o"
      "google-gla:gemini-1.5-flash"
      "google-gla:gemini-1.5-pro"

    For Anthropic models, prompt caching is enabled automatically by the
    Anthropic SDK when the same system prompt is used repeatedly — no extra
    configuration needed on our side.

    All additional kwargs are forwarded to Agent() (e.g., output_type, tools,
    system_prompt, retries). Global ``config.llm_temperature`` is applied unless
    the caller already sets ``temperature`` in ``model_settings``.
    """
    caller_settings = dict(kwargs.pop("model_settings", None) or {})
    caller_settings.setdefault("temperature", config.llm_temperature)
    kwargs["model_settings"] = ModelSettings(**caller_settings)
    _log.debug("Creating agent: model=%s kwargs=%s", model_str, list(kwargs.keys()))
    from processor.workspace_keys import current_openai_api_key

    openai_key = current_openai_api_key()
    if model_str.startswith("openai:") and openai_key:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model_name = model_str.split(":", 1)[1]
        return Agent(
            model=OpenAIChatModel(
                model_name,
                provider=OpenAIProvider(api_key=openai_key),
            ),
            **kwargs,
        )
    return Agent(model=model_str, **kwargs)


def _extract_usage_tokens(result: AgentRunResult[Any]) -> int | None:
    usage_fn = getattr(result, "usage", None)
    if not callable(usage_fn):
        return None
    try:
        usage = usage_fn()
    except Exception:
        return None
    if not usage.has_values():
        return None
    return usage.total_tokens


def _rate_limit_backoff_seconds(exc: ModelHTTPError, attempt: int) -> float:
    body = exc.body
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str):
            match = _RETRY_AFTER_RE.search(message)
            if match:
                return min(float(match.group(1)), _MAX_BACKOFF_SECONDS)

    exp = _BASE_BACKOFF_SECONDS * (2 ** attempt)
    jitter = random.uniform(0, exp * 0.1)
    return min(exp + jitter, _MAX_BACKOFF_SECONDS)


async def run_agent(
    agent: Agent[Any, Any],
    prompt: str,
    estimated_tokens: int = 0,
    *,
    call_site: str = "unknown",
    prompt_sections: dict[str, str] | None = None,
    profile_system_prompt: str | None = None,
    **kwargs: Any,
) -> AgentRunResult[Any]:
    """
    Rate-limited wrapper around agent.run().

    Acquires the global LLM rate limiter before firing the request, ensuring
    at most one LLM call is in-flight at a time, at least 1 second elapses
    between consecutive responses, and the rolling 60-second token budget is
    respected. `estimated_tokens` is a rough upper-bound for the request.

    Pass `call_site`, `prompt_sections`, and `profile_system_prompt` to log a
    per-section word/token profile before each LLM call. `profile_system_prompt`
    is for profiling only — the agent's system prompt is set at create_agent().

    Drop-in replacement for agent.run(prompt, **kwargs).
    """
    if call_site == "unknown":
        _log.warning("run_agent called without call_site — prompt profiling untagged")

    kwargs.pop("system_prompt", None)

    log_llm_prompt_profile(
        call_site=call_site,
        user_prompt=prompt,
        system_prompt=profile_system_prompt,
        sections=prompt_sections,
    )
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        async with llm_rate_limiter(estimated_tokens=estimated_tokens) as limit:
            try:
                result = await agent.run(prompt, **kwargs)
                actual_tokens = _extract_usage_tokens(result)
                if actual_tokens is not None:
                    limit.record_actual_tokens(actual_tokens)
                return result
            except ModelHTTPError as exc:
                if exc.status_code != 429:
                    raise
                if attempt >= _MAX_RATE_LIMIT_RETRIES - 1:
                    raise
                wait_seconds = _rate_limit_backoff_seconds(exc, attempt)
                _log.warning(
                    "LLM rate limited (429) at %s, retry %d/%d in %.2fs",
                    call_site,
                    attempt + 1,
                    _MAX_RATE_LIMIT_RETRIES - 1,
                    wait_seconds,
                )
        await asyncio.sleep(wait_seconds)

