"""
base/llm_rate_limiter.py — Global async LLM token tracker.

Allows fully concurrent LLM calls. Tracks rolling 60-second token usage
and warns when the TPM budget is exceeded, but does NOT serialise or delay
requests. If the rolling budget is exhausted, callers are warned and proceed
(backpressure is handled by provider 429s + retry logic in run_agent).

Usage:
    from base.llm_rate_limiter import llm_rate_limiter

    async with llm_rate_limiter(estimated_tokens=5000) as limit:
        result = await agent.run(prompt)
        limit.record_actual_tokens(result.usage().total_tokens)
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time

_log = logging.getLogger(__name__)

_TPM_WINDOW: float = 60.0
_DEFAULT_TPM_LIMIT: int = 180_000


class LLMRateLimiter:
    """
    Concurrent LLM token tracker with rolling 60-second TPM accounting.

    Multiple requests run in parallel. No serialisation, no inter-request
    delay. Token log is protected by a short-held asyncio.Lock only during
    read/write (microseconds), never across the HTTP call itself.

    Logs a warning when estimated rolling usage exceeds tpm_limit but does
    not block — real backpressure comes from provider 429s.
    """

    def __init__(self, tpm_limit: int = _DEFAULT_TPM_LIMIT) -> None:
        self._tpm_limit = tpm_limit
        self._lock = asyncio.Lock()
        self._token_log: collections.deque[tuple[float, int]] = collections.deque()

    def __call__(self, estimated_tokens: int = 0) -> "_RateLimiterContext":
        return _RateLimiterContext(self, estimated_tokens)

    async def _acquire(self, estimated_tokens: int) -> None:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - _TPM_WINDOW
            while self._token_log and self._token_log[0][0] < cutoff:
                self._token_log.popleft()
            rolling_tokens = sum(t for _, t in self._token_log)

        if rolling_tokens + estimated_tokens > self._tpm_limit:
            _log.warning(
                "LLMRateLimiter: rolling=%d + estimated=%d exceeds tpm_limit=%d; proceeding (watch for 429s)",
                rolling_tokens,
                estimated_tokens,
                self._tpm_limit,
            )

    async def _release(self, billed_tokens: int) -> None:
        if billed_tokens > 0:
            async with self._lock:
                self._token_log.append((time.monotonic(), billed_tokens))


class _RateLimiterContext:
    def __init__(self, limiter: LLMRateLimiter, estimated_tokens: int) -> None:
        self._limiter = limiter
        self._estimated_tokens = estimated_tokens
        self._actual_tokens: int | None = None

    def record_actual_tokens(self, actual_tokens: int) -> None:
        """Replace the pre-call estimate with provider-reported usage."""
        if actual_tokens >= 0:
            self._actual_tokens = actual_tokens

    async def __aenter__(self) -> "_RateLimiterContext":
        await self._limiter._acquire(self._estimated_tokens)
        return self

    async def __aexit__(self, exc_type: object, *_: object) -> None:
        billed_tokens = (
            self._actual_tokens
            if self._actual_tokens is not None
            else self._estimated_tokens
        )
        if self._actual_tokens is not None and self._actual_tokens != self._estimated_tokens:
            _log.debug(
                "LLMRateLimiter: reconciled estimated=%d actual=%d",
                self._estimated_tokens,
                self._actual_tokens,
            )
        await self._limiter._release(billed_tokens)


llm_rate_limiter = LLMRateLimiter()
