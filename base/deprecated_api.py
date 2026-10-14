"""Deprecated REST route marker."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import ParamSpec, TypeVar

_log = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def deprecated_api(
    *,
    reason: str,
    replacement: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Log deprecation warning on each request.

    Pair with ``@app.<method>(..., deprecated=True)`` so OpenAPI marks the route.
    """
    msg = reason
    if replacement:
        msg = f"{msg}; use {replacement} instead"

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _log.warning("Deprecated API %s: %s", func.__name__, msg)
            return await func(*args, **kwargs)

        wrapper.__deprecated_api__ = True  # type: ignore[attr-defined]
        wrapper.__deprecated_reason__ = reason  # type: ignore[attr-defined]
        wrapper.__deprecated_replacement__ = replacement  # type: ignore[attr-defined]
        return wrapper

    return decorator
