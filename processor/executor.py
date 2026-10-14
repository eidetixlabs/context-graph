"""
processor/executor.py — Semaphore-bounded executor for model/CPU ops.

All heavy model calls (GLiNER, embeddings, etc.) route through run_in_executor
to cap concurrent threads and prevent OOM from unbounded parallelism.
"""

from __future__ import annotations

import asyncio
from functools import partial

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        from processor.config import config

        _semaphore = asyncio.Semaphore(config.max_model_parallelism)
    return _semaphore


async def run_in_executor(fn, *args, **kwargs):
    """Bounded parallelism wrapper for CPU/model ops."""
    async with _get_semaphore():
        return await asyncio.to_thread(partial(fn, *args, **kwargs))
