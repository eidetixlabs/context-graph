"""
base/openai_embedding_client.py — Async OpenAI embedding client.

Replaces the inference-service embedding endpoints when
EMBEDDING_PROVIDER=openai (the default). Dimensions are fixed to match the
existing pgvector and Neo4j vector index schemas:

  text-embedding-3-large  dims=1024  → document_chunks vector(1024)  [BGE replacement]
  text-embedding-3-small  dims=384   → Neo4j vector indexes 384-dim   [MiniLM replacement]

No schema migration required — the dimensions align exactly.
"""

from __future__ import annotations

import asyncio
import logging
import os

_log = logging.getLogger(__name__)

_LARGE_MODEL = "text-embedding-3-large"
_SMALL_MODEL = "text-embedding-3-small"
_LARGE_DIMS = 1024
_SMALL_DIMS = 384

_MAX_RETRIES = 3
_RETRY_BACKOFF = 0.5


class OpenAIEmbeddingClient:
    """
    Async OpenAI embedding client.

    embed_large  → text-embedding-3-large at 1024 dims  (replaces BGE-large)
    embed_small  → text-embedding-3-small at 384  dims  (replaces MiniLM)
    """

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI  # noqa: PLC0415

        self._client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def _embed(
        self, texts: list[str], model: str, dimensions: int
    ) -> list[list[float]]:
        if any(not text for text in texts):
            raise ValueError(
                f"OpenAI embed {model}: refusing to embed empty string input(s)"
            )

        backoff = _RETRY_BACKOFF
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.embeddings.create(
                    model=model,
                    input=texts,
                    dimensions=dimensions,
                )
                return [d.embedding for d in resp.data]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _log.warning(
                    "OpenAI embed %s attempt %d/%d: %s — retrying in %.1fs",
                    model, attempt + 1, _MAX_RETRIES, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        _log.error("OpenAI embed %s failed after %d retries", model, _MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    async def embed_large(self, texts: list[str]) -> list[list[float]]:
        """Embed with text-embedding-3-large at 1024 dims (replaces BGE-large)."""
        return await self._embed(texts, _LARGE_MODEL, _LARGE_DIMS)

    async def embed_small(self, texts: list[str]) -> list[list[float]]:
        """Embed with text-embedding-3-small at 384 dims (replaces MiniLM)."""
        return await self._embed(texts, _SMALL_MODEL, _SMALL_DIMS)
