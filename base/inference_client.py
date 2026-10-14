"""
base/inference_client.py — Async httpx client for the inference-service.

Exposes embed_minilm, embed_bge, and ner with exponential backoff retry
(up to 3 attempts, 0.5 s / 1 s / 2 s delays) on connection errors or 5xx.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

_log = logging.getLogger(__name__)

_TIMEOUT = 60.0       # model inference can be slow
_MAX_RETRIES = 3
_RETRY_BACKOFF = 0.5  # seconds; doubles each retry


class InferenceClient:
    """Async HTTP client for the inference-service (default port 9111)."""

    def __init__(self, base_url: str = "http://localhost:9111") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=_TIMEOUT)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "InferenceClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _post(self, path: str, json: dict) -> dict:
        """POST with exponential backoff retry on connection errors and 5xx responses."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None
        backoff = _RETRY_BACKOFF

        for attempt in range(_MAX_RETRIES):
            try:
                resp = await self._client.post(url, json=json)
                resp.raise_for_status()
                return resp.json()

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if 400 <= status < 500:
                    _log.error("Inference %s → %s: %s", path, status, exc.response.text)
                    raise
                last_exc = exc
                _log.warning(
                    "Inference %s → %s (attempt %d/%d), retrying in %.1fs",
                    path, status, attempt + 1, _MAX_RETRIES, backoff,
                )

            except httpx.RequestError as exc:
                last_exc = exc
                _log.warning(
                    "Inference %s network error (attempt %d/%d): %s, retrying in %.1fs",
                    path, attempt + 1, _MAX_RETRIES, exc, backoff,
                )

            await asyncio.sleep(backoff)
            backoff *= 2

        _log.error("Inference %s failed after %d retries", path, _MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    async def embed_minilm(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with all-MiniLM-L6-v2 (384-dim)."""
        result = await self._post("/embed/minilm", {"texts": texts})
        return result["embeddings"]

    async def embed_bge(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with BAAI/bge-large-en-v1.5 (1024-dim)."""
        result = await self._post("/embed/bge", {"texts": texts})
        return result["embeddings"]

    async def ner(self, text: str, labels: list[str]) -> list[dict]:
        """Run zero-shot NER; returns list of {label, text, score, start, end}."""
        result = await self._post("/ner", {"text": text, "labels": labels})
        return result["entities"]
