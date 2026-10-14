"""
processor/embedding.py — Text embedding for Neo4j entity nodes (384-dim).

Routes to OpenAI text-embedding-3-small (dims=384) by default, or to the
inference-service MiniLM endpoint when embedding_provider="inference".

Both produce 384-dim vectors that match the existing Neo4j vector indexes.
"""

from __future__ import annotations

import logging

from processor.config import config

_log = logging.getLogger(__name__)

_openai_clients: dict[str, object] = {}
_inference_client = None


def _get_openai_client():
    from processor.workspace_keys import (  # noqa: PLC0415
        current_credentials_id,
        current_openai_api_key,
    )

    key = current_openai_api_key() or ""
    cache_key = current_credentials_id()
    client = _openai_clients.get(cache_key)
    if client is None:
        from base.openai_embedding_client import OpenAIEmbeddingClient  # noqa: PLC0415

        client = OpenAIEmbeddingClient(api_key=key or None)
        _openai_clients[cache_key] = client
    return client


def _get_inference_client():
    global _inference_client
    if _inference_client is None:
        from base.inference_client import InferenceClient  # noqa: PLC0415
        _inference_client = InferenceClient(config.inference_url)
    return _inference_client


async def embed_text(text: str) -> list[float]:
    """Embed text (384-dim) for Neo4j entity vector indexes."""
    if config.embedding_provider == "inference":
        embeddings = await _get_inference_client().embed_minilm([text])
    else:
        embeddings = await _get_openai_client().embed_small([text])
    return embeddings[0]
