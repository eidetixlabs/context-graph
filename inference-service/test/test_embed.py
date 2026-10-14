"""
Integration test for inference-service /embed/minilm endpoint.

Uses httpx.AsyncClient with the FastAPI app directly (no real server needed).
NOTE: This test is slow on first run because it loads the MiniLM model (~90MB).
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Import the app from the parent package
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app  # noqa: E402


@pytest.mark.asyncio
async def test_embed_minilm_returns_384_dim_vector():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/embed/minilm", json={"texts": ["hello world"]}
        )

    assert response.status_code == 200
    body = response.json()
    assert "embeddings" in body
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 384
