"""
inference-service/gliner_main.py — GLiNER-only NER service.

Stripped-down variant of the full inference-service — no sentence-transformers,
no BGE, no MiniLM, no torch embedding.  Serves only the /ner endpoint.

Used as the eidetix-gliner image for:
  - edicli k8s processor --inference-service  (on-demand testing, any cluster)
  - Any deployment where embeddings are handled by OpenAI but GLiNER NER
    is still desired.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# ── Bounded executor ──────────────────────────────────────────────────────────

_MAX_PARALLELISM = int(os.environ.get("MAX_MODEL_PARALLELISM", "2"))
_executor = ThreadPoolExecutor(max_workers=_MAX_PARALLELISM)
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_PARALLELISM)
    return _semaphore


async def _run_in_executor(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    async with _get_semaphore():
        return await loop.run_in_executor(_executor, partial(fn, *args, **kwargs))


# ── GLiNER singleton ──────────────────────────────────────────────────────────

_gliner_model = None
_gliner_lock = threading.Lock()


def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        with _gliner_lock:
            if _gliner_model is None:
                from gliner import GLiNER
                _log.info("Loading GLiNER model: urchade/gliner_medium-v2.1")
                _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
                _log.info("GLiNER model loaded")
    return _gliner_model


def _predict_ner(text: str, labels: list[str]) -> list[dict[str, Any]]:
    model = _get_gliner()
    raw = model.predict_entities(text, labels)
    return [
        {
            "label": item["label"],
            "text": item["text"],
            "score": float(item.get("score", 0.7)),
            "start": item.get("start", 0),
            "end": item.get("end", 0),
        }
        for item in raw
    ]


# ── Request/response schemas ──────────────────────────────────────────────────

class NERRequest(BaseModel):
    text: str
    labels: list[str]


class NEREntity(BaseModel):
    label: str
    text: str
    score: float
    start: int
    end: int


class NERResponse(BaseModel):
    entities: list[NEREntity]


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="eidetix-gliner", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ner", response_model=NERResponse)
async def ner(req: NERRequest):
    entities = await _run_in_executor(_predict_ner, req.text, req.labels)
    return NERResponse(entities=[NEREntity(**e) for e in entities])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9111)
