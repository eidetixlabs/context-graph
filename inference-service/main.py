"""
inference-service/main.py — HTTP inference service for torch-heavy models.

Serves MiniLM (384-dim), BGE-large (1024-dim), and GLiNER NER over HTTP.
All models are lazy singletons (double-checked locking). Concurrent calls are
gated through a bounded asyncio.Semaphore → ThreadPoolExecutor so the service
never oversubscribes CPU/RAM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# ── Bounded executor ──────────────────────────────────────────────────────────

_MAX_PARALLELISM = int(os.environ.get("MAX_MODEL_PARALLELISM", "2"))

# Pin intra-op threads so concurrent workers don't oversubscribe CPU cores.
# Formula: floor((reserved_cores - 1) / MAX_PARALLELISM).
# With 7 reserved cores and parallelism=2: floor(6/2) = 3.
# Override via OMP_NUM_THREADS env var.
_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", "3"))
torch.set_num_threads(_OMP_THREADS)
torch.set_num_interop_threads(1)
_log.info("torch threads: intra=%d  interop=1  max_parallelism=%d", _OMP_THREADS, _MAX_PARALLELISM)
_executor = ThreadPoolExecutor(max_workers=_MAX_PARALLELISM)
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_PARALLELISM)
    return _semaphore


async def _run_in_executor(fn, *args, **kwargs):
    """Bounded parallelism wrapper for CPU/model ops."""
    loop = asyncio.get_event_loop()
    async with _get_semaphore():
        return await loop.run_in_executor(_executor, partial(fn, *args, **kwargs))


# ── Model singletons ──────────────────────────────────────────────────────────

_minilm_model = None
_minilm_lock = threading.Lock()

_bge_model = None
_bge_lock = threading.Lock()

_gliner_model = None
_gliner_lock = threading.Lock()


def _get_minilm():
    global _minilm_model
    if _minilm_model is None:
        with _minilm_lock:
            if _minilm_model is None:
                from sentence_transformers import SentenceTransformer
                _log.info("Loading MiniLM model: all-MiniLM-L6-v2")
                _minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
                _log.info("MiniLM model loaded")
    return _minilm_model


def _get_bge():
    global _bge_model
    if _bge_model is None:
        with _bge_lock:
            if _bge_model is None:
                from sentence_transformers import SentenceTransformer
                _log.info("Loading BGE model: BAAI/bge-large-en-v1.5")
                _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
                _log.info("BGE model loaded")
    return _bge_model


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


# ── Sync model call implementations ──────────────────────────────────────────

def _encode_minilm(texts: list[str]) -> list[list[float]]:
    model = _get_minilm()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def _encode_bge(texts: list[str]) -> list[list[float]]:
    model = _get_bge()
    vectors = model.encode(texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


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

class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


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

app = FastAPI(title="inference-service", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/embed/minilm", response_model=EmbedResponse)
async def embed_minilm(req: EmbedRequest):
    embeddings = await _run_in_executor(_encode_minilm, req.texts)
    return EmbedResponse(embeddings=embeddings)


@app.post("/embed/bge", response_model=EmbedResponse)
async def embed_bge(req: EmbedRequest):
    embeddings = await _run_in_executor(_encode_bge, req.texts)
    return EmbedResponse(embeddings=embeddings)


@app.post("/ner", response_model=NERResponse)
async def ner(req: NERRequest):
    entities = await _run_in_executor(_predict_ner, req.text, req.labels)
    return NERResponse(entities=[NEREntity(**e) for e in entities])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9111)
