import os
from pathlib import Path

from processor.prompts.scoring import SCORING_PROMPT
from processor.prompts.identity_v2 import IDENTITY_V2_PROMPT
from processor.prompts.context import CONTEXT_PROMPT
from processor.prompts.extraction import EXTRACTION_CHUNK_PROMPT
from processor.prompts.disambiguation import DISAMBIGUATION_PROMPT

_PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/etc/eidetix/prompts"))

_DEFAULTS: dict[str, str] = {
    "scoring": SCORING_PROMPT,
    "identity_v2": IDENTITY_V2_PROMPT,
    "context": CONTEXT_PROMPT,
    "extraction_chunk": EXTRACTION_CHUNK_PROMPT,
    "disambiguation": DISAMBIGUATION_PROMPT,
}

_cache: dict[str, str] = {}


def get_prompt(name: str) -> str:
    if name not in _cache:
        path = _PROMPTS_DIR / f"{name}.txt"
        _cache[name] = path.read_text() if path.exists() else _DEFAULTS[name]
    return _cache[name]


__all__ = [
    "SCORING_PROMPT", "IDENTITY_V2_PROMPT",
    "CONTEXT_PROMPT", "EXTRACTION_CHUNK_PROMPT",
    "DISAMBIGUATION_PROMPT",
    "get_prompt",
]
