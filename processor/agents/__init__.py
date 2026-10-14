"""Lazy public agent exports.

Avoid importing the full extraction/Neo4j stack when a lightweight agent
module (such as repository profile synthesis) is imported.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "run_extraction_pipeline",
    "run_scoring",
    "synthesize_repository_profile",
]


def __getattr__(name: str) -> Any:
    if name == "run_scoring":
        from processor.agents.scoring import run_scoring

        return run_scoring
    if name == "run_extraction_pipeline":
        from processor.agents.extraction.pipeline import run_extraction_pipeline

        return run_extraction_pipeline
    if name == "synthesize_repository_profile":
        from processor.agents.repository_profile import synthesize_repository_profile

        return synthesize_repository_profile
    raise AttributeError(name)
