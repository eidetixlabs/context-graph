"""
processor/schemas/scoring.py — Output schema for the scoring agent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SignalType = Literal["decision", "constraint", "insight", "signal"]


class ScoringResult(BaseModel):
    signal_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Signal density score. 0=no signal, 1=extremely high signal.",
    )
    signal_types: list[SignalType] = Field(
        default_factory=list,
        description="Signal types detected in this document.",
    )
    reasoning: str = Field(
        description="Brief explanation of why this score was assigned.",
    )
    suggested_topics: list[str] = Field(
        default_factory=list,
        description="1-5 topic keywords/phrases that describe the document's subject matter. "
                    "Used as seeds for Phase 2 context gathering.",
    )
