"""
processor/prompt_profiler.py — Word/token profiling for LLM prompts.

Logs per-section breakdowns so we can identify which prompt components
are driving TPM usage.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative estimate aligned with existing len(text) // 3 usage.
    return max(len(text) // 3, count_words(text))


def log_llm_prompt_profile(
    *,
    call_site: str,
    user_prompt: str,
    system_prompt: str | None = None,
    sections: dict[str, str] | None = None,
) -> None:
    parts: list[str] = [f"call_site={call_site}"]

    if system_prompt:
        parts.append(
            f"system_prompt words={count_words(system_prompt)} "
            f"tokens_est={estimate_tokens(system_prompt)}"
        )

    if sections:
        for name, text in sections.items():
            if not text:
                continue
            parts.append(
                f"{name} words={count_words(text)} tokens_est={estimate_tokens(text)}"
            )

    parts.append(
        f"user_prompt_total words={count_words(user_prompt)} "
        f"tokens_est={estimate_tokens(user_prompt)}"
    )

    combined_tokens = estimate_tokens(user_prompt)
    if system_prompt:
        combined_tokens += estimate_tokens(system_prompt)
    parts.append(f"combined_tokens_est={combined_tokens}")

    _log.info("LLM prompt profile | %s", " | ".join(parts))
