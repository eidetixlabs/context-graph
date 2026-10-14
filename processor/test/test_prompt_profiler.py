"""Tests for processor/prompt_profiler.py"""

from processor.prompt_profiler import count_words, estimate_tokens, log_llm_prompt_profile


def test_count_words_empty():
    assert count_words("") == 0


def test_count_words_basic():
    assert count_words("one two three") == 3


def test_estimate_tokens_uses_char_and_word_heuristic():
    text = "a" * 300
    assert estimate_tokens(text) == 100


def test_log_llm_prompt_profile_skips_empty_sections(caplog):
    import logging

    caplog.set_level(logging.INFO)

    log_llm_prompt_profile(
        call_site="test.call",
        user_prompt="hello world",
        system_prompt="",
        sections={"title": "", "content": "hello world"},
    )

    message = caplog.records[0].message
    assert "title words=" not in message
    assert "system_prompt words=" not in message
    assert "content words=" in message


def test_log_llm_prompt_profile_emits_sections(caplog):
    import logging

    caplog.set_level(logging.INFO)

    log_llm_prompt_profile(
        call_site="test.call",
        user_prompt="hello world",
        system_prompt="system instructions here",
        sections={"chunk_text": "chunk body", "schema_json": "{}"},
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "LLM prompt profile" in message
    assert "call_site=test.call" in message
    assert "system_prompt words=" in message
    assert "chunk_text words=" in message
    assert "schema_json words=" in message
    assert "user_prompt_total words=" in message
    assert "combined_tokens_est=" in message
