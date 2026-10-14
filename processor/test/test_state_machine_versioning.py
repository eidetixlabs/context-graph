from __future__ import annotations

import pytest

from processor.state_machine import (
    StaleDocumentWork,
    transition_extracted,
    transition_indexing,
)


class _Pool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args):
        self.calls.append((query, args))
        return self.result


@pytest.mark.asyncio
async def test_transition_indexing_cas_includes_version_and_extracting_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool("UPDATE 0")
    monkeypatch.setattr("processor.state_machine.db_pool.get", lambda: pool)

    with pytest.raises(StaleDocumentWork):
        await transition_indexing("doc:1", document_version=1)

    query, args = pool.calls[0]
    assert "current_version = $2" in query
    assert "processing_state = 'extracting'" in query
    assert args == ("doc:1", 1)


@pytest.mark.asyncio
async def test_transition_extracted_cas_includes_version_and_indexing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool("UPDATE 0")
    monkeypatch.setattr("processor.state_machine.db_pool.get", lambda: pool)

    with pytest.raises(StaleDocumentWork):
        await transition_extracted("doc:1", document_version=2, model="model")

    query, args = pool.calls[0]
    assert "current_version = $3" in query
    assert "processing_state = 'indexing'" in query
    assert args == ("model", "doc:1", 2)
