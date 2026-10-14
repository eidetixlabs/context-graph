"""Tests for base.temporal."""

from __future__ import annotations

import pytest
from neo4j.time import DateTime

from base.temporal import parse_neo4j_datetime


@pytest.mark.parametrize(
    "value",
    [
        "2023-10-01",
        "2023-10-01T00:00:00",
        "2023-10-01T00:00:00Z",
        "2023-10-01T12:30:00+00:00",
    ],
)
def test_parse_neo4j_datetime_accepts_common_iso_formats(value: str) -> None:
    result = parse_neo4j_datetime(value)
    assert isinstance(result, DateTime)
    assert result.year == 2023
    assert result.month == 10
    assert result.day == 1


def test_parse_neo4j_datetime_rejects_garbage() -> None:
    with pytest.raises(Exception):
        parse_neo4j_datetime("not-a-date")
