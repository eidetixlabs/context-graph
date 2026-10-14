"""Parse ISO8601 temporal strings for Neo4j datetime properties."""

from __future__ import annotations


def parse_neo4j_datetime(value: str):
    """Parse an ISO8601 date or datetime string into neo4j.time.DateTime.

    Accepts full datetimes (with optional Z suffix) and date-only YYYY-MM-DD
    strings (coerced to midnight).
    """
    from neo4j.time import Date, DateTime  # noqa: PLC0415

    normalized = f"{value[:-1]}+00:00" if value.endswith(("Z", "z")) else value
    try:
        return DateTime.fromisoformat(normalized)
    except Exception:
        pass

    if "T" in value:
        try:
            return DateTime.fromisoformat(normalized)
        except Exception:
            pass

    d = Date.fromisoformat(value)
    return DateTime(d.year, d.month, d.day, 0, 0, 0)
