"""Lazy storage exports to keep S3-only workers lightweight."""

from __future__ import annotations

from typing import Any

__all__ = [
    "read_document_content",
    "read_json_object",
    "write_extraction_result",
    "write_json_object",
]


def __getattr__(name: str) -> Any:
    if name in {"read_document_content", "read_json_object", "write_json_object"}:
        from processor.storage import s3

        return getattr(s3, name)
    if name == "write_extraction_result":
        from processor.storage.ingest import write_extraction_result

        return write_extraction_result
    raise AttributeError(name)
