"""Tests for get_document_content id/s3_key resolution helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base.tools.query import (
    _doc_id_from_s3_key,
    _strip_s3_bucket_prefix,
    get_document_content,
)


class TestDocIdFromS3Key:
    def test_full_s3_key(self):
        key = "org:dev/notion/raw/PAGE-q2-2026-gtm-strategy-brief/v1.json"
        assert _doc_id_from_s3_key(key) == "doc:notion:PAGE-q2-2026-gtm-strategy-brief"

    def test_partial_s3_key_without_org(self):
        key = "notion/raw/PAGE-q2-2026-gtm-strategy-brief/v1.json"
        assert _doc_id_from_s3_key(key) == "doc:notion:PAGE-q2-2026-gtm-strategy-brief"

    def test_non_raw_key_returns_none(self):
        assert _doc_id_from_s3_key("orgs/myorg/docs/doc:001.txt") is None


class TestStripS3BucketPrefix:
    def test_strips_configured_bucket(self):
        with patch("base.tools.s3._bucket", return_value="eidetix"):
            stripped = _strip_s3_bucket_prefix(
                "eidetix/org:dev/notion/raw/PAGE-abc/v1.json"
            )
        assert stripped == "org:dev/notion/raw/PAGE-abc/v1.json"

    def test_leaves_key_without_bucket_unchanged(self):
        key = "org:dev/notion/raw/PAGE-abc/v1.json"
        with patch("base.tools.s3._bucket", return_value="eidetix"):
            assert _strip_s3_bucket_prefix(key) == key


class TestGetDocumentContentS3KeyFallback:
    def _doc_row(self) -> MagicMock:
        row = MagicMock()
        row.__getitem__ = lambda self, key: {
            "id": "doc:notion:PAGE-q2-2026-gtm-strategy-brief",
            "title": "Q2 2026 GTM Strategy Brief",
            "s3_key": "org:dev/notion/raw/PAGE-q2-2026-gtm-strategy-brief/v1.json",
            "source_scope": "scope:notion:org:dev",
        }[key]
        return row

    @pytest.mark.asyncio
    async def test_accepts_full_s3_key_with_bucket_prefix(self):
        pool = AsyncMock()
        doc_row = self._doc_row()

        async def fetchrow(query, value):
            if query.endswith("WHERE id = $1") and value == "doc:notion:PAGE-q2-2026-gtm-strategy-brief":
                return doc_row
            return None

        pool.fetchrow = AsyncMock(side_effect=fetchrow)

        with patch("base.tools.s3._bucket", return_value="eidetix"), patch(
            "base.tools.s3.read_document_content",
            new=AsyncMock(return_value="brief text"),
        ):
            result = await get_document_content(
                pool,
                "eidetix/org:dev/notion/raw/PAGE-q2-2026-gtm-strategy-brief/v1.json",
            )

        assert "error" not in result
        assert result["doc_id"] == "doc:notion:PAGE-q2-2026-gtm-strategy-brief"
        assert result["content"] == "brief text"

    @pytest.mark.asyncio
    async def test_accepts_partial_s3_key(self):
        pool = AsyncMock()
        doc_row = self._doc_row()

        async def fetchrow(query, value):
            if query.endswith("WHERE id = $1") and value == "doc:notion:PAGE-q2-2026-gtm-strategy-brief":
                return doc_row
            return None

        pool.fetchrow = AsyncMock(side_effect=fetchrow)

        with patch(
            "base.tools.s3.read_document_content",
            new=AsyncMock(return_value="brief text"),
        ):
            result = await get_document_content(
                pool,
                "notion/raw/PAGE-q2-2026-gtm-strategy-brief/v1.json",
            )

        assert "error" not in result
        assert result["doc_id"] == "doc:notion:PAGE-q2-2026-gtm-strategy-brief"

    @pytest.mark.asyncio
    async def test_chunk_uses_exact_durable_index_payload_text(self):
        pool = AsyncMock()
        doc_row = {
            "id": "doc:github-profile:acme/service/_profile/main",
            "title": "Repository profile",
            "s3_key": "org:acme/github-profile/raw/profile/v1.json",
            "source_scope": "scope:github:acme/service",
            "current_version": 1,
        }
        chunk_row = {"chunk_index": 0, "char_offset": 0, "char_length": 18}
        pool.fetchrow = AsyncMock(side_effect=[doc_row, chunk_row])
        pool.fetchval = AsyncMock(return_value="index-payload.json")

        with patch(
            "base.tools.s3.read_json_object",
            new=AsyncMock(
                return_value={
                    "chunks": [
                        {
                            "chunk_index": 0,
                            "char_offset": 0,
                            "char_length": 18,
                            "text": "# Rendered Profile",
                        }
                    ]
                }
            ),
        ), patch(
            "base.tools.s3.read_document_content",
            new=AsyncMock(return_value='{"schema_version":"github_repository_profile.v1"}'),
        ) as raw_reader:
            result = await get_document_content(
                pool,
                doc_row["id"],
                chunk_index=0,
            )

        assert result["chunk"]["text"] == "# Rendered Profile"
        raw_reader.assert_not_awaited()
