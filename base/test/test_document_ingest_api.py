from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from base.tools.document_ingest import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentNotRetryableError,
    build_portal_payload,
    persist_portal_document,
    portal_document_id,
    retry_portal_document,
)
from processor.storage.s3 import _extract_text
from webserver import main as webserver_main
from webserver.main import (
    DocumentIngestPayload,
    _document_status,
    _require_document_write,
    ingest_document,
    retry_ingested_document,
)


class _AsyncContext:
    def __init__(self, value=None) -> None:
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_args):
        return False


def _pool_and_conn():
    conn = AsyncMock()
    conn.transaction = MagicMock(return_value=_AsyncContext())
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_AsyncContext(conn))
    return pool, conn


@pytest.mark.parametrize(
    "doc_type",
    ["architecture", "hld", "lld", "design", "roadmap", "ticket_retrospective", "context"],
)
def test_document_payload_accepts_business_context_types(doc_type: str) -> None:
    payload = DocumentIngestPayload(
        repo="acme/widgets",
        doc_type=doc_type,
        title="Context",
        content="Narrative architecture and business context.",
    )
    assert payload.doc_type == doc_type


@pytest.mark.parametrize("doc_type", ["source_code", "code", "patch", "adr"])
def test_document_payload_rejects_code_specific_or_unknown_types(doc_type: str) -> None:
    with pytest.raises(ValidationError):
        DocumentIngestPayload(
            repo="acme/widgets",
            doc_type=doc_type,
            title="Context",
            content="Narrative context.",
        )


def test_document_payload_validates_repo_content_and_source_ref() -> None:
    with pytest.raises(ValidationError):
        DocumentIngestPayload(
            repo="not-owner-name",
            doc_type="context",
            title="Context",
            content="Narrative context.",
        )
    with pytest.raises(ValidationError):
        DocumentIngestPayload(
            repo="acme/widgets",
            doc_type="context",
            title="Context",
            content=" ",
        )
    with pytest.raises(ValidationError):
        DocumentIngestPayload(
            repo="acme/widgets",
            doc_type="context",
            title="Context",
            content="Narrative context.",
            source_ref="../unsafe",
        )


def test_read_only_service_principal_is_denied_document_ingest() -> None:
    with pytest.raises(HTTPException) as exc:
        _require_document_write(
            {"principal_type": "service", "auth_scopes": ["mcp:read"]}
        )
    assert exc.value.status_code == 403


def test_wildcard_service_principal_can_write_documents() -> None:
    _require_document_write({"principal_type": "service", "auth_scopes": ["*"]})


def test_document_payload_allows_org_wide_ingest_without_repo() -> None:
    payload = DocumentIngestPayload(
        doc_type="context",
        title="Shared policy",
        content="All services use the same incident severity definitions.",
    )
    assert payload.repo is None


def test_portal_document_ids_are_org_scoped() -> None:
    assert portal_document_id("org:acme", "architecture-v1") == (
        "doc:portal:org:acme:architecture-v1"
    )
    assert portal_document_id("org:acme", "architecture-v1") != portal_document_id(
        "org:beta", "architecture-v1"
    )


@pytest.mark.asyncio
async def test_document_ingest_maps_source_ref_conflict_to_safe_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        webserver_main,
        "persist_portal_document",
        AsyncMock(side_effect=DocumentConflictError("database details must not leak")),
    )
    payload = DocumentIngestPayload(
        repo="acme/widgets",
        doc_type="architecture",
        title="Architecture",
        content="The gateway owns authentication.",
        source_ref="architecture-main",
    )
    principal = {
        "id": "emp:author",
        "org_id": "org:acme",
        "principal_type": "employee",
        "resolved_org_ctx": SimpleNamespace(org_id="org:acme", pool=AsyncMock()),
    }

    with pytest.raises(HTTPException) as exc:
        await ingest_document(payload, principal, None)

    assert exc.value.status_code == 409
    assert exc.value.detail == "source_ref is already used by another document"
    assert "database details" not in exc.value.detail


@pytest.mark.asyncio
async def test_document_ingest_maps_storage_failure_to_safe_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        webserver_main,
        "persist_portal_document",
        AsyncMock(side_effect=RuntimeError("s3 credentials must not leak")),
    )
    payload = DocumentIngestPayload(
        repo="acme/widgets",
        doc_type="architecture",
        title="Architecture",
        content="The gateway owns authentication.",
    )
    principal = {
        "id": "emp:author",
        "org_id": "org:acme",
        "principal_type": "employee",
        "resolved_org_ctx": SimpleNamespace(org_id="org:acme", pool=AsyncMock()),
    }

    with pytest.raises(HTTPException) as exc:
        await ingest_document(payload, principal, None)

    assert exc.value.status_code == 503
    assert exc.value.detail == "Document storage is temporarily unavailable."
    assert "credentials" not in exc.value.detail


def test_portal_payload_content_is_selected_for_processor_extraction() -> None:
    raw = build_portal_payload(
        repo="acme/widgets",
        doc_type="architecture",
        title="Architecture",
        content="The gateway owns authentication.",
        source_ref="architecture-main",
    )

    assert _extract_text(
        json.loads(raw),
        "org:acme/portal/raw/architecture-main/v1.json",
    ) == "The gateway owns authentication."


@pytest.mark.asyncio
async def test_document_ingest_writes_raw_s3_then_reuses_connector_upsert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(return_value=None)
    writer = MagicMock(return_value="org:acme/portal/raw/architecture-v1/v1.json")
    upsert = AsyncMock()
    monkeypatch.setattr("base.tools.document_ingest.write_raw_s3", writer)
    monkeypatch.setattr("base.tools.document_ingest.upsert_document", upsert)

    result = await persist_portal_document(
        pool=pool,
        org_id="org:acme",
        employee_id="emp:author",
        repo="acme/widgets",
        doc_type="architecture",
        title="Architecture",
        content="The gateway owns authentication.",
        source_ref="architecture-v1",
    )

    assert result["id"] == "doc:portal:org:acme:architecture-v1"
    assert result["processing_state"] == "raw"
    assert result["status"] == "processing"
    assert result["version"] == 1
    assert result["unchanged"] is False
    writer.assert_called_once()
    upsert.assert_awaited_once()
    assert upsert.await_args.kwargs["provider"] == "portal"
    assert upsert.await_args.kwargs["source_scope"] == "scope:github:acme/widgets"
    metadata_updates = [
        call for call in conn.execute.await_args_list if "SET repo" in call.args[0]
    ]
    assert metadata_updates
    assert metadata_updates[0].args[2:4] == ("acme/widgets", "architecture")


@pytest.mark.asyncio
async def test_document_ingest_is_idempotent_for_unchanged_source_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = build_portal_payload(
        repo="acme/widgets",
        doc_type="hld",
        title="HLD",
        content="Narrative only",
        source_ref="hld-main",
    )
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": "doc:portal:org:acme:hld-main",
            "current_version": 3,
            "content_hash": hashlib.sha256(raw).hexdigest(),
            "processing_state": "extracted",
        }
    )
    writer = MagicMock()
    upsert = AsyncMock()
    monkeypatch.setattr("base.tools.document_ingest.write_raw_s3", writer)
    monkeypatch.setattr("base.tools.document_ingest.upsert_document", upsert)

    result = await persist_portal_document(
        pool=pool,
        org_id="org:acme",
        employee_id="emp:author",
        repo="acme/widgets",
        doc_type="hld",
        title="HLD",
        content="Narrative only",
        source_ref="hld-main",
    )

    assert result["version"] == 3
    assert result["status"] == "ready"
    assert result["unchanged"] is True
    writer.assert_not_called()
    upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_ingest_versions_changed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": "doc:portal:org:acme:architecture-main",
            "current_version": 2,
            "content_hash": "old-hash",
            "processing_state": "extracted",
        }
    )
    writer = MagicMock(
        return_value="org:acme/portal/raw/architecture-main/v3.json"
    )
    upsert = AsyncMock()
    monkeypatch.setattr("base.tools.document_ingest.write_raw_s3", writer)
    monkeypatch.setattr("base.tools.document_ingest.upsert_document", upsert)

    result = await persist_portal_document(
        pool=pool,
        org_id="org:acme",
        employee_id="emp:author",
        repo="acme/widgets",
        doc_type="architecture",
        title="Architecture",
        content="The gateway now delegates authentication.",
        source_ref="architecture-main",
    )

    assert result["version"] == 3
    assert result["processing_state"] == "raw"
    assert upsert.await_args.kwargs["version"] == 3
    assert upsert.await_args.kwargs["s3_key"].endswith("/v3.json")


@pytest.mark.parametrize(
    ("processing_state", "expected"),
    [
        ("raw", "processing"),
        ("indexing", "processing"),
        ("extracted", "ready"),
        ("skipped", "skipped"),
        ("failed", "failed"),
    ],
)
def test_document_status_contract(processing_state: str, expected: str) -> None:
    assert _document_status(processing_state) == expected


@pytest.mark.asyncio
async def test_retry_resets_failed_portal_document_to_raw() -> None:
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(
        return_value={
            "id": "doc:portal:org:acme:architecture-v1",
            "repo": "acme/widgets",
            "doc_type": "architecture",
            "title": "Architecture",
            "source_ref": "architecture-v1",
            "current_version": 2,
            "processing_state": "raw",
        }
    )

    result = await retry_portal_document(
        pool=pool,
        org_id="org:acme",
        document_id="doc:portal:org:acme:architecture-v1",
    )

    assert result["id"] == "doc:portal:org:acme:architecture-v1"
    assert result["status"] == "processing"
    assert result["processing_state"] == "raw"
    assert result["unchanged"] is False
    update_sql = conn.fetchrow.await_args_list[0].args[0]
    assert "processing_state = 'raw'" in update_sql
    assert "extraction_error = NULL" in update_sql
    assert conn.fetchrow.await_args_list[0].args[1:] == (
        "doc:portal:org:acme:architecture-v1",
        "org:acme",
        ["failed", "skipped"],
    )


@pytest.mark.asyncio
async def test_retry_rejects_in_progress_portal_document() -> None:
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(
        side_effect=[None, {"processing_state": "extracting"}]
    )

    with pytest.raises(DocumentNotRetryableError) as exc:
        await retry_portal_document(
            pool=pool,
            org_id="org:acme",
            document_id="doc:portal:org:acme:architecture-v1",
        )

    assert exc.value.processing_state == "extracting"


@pytest.mark.asyncio
async def test_retry_rejects_unknown_portal_document() -> None:
    pool, conn = _pool_and_conn()
    conn.fetchrow = AsyncMock(return_value=None)

    with pytest.raises(DocumentNotFoundError):
        await retry_portal_document(
            pool=pool,
            org_id="org:acme",
            document_id="doc:portal:org:acme:missing",
        )


@pytest.mark.asyncio
async def test_retry_endpoint_maps_missing_document_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        webserver_main,
        "retry_portal_document",
        AsyncMock(side_effect=DocumentNotFoundError("document not found")),
    )
    principal = {
        "id": "emp:author",
        "org_id": "org:acme",
        "principal_type": "employee",
        "resolved_org_ctx": SimpleNamespace(org_id="org:acme", pool=AsyncMock()),
    }

    with pytest.raises(HTTPException) as exc:
        await retry_ingested_document("doc:portal:org:acme:missing", principal)

    assert exc.value.status_code == 404
    assert exc.value.detail == "document not found"


@pytest.mark.asyncio
async def test_retry_endpoint_maps_busy_document_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        webserver_main,
        "retry_portal_document",
        AsyncMock(side_effect=DocumentNotRetryableError("extracting")),
    )
    principal = {
        "id": "emp:author",
        "org_id": "org:acme",
        "principal_type": "employee",
        "resolved_org_ctx": SimpleNamespace(org_id="org:acme", pool=AsyncMock()),
    }

    with pytest.raises(HTTPException) as exc:
        await retry_ingested_document("doc:portal:org:acme:architecture-v1", principal)

    assert exc.value.status_code == 409
    assert "cannot be retried" in str(exc.value.detail)
