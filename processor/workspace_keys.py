"""Resolve per-org Workspace LLM keys from the portal gateway."""

from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator

import httpx

from processor.config import config

_current_openai_key: ContextVar[str | None] = ContextVar(
    "processor_workspace_openai_key", default=None
)


class WorkspaceKeyError(RuntimeError):
    """Raised when a required workspace LLM key cannot be resolved."""


def current_openai_api_key() -> str | None:
    bound = _current_openai_key.get()
    if bound and bound.strip():
        return bound.strip()
    fallback = (config.openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    return fallback or None


def current_credentials_id() -> str:
    key = current_openai_api_key() or ""
    if not key:
        return ""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _portal_token(explicit: str | None = None) -> str:
    return (explicit or config.portal_token or "").strip()


async def fetch_workspace_llm_key(
    org_id: str,
    provider: str,
    *,
    portal_url: str | None = None,
    token: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    org = (org_id or "").strip()
    name = (provider or "").strip().lower()
    if not org:
        raise WorkspaceKeyError("org_id is required")
    if not name:
        raise WorkspaceKeyError("provider is required")

    base = (portal_url if portal_url is not None else config.portal_url).rstrip("/")
    if not base:
        fallback = current_openai_api_key() if name == "openai" else ""
        if fallback:
            return fallback
        raise WorkspaceKeyError("No OpenAI key configured")

    auth = _portal_token(token)
    if not auth:
        raise WorkspaceKeyError("Portal processor token is not configured")

    owns_client = client is None
    use_client = client or httpx.AsyncClient(timeout=10.0)
    try:
        try:
            response = await use_client.get(
                f"{base}/internal/llm-keys/{name}",
                headers={
                    "Authorization": f"Bearer {auth}",
                    "X-Eidetix-Org": org,
                },
            )
        except httpx.HTTPError as exc:
            raise WorkspaceKeyError(
                f"Could not read workspace {name} key from portal: {exc}"
            ) from exc
    finally:
        if owns_client:
            await use_client.aclose()

    if response.status_code == 404:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("detail") or "")
        except ValueError:
            detail = ""
        raise WorkspaceKeyError(
            detail or f"No {name} key configured in Workspace"
        )
    if response.status_code >= 400:
        raise WorkspaceKeyError(
            f"Portal workspace key lookup failed ({response.status_code})"
        )
    payload = response.json()
    key = str(payload.get("api_key") or "").strip()
    if not key:
        raise WorkspaceKeyError(f"No {name} key configured in Workspace")
    return key


@asynccontextmanager
async def use_workspace_openai_key(org_id: str) -> AsyncIterator[str]:
    key = await fetch_workspace_llm_key(org_id, "openai")
    token = _current_openai_key.set(key)
    try:
        yield key
    finally:
        _current_openai_key.reset(token)
