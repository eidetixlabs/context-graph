"""
base/tools/s3.py — Thin S3 content reader for the webserver layer.

Mirrors the logic in processor/storage/s3.py but reads config purely from
environment variables so that the webserver (which does not import the
processor package) can fetch document content without a circular dependency.

Environment variables:
    AWS_S3_BUCKET   (or PROCESSOR_AWS_S3_BUCKET) — bucket name
    AWS_REGION      (or AWS_DEFAULT_REGION)       — AWS region (default us-east-1)
    S3_ENDPOINT_URL — optional, for local MinIO / test overrides
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import boto3
import botocore.exceptions

_log = logging.getLogger(__name__)

_s3: Any = None
_bucket_ensured: bool = False


def _s3_client() -> Any:
    global _s3
    if _s3 is None:
        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        kwargs: dict = {"region_name": region}
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            from botocore.config import Config

            kwargs["endpoint_url"] = endpoint
            kwargs["config"] = Config(s3={"addressing_style": "path"})
        _s3 = boto3.client("s3", **kwargs)
    return _s3


def _bucket() -> str:
    bucket = (
        os.environ.get("PROCESSOR_AWS_S3_BUCKET")
        or os.environ.get("AWS_S3_BUCKET")
    )
    if not bucket:
        raise RuntimeError(
            "AWS_S3_BUCKET (or PROCESSOR_AWS_S3_BUCKET) env var is not set."
        )
    return bucket


def ensure_bucket() -> None:
    """Create the configured bucket if it does not exist (MinIO / local dev)."""
    global _bucket_ensured
    if _bucket_ensured:
        return
    bucket = _bucket()
    client = _s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchBucket", "NotFound"):
            raise
        try:
            client.create_bucket(Bucket=bucket)
            _log.info("Created S3 bucket %r", bucket)
        except botocore.exceptions.ClientError as create_exc:
            create_code = create_exc.response.get("Error", {}).get("Code", "")
            if create_code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise
    _bucket_ensured = True


async def ensure_bucket_async() -> None:
    await asyncio.to_thread(ensure_bucket)


async def read_document_content(s3_key: str) -> str:
    """Read and extract text content from a document stored in S3.

    Runs the blocking S3 call in a thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(_read_sync, s3_key)


async def read_json_object(s3_key: str) -> dict[str, Any]:
    """Read an S3 object as JSON without provider text extraction."""
    return await asyncio.to_thread(_read_json_sync, s3_key)


def _read_json_sync(s3_key: str) -> dict[str, Any]:
    try:
        resp = _s3_client().get_object(Bucket=_bucket(), Key=s3_key)
        raw_bytes: bytes = resp["Body"].read()
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise RuntimeError(f"S3 read failed for {s3_key!r}: {code}") from exc
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"S3 object {s3_key!r} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"S3 object {s3_key!r} is not a JSON object")
    return payload


def _read_sync(s3_key: str) -> str:
    try:
        resp = _s3_client().get_object(Bucket=_bucket(), Key=s3_key)
        raw_bytes: bytes = resp["Body"].read()
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise RuntimeError(f"S3 read failed for {s3_key!r}: {code}") from exc

    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError:
        _log.warning("S3 object %s is not JSON — treating as plain text", s3_key)
        return raw_bytes.decode("utf-8", errors="replace")

    return _extract_text(payload, s3_key)


def _extract_text(payload: Any, s3_key: str) -> str:
    if isinstance(payload, str):
        return payload

    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2)

    for field in ("content", "text", "body", "markdown", "plain_text"):
        val = payload.get(field)
        if val and isinstance(val, str):
            return val

    if "blocks" in payload:
        return _extract_notion_blocks(payload["blocks"])

    if "messages" in payload:
        return _extract_slack_messages(payload["messages"])

    title = payload.get("title") or payload.get("name") or ""
    _log.debug("No content field found in S3 object %s — using full JSON dump", s3_key)
    prefix = f"Title: {title}\n\n" if title else ""
    return prefix + json.dumps(payload, indent=2)


def _extract_notion_blocks(blocks: list) -> str:
    lines: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        content = block.get(block_type, {})
        rich_text = content.get("rich_text", [])
        text = "".join(
            rt.get("plain_text", "") or rt.get("text", {}).get("content", "")
            for rt in rich_text
            if isinstance(rt, dict)
        )
        if text:
            lines.append(text)
    return "\n".join(lines)


def _extract_slack_messages(messages: list) -> str:
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        user = msg.get("user") or msg.get("username") or "unknown"
        text = msg.get("text") or msg.get("message") or ""
        if text:
            lines.append(f"[{user}]: {text}")
    return "\n".join(lines)


def write_json(s3_key: str, data: Any) -> None:
    """Write a JSON-serialisable object to S3."""
    ensure_bucket()
    body = json.dumps(data, default=str).encode("utf-8")
    _s3_client().put_object(
        Bucket=_bucket(),
        Key=s3_key,
        Body=body,
        ContentType="application/json",
    )


async def write_json_async(s3_key: str, data: Any) -> None:
    """Async wrapper around write_json — runs in a thread to avoid blocking the event loop."""
    await asyncio.to_thread(write_json, s3_key, data)


def list_keys(prefix: str) -> list[str]:
    """List all S3 keys under a prefix (handles pagination)."""
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def read_json(s3_key: str) -> Any:
    """Read and JSON-parse an S3 object."""
    resp = _s3_client().get_object(Bucket=_bucket(), Key=s3_key)
    return json.loads(resp["Body"].read())

async def list_keys_async(prefix: str) -> list[str]:
    return await asyncio.to_thread(list_keys, prefix)


async def read_json_async(s3_key: str) -> Any:
    return await asyncio.to_thread(read_json, s3_key)
