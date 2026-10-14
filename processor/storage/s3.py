"""
processor/storage/s3.py — Read raw document content from S3.

Documents are written by connectors in the format:
    {org_id}/{provider}/raw/{source_ref}/v{version}.json

The JSON payload structure is provider-specific but always contains a
`content` or `blocks` field. This module reads the raw bytes and returns
the text suitable for passing to scoring/extraction agents.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3
import botocore.exceptions

from processor.config import config

_log = logging.getLogger(__name__)

_s3: Any = None


def _s3_client() -> Any:
    global _s3
    if _s3 is None:
        kwargs: dict = {"region_name": config.aws_region}
        endpoint = os.environ.get("S3_ENDPOINT_URL")
        if endpoint:
            from botocore.config import Config

            kwargs["endpoint_url"] = endpoint
            kwargs["config"] = Config(s3={"addressing_style": "path"})
        _s3 = boto3.client("s3", **kwargs)
    return _s3


def _bucket() -> str:
    # Read live from env (supports both prefixed and bare names) so this works
    # even when the module was imported before env vars were set (e.g. in tests).
    bucket = (
        os.environ.get("PROCESSOR_AWS_S3_BUCKET")
        or os.environ.get("AWS_S3_BUCKET")
        or config.aws_s3_bucket
    )
    if not bucket:
        raise RuntimeError(
            "PROCESSOR_AWS_S3_BUCKET (or AWS_S3_BUCKET) env var is not set."
        )
    return bucket


async def read_document_content(s3_key: str) -> str:
    """
    Read and parse a document's raw content from S3.

    The stored JSON is provider-specific. We try common content fields in
    order: "content", "text", "body", "markdown", then fall back to a
    pretty-printed dump of the full JSON (so the agent always gets something).

    Args:
        s3_key: The S3 key from documents.s3_key
                (e.g., "org:acme-corp/notion/raw/PAGE-abc/v1.json").

    Returns:
        Extracted text content as a string.

    Raises:
        RuntimeError: If the object cannot be read.
    """
    import asyncio

    return await asyncio.to_thread(_read_sync, s3_key)


async def read_json_object(s3_key: str) -> dict[str, Any]:
    """Read one S3 object and require a JSON object payload."""
    import asyncio

    return await asyncio.to_thread(_read_json_sync, s3_key)


def _read_json_sync(s3_key: str) -> dict[str, Any]:
    try:
        response = _s3_client().get_object(Bucket=_bucket(), Key=s3_key)
        raw_bytes: bytes = response["Body"].read()
    except botocore.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        raise RuntimeError(f"S3 read failed for {s3_key!r}: {code}") from exc
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"S3 object {s3_key!r} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"S3 object {s3_key!r} must contain a JSON object")
    return payload


async def write_json_object(s3_key: str, content: bytes) -> None:
    """Write canonical JSON bytes to an explicit S3 key."""
    import asyncio

    await asyncio.to_thread(_write_json_sync, s3_key, content)


def _write_json_sync(s3_key: str, content: bytes) -> None:
    _s3_client().put_object(
        Bucket=_bucket(),
        Key=s3_key,
        Body=content,
        ContentType="application/json",
    )


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
        # Not JSON — treat raw bytes as text
        _log.warning("S3 object %s is not JSON, treating as plain text", s3_key)
        return raw_bytes.decode("utf-8", errors="replace")

    return _extract_text(payload, s3_key)


def _extract_text(payload: Any, s3_key: str) -> str:
    """Extract human-readable text from a provider-specific JSON payload."""
    if isinstance(payload, str):
        return payload

    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2)

    # Try common content field names in priority order
    for field in ("content", "text", "body", "markdown", "plain_text"):
        val = payload.get(field)
        if val and isinstance(val, str):
            return val

    # Notion-specific: blocks array with rich_text
    if "blocks" in payload:
        return _extract_notion_blocks(payload["blocks"])

    # GitHub PDF: sections array with headings
    if "sections" in payload:
        return _extract_github_pdf_sections(payload["sections"], payload.get("title"))

    # Slack-specific: messages array
    if "messages" in payload:
        return _extract_slack_messages(payload["messages"])

    # Fireflies-specific: full transcript payload
    if "transcript" in payload and isinstance(payload["transcript"], dict):
        return _extract_fireflies_transcript(payload)

    # Generic: title + metadata + full dump
    title = payload.get("title") or payload.get("name") or ""
    _log.debug("No content field found in S3 object %s, using full JSON dump", s3_key)
    prefix = f"Title: {title}\n\n" if title else ""
    return prefix + json.dumps(payload, indent=2)


def _extract_notion_blocks(blocks: list) -> str:
    """Flatten Notion block objects into plain text."""
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
    """Flatten Slack message objects into plain text."""
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        user = msg.get("user") or msg.get("username") or "unknown"
        text = msg.get("text") or msg.get("message") or ""
        if text:
            lines.append(f"[{user}]: {text}")
    return "\n".join(lines)


def _extract_fireflies_transcript(payload: dict) -> str:
    """
    Render a Fireflies transcript payload as structured plain text for the LLM.

    Layout:
      # <Meeting Title>
      Date: <date> | Duration: <N>min | Attendees: <names>

      ## Summary
      <overview>

      ### Action Items
      - <action items>

      ### Topics
      - <outline>

      ## Full Transcript
      [Speaker Name]: <sentence>
      ...

    Speaker names come directly from the sentences' `speaker_name` field.
    They are already alias-registered by the ingestion worker so the resolver
    chain matches them via alias-exact (conf 0.95) without any text substitution.
    """
    transcript = payload.get("transcript") or {}
    meta = payload.get("meta") or {}

    title = transcript.get("title") or meta.get("title") or "Untitled Meeting"
    date_ms = transcript.get("date") or meta.get("date_ms")
    duration_s = transcript.get("duration")

    parts: list[str] = [f"# {title}"]

    # Header line
    header_parts = []
    if date_ms:
        try:
            dt = datetime.fromtimestamp(date_ms / 1000, tz=timezone.utc)
            header_parts.append(f"Date: {dt.strftime('%Y-%m-%d %H:%M UTC')}")
        except Exception:
            pass
    if duration_s:
        try:
            header_parts.append(f"Duration: {int(duration_s) // 60}min")
        except Exception:
            pass

    attendee_names = _fireflies_attendee_names(transcript)
    if attendee_names:
        header_parts.append(f"Attendees: {', '.join(attendee_names)}")
    if header_parts:
        parts.append(" | ".join(header_parts))

    # Summary block — Fireflies may return action_items as a string or {person: [items]}.
    summary = transcript.get("summary") or {}
    overview = (summary.get("overview") or "").strip()
    action_items_raw = _format_fireflies_action_items(summary.get("action_items"))
    gist = (summary.get("gist") or "").strip()
    outline = (summary.get("outline") or "").strip()
    topics_discussed = (summary.get("topics_discussed") or "").strip()
    keywords_raw = summary.get("keywords") or []
    keywords: list[str] = keywords_raw if isinstance(keywords_raw, list) else [keywords_raw]

    if overview or gist or action_items_raw or outline or topics_discussed:
        parts.append("\n## Summary")
        if gist:
            parts.append(gist)
        if overview:
            parts.append("\n### Overview")
            parts.append(overview)
        if action_items_raw:
            parts.append("\n### Action Items")
            parts.append(action_items_raw)
        if topics_discussed:
            parts.append("\n### Topics Discussed")
            parts.append(topics_discussed)
        if outline:
            parts.append("\n### Outline")
            parts.append(outline)

    if keywords:
        parts.append(f"\nKeywords: {', '.join(k for k in keywords if k)}")

    # Full transcript
    sentences = transcript.get("sentences") or []
    if sentences:
        parts.append("\n## Full Transcript")
        current_speaker: str | None = None
        for sent in sentences:
            if not isinstance(sent, dict):
                continue
            speaker = (sent.get("speaker_name") or sent.get("speaker_id") or "").strip()
            text = (sent.get("raw_text") or "").strip()
            if not text:
                continue
            if speaker != current_speaker:
                current_speaker = speaker
                parts.append(f"\n[{speaker}]: {text}")
            else:
                parts.append(text)

    return "\n".join(parts)


def _fireflies_attendee_names(transcript: dict) -> list[str]:
    """Resolve attendee display labels from participants + meeting_attendees."""
    name_by_email: dict[str, str] = {}
    for attendee in transcript.get("meeting_attendees") or []:
        if not isinstance(attendee, dict):
            continue
        email = (attendee.get("email") or "").strip().lower()
        name = (attendee.get("displayName") or attendee.get("name") or "").strip()
        if email and name:
            name_by_email[email] = name

    seen: set[str] = set()
    names: list[str] = []

    def _add(label: str) -> None:
        key = label.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        names.append(label.strip())

    for entry in transcript.get("participants") or []:
        if isinstance(entry, str):
            for email in (e.strip() for e in entry.split(",") if e.strip()):
                _add(name_by_email.get(email.lower()) or email)
        elif isinstance(entry, dict):
            _add(
                entry.get("displayName")
                or entry.get("name")
                or entry.get("email")
                or ""
            )

    if not names:
        for attendee in transcript.get("meeting_attendees") or []:
            if not isinstance(attendee, dict):
                continue
            _add(
                attendee.get("displayName")
                or attendee.get("name")
                or attendee.get("email")
                or ""
            )

    return names


def _format_fireflies_action_items(raw: Any) -> str:
    """Normalize Fireflies action_items (string or person -> list mapping)."""
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        lines: list[str] = []
        for person, items in raw.items():
            if isinstance(items, list):
                if person:
                    lines.append(f"{person}:")
                for item in items:
                    if item:
                        lines.append(f"- {item}")
            elif items:
                lines.append(f"- {person}: {items}")
        return "\n".join(lines)
    return str(raw).strip()


def _extract_github_pdf_sections(sections: list, title: str | None = None) -> str:
    """Reconstruct readable text from GitHub PDF sections payload.

    Each section: {"heading": str, "text": str}
    Headings are formatted as markdown ## headings so the chunker's structural
    splitter can treat them as section boundaries.
    """
    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading", "").strip()
        text = section.get("text", "").strip()
        if heading:
            parts.append(f"## {heading}")
        if text:
            parts.append(text)
    return "\n\n".join(parts)
