"""
processor/providers/gmail.py — Gmail-specific processor worker.

Gmail raw payloads are structured JSON. For extraction quality we render the
payload into concise text that keeps participant metadata and substantive body
content while dropping transport/header noise.
"""

from __future__ import annotations

import json
from email.utils import getaddresses

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker


def _gmail_to_extraction_text(doc: DocumentRecord, content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        # If content is already plain text (or malformed), keep existing behavior.
        if doc.title and not content.startswith(doc.title) and not content.startswith(f"# {doc.title}"):
            return f"# {doc.title}\n\n{content}"
        return content

    if not isinstance(payload, dict):
        return content

    headers = payload.get("headers")
    body = payload.get("body")
    if not isinstance(headers, dict):
        headers = {}
    if not isinstance(body, dict):
        body = {}

    subject = str(headers.get("subject") or doc.title or "Gmail message").strip()
    sender = str(headers.get("from") or "").strip()
    date = str(headers.get("date") or "").strip()
    # Prefer body.normalized; fall back to legacy plain_text for older S3 payloads.
    primary_body = str(body.get("normalized") or body.get("plain_text") or "").strip()

    to_values = headers.get("to") or []
    cc_values = headers.get("cc") or []
    to_list = to_values if isinstance(to_values, list) else [str(to_values)]
    cc_list = cc_values if isinstance(cc_values, list) else [str(cc_values)]
    to_text = ", ".join(v.strip() for v in to_list if str(v).strip())
    cc_text = ", ".join(v.strip() for v in cc_list if str(v).strip())

    participants: list[str] = []
    seen_emails: set[str] = set()
    for _display_name, email in getaddresses([sender, to_text, cc_text]):
        normalized = email.strip().lower()
        if not normalized or normalized in seen_emails:
            continue
        seen_emails.add(normalized)
        participants.append(normalized)

    lines: list[str] = [f"# {subject}"]
    if sender:
        lines.append(f"From: {sender}")
    if to_text:
        lines.append(f"To: {to_text}")
    if cc_text:
        lines.append(f"Cc: {cc_text}")
    if date:
        lines.append(f"Date: {date}")
    if participants:
        lines.append(f"Participants: {', '.join(participants)}")

    if primary_body:
        lines.append("")
        lines.append("## Message")
        lines.append(primary_body)

    rendered = "\n".join(lines).strip()
    return rendered or content


class GmailProcessorWorker(BaseProcessorWorker):
    """Processor worker for Gmail documents."""

    PROVIDER = "gmail"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        return _gmail_to_extraction_text(doc, content)

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        return _gmail_to_extraction_text(doc, content)
