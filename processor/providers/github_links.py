"""
processor/providers/github_links.py — GitHub Links processor worker.

Processes documents with source_provider='github-links'.

S3 payload written by the GitHub ingestion worker:
    {
        "url": "https://...",
        "title": "Page title",
        "text": "Main content extracted by trafilatura",
        "meta": {"repo": ..., "fetched_at": ...}
    }

The S3 reader (processor/storage/s3.py) returns the "text" field directly
(already handled by the generic field-priority check). The hooks here prepend
the title and source URL so agents have full context.
"""

from __future__ import annotations

import logging

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)


class GithubLinksProcessorWorker(BaseProcessorWorker):
    """
    Processor worker for GitHub Links documents (crawled URLs).

    Inherits the full scoring + extraction pipeline from BaseProcessorWorker.
    Prepends title + URL so extraction agents have source context.
    """

    PROVIDER = "github-links"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        content = _prefix_title_and_url(doc, content)
        stripped = content.strip()
        if len(stripped) < 100:
            _log.debug(
                "github-links doc %s is very short (%d chars) — may score low",
                doc.id, len(stripped),
            )
        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        return _prefix_title_and_url(doc, content)


def _prefix_title_and_url(doc: DocumentRecord, content: str) -> str:
    """Prepend 'Title: ...\nSource: ...\n\n' if not already present."""
    prefix_parts: list[str] = []
    if doc.title and not content.startswith(doc.title):
        prefix_parts.append(f"# {doc.title}")
    # source_ref is the SHA256 of the URL — not human-readable.
    # The actual URL is in the content payload's "url" field, which the S3
    # reader doesn't expose. The title is the best available hint here.
    if prefix_parts:
        content = "\n".join(prefix_parts) + "\n\n" + content
    return content
