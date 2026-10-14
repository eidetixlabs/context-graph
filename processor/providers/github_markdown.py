"""
processor/providers/github_markdown.py — GitHub Markdown processor worker.

Processes documents with source_provider='github-markdown'.

S3 payload written by the GitHub ingestion worker:
    {
        "title": "README.md",
        "markdown": "# ...",
        "meta": {"repo": ..., "path": ..., "blob_sha": ..., "fetched_at": ...}
    }

The S3 reader (processor/storage/s3.py) returns the "markdown" field directly
(already handled by generic field-priority extraction). The hooks here prepend
the document title so agents have full context.
"""

from __future__ import annotations

import logging

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)


class GithubMarkdownProcessorWorker(BaseProcessorWorker):
    """
    Processor worker for GitHub Markdown documents.

    Inherits the full scoring + extraction pipeline from BaseProcessorWorker.
    Adds title-prefix normalization so agents always see the markdown filename.
    """

    PROVIDER = "github-markdown"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"
        stripped = content.strip()
        if len(stripped) < 100:
            _log.debug(
                "github-markdown doc %s is very short (%d chars) — may score low",
                doc.id, len(stripped),
            )
        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"
        return content
