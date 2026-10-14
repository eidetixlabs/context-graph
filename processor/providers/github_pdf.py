"""
processor/providers/github_pdf.py — GitHub PDF processor worker.

Processes documents with source_provider='github-pdf'.

S3 payload written by the GitHub ingestion worker:
    {
        "title": "filename.pdf",
        "sections": [{"heading": str, "text": str}, ...],
        "meta": {"repo": ..., "path": ..., "blob_sha": ..., "fetched_at": ...}
    }

The S3 reader (processor/storage/s3.py) extracts text from "sections" into
a flat markdown-ish string. The hooks here prepend the document title so
scoring/extraction agents have full context — same pattern as NotionProcessorWorker.
"""

from __future__ import annotations

import logging

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)


class GithubPdfProcessorWorker(BaseProcessorWorker):
    """
    Processor worker for GitHub PDF documents.

    Inherits the full scoring + extraction pipeline from BaseProcessorWorker.
    Adds title-prefix normalisation so agents always see the document name.
    """

    PROVIDER = "github-pdf"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"
        stripped = content.strip()
        if len(stripped) < 100:
            _log.debug(
                "github-pdf doc %s is very short (%d chars) — may score low",
                doc.id, len(stripped),
            )
        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"
        return content
