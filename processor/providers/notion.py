"""
processor/providers/notion.py — Notion-specific processor worker.

Notion documents have a few quirks that affect scoring and extraction:
  - Pages can be very short (just a title + a few blocks) → tend to score low
  - Database pages contain structured properties in addition to body text
  - Notion's block format nests children — the ingestion worker already
    flattens these, so the text is pre-processed by the time we read it
  - Authors are referenced by Notion user IDs (not names) in metadata
"""

from __future__ import annotations

import logging

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)


class NotionProcessorWorker(BaseProcessorWorker):
    """
    Processor worker for Notion documents.

    Inherits the full scoring + extraction pipeline from BaseProcessorWorker.
    Adds Notion-specific content normalisation hooks.
    """

    PROVIDER = "notion"

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        """
        Pre-process Notion content before scoring.

        Adds the document title as a prefix if it isn't already embedded
        in the content, so the scoring agent has full context.
        """
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"

        # Skip very short Notion pages (< 100 chars after stripping)
        # by prepending a note — scoring agent will likely score them low.
        stripped = content.strip()
        if len(stripped) < 100:
            _log.debug(
                "Notion doc %s is very short (%d chars) — may score low",
                doc.id,
                len(stripped),
            )

        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        """
        Pre-process Notion content before extraction.

        Same as scoring hook: ensure title is present.
        """
        if doc.title and not content.startswith(doc.title):
            content = f"# {doc.title}\n\n{content}"
        return content
