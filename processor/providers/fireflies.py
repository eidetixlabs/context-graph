"""
processor/providers/fireflies.py — Fireflies-specific processor worker.

Fireflies meeting transcripts have a number of properties that distinguish
them from standard document types:

1. Full content (summary + sentences) is fed to the LLM as a single shot
   (no chunking for extraction). The whole meeting fits in one LLM call and
   seeing the full context prevents spurious contradictions ("decided X then
   rejected X" is resolved naturally when the model reads both).

2. Attendees have already been resolved to Postgres entities (by email, conf 1.0)
   during ingestion. Their canonical names appear in the transcript text via
   speaker_name, and because resolve_or_create_identity writes aliases, the
   resolver chain will match them at alias-exact confidence (0.95).

3. Retrieval chunking (for PgVector / recall()) still runs on the normal
   structural chunker so small, searchable chunks are stored even though
   extraction sees the whole document.

TODO: add a deterministic Fireflies transformer (model after
    processor/trace/transformer.py) that converts structured Fireflies fields
    (action_items[], participants[], outline[]) directly into nodes, bypassing
    the LLM for those fields. This adds schema-drift resilience and ensures
    action items are never missed due to LLM discretion. Wire it in
    on_extract_start before returning the content to the pipeline.
"""

from __future__ import annotations

import logging

from processor.state_machine import DocumentRecord
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)


class FirefliesProcessorWorker(BaseProcessorWorker):
    """
    Processor worker for Fireflies meeting transcripts.

    Inherits the full scoring + extraction pipeline from BaseProcessorWorker.
    Adds Fireflies-specific content normalisation hooks and enables
    whole-document (single-shot) extraction.
    """

    PROVIDER = "fireflies"

    # Whole-document (single-shot) extraction is configured in
    # processor/agents/extraction/pipeline.py::_WHOLE_DOCUMENT_PROVIDERS.
    # Adding a class attribute here has no effect — BaseProcessorWorker does
    # not read EXTRACTION_CHUNKING_STRATEGY.

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        """
        Pre-process Fireflies content before scoring.

        The S3 reader already renders the transcript into structured plain text
        (title, summary, full transcript). Add the title prefix if not present.
        """
        if doc.title and not content.startswith(doc.title) and not content.startswith(f"# {doc.title}"):
            content = f"# {doc.title}\n\n{content}"
        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        """
        Pre-process Fireflies content before extraction.

        The S3 reader has already:
        - Formatted the transcript with speaker names ([Speaker Name]: text)
        - Rendered the summary/action-items/topics block

        Attendees were resolved by email (conf 1.0) during ingestion, and
        resolve_or_create_identity wrote their names as aliases, so the
        resolver chain will match speaker names at alias-exact confidence.

        We just ensure the title is present so the LLM has full context.
        """
        if doc.title and not content.startswith(doc.title) and not content.startswith(f"# {doc.title}"):
            content = f"# {doc.title}\n\n{content}"
        return content
