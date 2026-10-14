"""
processor/worker.py — BaseProcessorWorker and polling loop.

One worker instance per source_provider. Every poll cycle:
  1. Reset stale in-progress documents (from previous crashed runs)
  2. Claim 'raw' documents → run scoring → persist state (scored/skipped/failed)
  3. Claim 'scored' documents → run extraction pipeline → persist state (extracted/failed)
  4. Sleep for poll_interval_seconds

Usage:
    # Direct (development)
    python -m processor.worker --provider notion

    # Subclass (production)
    class NotionProcessorWorker(BaseProcessorWorker):
        PROVIDER = "notion"

    asyncio.run(NotionProcessorWorker().run())
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import traceback
from abc import ABC
from pathlib import Path
from typing import NamedTuple

# Ensure repo root is on PYTHONPATH when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from base.db.client import db_pool
from base.log_level import log_level_from_env
from processor.agents.extraction.pipeline import run_extraction_pipeline
from processor.agents.scoring import run_scoring
from processor.config import config
from processor.workspace_keys import use_workspace_openai_key
from processor.dual_write import IndexingIncompleteError
from processor.schemas.scoring import ScoringResult
from processor.state_machine import (
    DocumentRecord,
    StaleDocumentWork,
    claim_documents,
    reset_stale_in_progress,
    transition_extracted,
    transition_failed,
    transition_scored,
)
from processor.storage.ingest import _neo4j_client
from processor.storage.s3 import read_document_content
from processor.tools.eidetix import close_shared_client

_log = logging.getLogger(__name__)


class CycleStats(NamedTuple):
    """Per-poll-cycle batch sizes and extraction outcomes."""

    indexing_batch: int
    indexed_ok: int
    scoring_batch: int
    extraction_batch: int
    extracted_ok: int


class BaseProcessorWorker(ABC):
    """
    Abstract base for per-provider processor workers.

    Subclass and set PROVIDER = "notion" (or other provider string).
    Override score_hook() / extract_hook() for provider-specific pre/post
    processing without touching the core pipeline.
    """

    PROVIDER: str = ""

    def __init__(self) -> None:
        if not self.PROVIDER:
            raise ValueError(
                f"{self.__class__.__name__} must set PROVIDER class attribute."
            )
        # Cache scoring results within a cycle so extraction doesn't re-score.
        # Keys are document IDs; entries are evicted after successful extraction.
        self._scoring_cache: dict[str, ScoringResult] = {}

    # ── Hooks (override in subclasses for provider-specific behaviour) ────────

    async def on_score_start(self, doc: DocumentRecord, content: str) -> str:
        """Called before scoring. May transform content (e.g., strip HTML). Returns content."""
        return content

    async def on_extract_start(self, doc: DocumentRecord, content: str) -> str:
        """Called before extraction. May transform content. Returns content."""
        return content

    async def score_content(
        self,
        doc: DocumentRecord,
        content: str,
    ) -> ScoringResult:
        """Return the admission score for a document.

        Providers with authoritative structured content may override this hook
        with a deterministic score instead of invoking the generic scorer.
        """
        return await run_scoring(content, title=doc.title)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def score_document(self, doc: DocumentRecord) -> None:
        """Score a single document and persist the result."""
        try:
            async with use_workspace_openai_key(doc.org_id):
                await self._score_document_with_workspace_key(doc)
        except StaleDocumentWork:
            _log.info(
                "Scoring cancelled for stale document work: %s@v%s",
                doc.id,
                doc.current_version,
            )
        except Exception as exc:
            _log.error(
                "Scoring failed for doc=%s:\n%s", doc.id, traceback.format_exc()
            )
            try:
                await transition_failed(
                    doc.id,
                    doc.current_version,
                    "scoring",
                    f"scoring error: {exc}",
                )
            except StaleDocumentWork:
                _log.info("Discarded stale scoring failure for %s", doc.id)

    async def _score_document_with_workspace_key(self, doc: DocumentRecord) -> None:
        content = await read_document_content(doc.s3_key)
        content = await self.on_score_start(doc, content)

        scoring_result = await self.score_content(doc, content)

        # Cache so extract_document can use it without re-scoring
        self._scoring_cache[doc.id] = scoring_result

        new_state = await transition_scored(
            doc_id=doc.id,
            document_version=doc.current_version,
            score=scoring_result.signal_score,
            threshold=config.signal_score_threshold,
        )

        _log.info(
            "Scored: doc=%s score=%.3f → %s",
            doc.id,
            scoring_result.signal_score,
            new_state,
        )

    async def extract_document(self, doc: DocumentRecord) -> bool:
        """Run the three-phase extraction pipeline on a single document. True if extracted."""
        try:
            async with use_workspace_openai_key(doc.org_id):
                return await self._extract_document_with_workspace_key(doc)

        except (IndexingIncompleteError, StaleDocumentWork):
            _log.warning(
                "Extraction indexing incomplete or stale for doc=%s",
                doc.id,
            )
            return False
        except Exception as exc:
            _log.error(
                "Extraction failed for doc=%s:\n%s", doc.id, traceback.format_exc()
            )
            try:
                await transition_failed(
                    doc.id,
                    doc.current_version,
                    "extracting",
                    f"extraction error: {exc}",
                )
            except StaleDocumentWork:
                _log.info("Discarded stale extraction failure for %s", doc.id)
            return False

    async def _extract_document_with_workspace_key(self, doc: DocumentRecord) -> bool:
        content = await read_document_content(doc.s3_key)
        content = await self.on_extract_start(doc, content)

        # Use cached scoring result from this cycle when available.
        # Documents scored in a prior cycle (cache miss) get a fresh score —
        # cheap Haiku call; the stored signal_score in Postgres is used only
        # for the skip decision (already made), so this is only for topic hints.
        # TODO - this won't work when processor pods will be scaled.
        scoring_result = self._scoring_cache.pop(doc.id, None)
        if scoring_result is None:
            _log.debug("No cached scoring for doc=%s — re-scoring for topic hints", doc.id)
            scoring_result = await self.score_content(doc, content)

        extraction_result = await run_extraction_pipeline(
            doc=doc,
            content=content,
            scoring_result=scoring_result,
            neo4j_client=_neo4j_client(),
        )

        await transition_extracted(
            doc.id,
            doc.current_version,
            model=config.extraction_model,
        )

        _log.info(
            "Extracted: doc=%s nodes=%d edges=%d model=%s",
            doc.id,
            len(extraction_result.nodes),
            len(extraction_result.edges),
            config.extraction_model,
        )
        return True

    # ── Poll cycle ────────────────────────────────────────────────────────────

    def _log_cycle_complete(self, stats: CycleStats) -> None:
        extraction_failed = stats.extraction_batch - stats.extracted_ok
        _log.info(
            "Cycle complete: provider=%s indexing_batch=%d indexed_ok=%d scoring_batch=%d "
            "extraction_batch=%d extracted_ok=%d extraction_failed=%d",
            self.PROVIDER,
            stats.indexing_batch,
            stats.indexed_ok,
            stats.scoring_batch,
            stats.extraction_batch,
            stats.extracted_ok,
            extraction_failed,
        )

    async def _run_cycle(self) -> CycleStats:
        """
        Run one complete scoring + extraction cycle.
        Returns batch sizes and count of documents that reached extracted state.
        """
        await reset_stale_in_progress(self.PROVIDER, stale_minutes=30)

        # Resume durable jobs first. Documents that enter indexing during this
        # cycle wait until the next cycle, avoiding immediate hot-loop retries.
        indexing_docs = await claim_documents(
            self.PROVIDER,
            state="indexing",
            batch_size=config.extraction_batch_size,
        )
        indexed_ok = 0
        for doc in indexing_docs:
            if await self.resume_index_document(doc):
                indexed_ok += 1

        raw_docs = await claim_documents(
            self.PROVIDER,
            state="raw",
            batch_size=config.scoring_batch_size,
        )
        if raw_docs:
            _log.info(
                "Scoring %d raw documents (provider=%s)", len(raw_docs), self.PROVIDER
            )
        for doc in raw_docs:
            await self.score_document(doc)

        scored_docs = await claim_documents(
            self.PROVIDER,
            state="scored",
            batch_size=config.extraction_batch_size,
        )
        if scored_docs:
            _log.info(
                "Extracting %d scored documents (provider=%s)",
                len(scored_docs),
                self.PROVIDER,
            )
        extracted_ok = 0
        for doc in scored_docs:
            if await self.extract_document(doc):
                extracted_ok += 1

        return CycleStats(
            indexing_batch=len(indexing_docs),
            indexed_ok=indexed_ok,
            scoring_batch=len(raw_docs),
            extraction_batch=len(scored_docs),
            extracted_ok=extracted_ok,
        )

    async def resume_index_document(self, doc: DocumentRecord) -> bool:
        """Resume persisted sink work without re-running scoring or extraction."""
        from processor.dual_write import resume_index_job  # noqa: PLC0415

        try:
            summary = await resume_index_job(doc, _neo4j_client())
            await transition_extracted(
                doc.id,
                doc.current_version,
                model=config.extraction_model,
            )
            _log.info("Indexing resumed and completed: doc=%s result=%s", doc.id, summary)
            return True
        except Exception:
            _log.error(
                "Indexing remains incomplete for doc=%s:\n%s",
                doc.id,
                traceback.format_exc(),
            )
            return False

    async def run(
        self,
        poll_interval: float | None = None,
        run_once: bool = False,
    ) -> None:
        """
        Start the polling loop.

        Handles SIGTERM and SIGINT gracefully: finishes the current cycle then
        shuts down without abandoning in-progress documents (the stale-reset
        on next startup will reclaim them automatically).

        Args:
            poll_interval: Seconds between cycles. Defaults to config value (300s).
            run_once:      If True, run exactly one cycle then exit (useful for tests/cron).
        """
        interval = poll_interval or config.poll_interval_seconds

        _log.info(
            "ProcessorWorker starting: provider=%s scoring_model=%s "
            "entity_model=%s extraction_model=%s poll=%ss",
            self.PROVIDER,
            config.scoring_model,
            config.entity_model,
            config.extraction_model,
            interval,
        )

        stop_event = asyncio.Event()

        def _handle_signal(sig: int) -> None:
            _log.info(
                "ProcessorWorker received signal %s — finishing current cycle then stopping",
                signal.Signals(sig).name,
            )
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle_signal, sig)
            except (NotImplementedError, RuntimeError):
                # Windows / restricted environments don't support add_signal_handler
                pass

        await db_pool.open()

        try:
            while not stop_event.is_set():
                try:
                    stats = await self._run_cycle()
                    self._log_cycle_complete(stats)
                except Exception:
                    _log.error(
                        "Unexpected error in processing cycle (provider=%s):\n%s",
                        self.PROVIDER,
                        traceback.format_exc(),
                    )

                if run_once or stop_event.is_set():
                    break

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass  # Normal: no signal received, start next cycle

        finally:
            await close_shared_client()
            await db_pool.close()
            _log.info("ProcessorWorker stopped: provider=%s", self.PROVIDER)


# ---------------------------------------------------------------------------
# Generic worker (used when running directly: python -m processor.worker)
# ---------------------------------------------------------------------------


class _GenericProcessorWorker(BaseProcessorWorker):
    def __init__(self, provider: str) -> None:
        self.PROVIDER = provider
        super().__init__()

    async def score_content(
        self,
        doc: DocumentRecord,
        content: str,
    ) -> ScoringResult:
        if self.PROVIDER == "portal":
            return ScoringResult(
                signal_score=1.0,
                signal_types=["decision", "constraint", "insight", "signal"],
                reasoning="Portal-uploaded context is deterministically admitted for indexing.",
                suggested_topics=[topic for topic in (doc.repo, doc.doc_type, doc.title) if topic],
            )
        return await super().score_content(doc, content)


# ---------------------------------------------------------------------------
# Provider registry — maps --provider value to a concrete worker class.
# Populated lazily inside _main() to avoid circular imports.
# ---------------------------------------------------------------------------


PROVIDER_WORKERS: dict[str, type[BaseProcessorWorker]] = {}


def _worker_for_provider(provider: str) -> BaseProcessorWorker:
    worker_cls = PROVIDER_WORKERS.get(provider)
    if worker_cls is not None:
        return worker_cls()
    return _GenericProcessorWorker(provider=provider)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=log_level_from_env(),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Prevent SDK debug logs from dumping full LLM request/response payloads.
    for logger_name in (
        "openai",
        "openai._base_client",
        "anthropic",
        "anthropic._base_client",
    ):
        logging.getLogger(logger_name).setLevel(logging.INFO)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Eidetix document processor")
    parser.add_argument(
        "--provider",
        required=True,
        help="source_provider to process (e.g., notion, github, slack)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Poll interval in seconds (default: PROCESSOR_POLL_INTERVAL_SECONDS env var or 300)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle and exit (useful for cron/batch mode)",
    )
    args = parser.parse_args()

    _setup_logging()

    if args.provider == "github-profile":
        from processor.repository_profile_worker import (  # noqa: PLC0415
            RepositoryProfileWorker,
        )

        await RepositoryProfileWorker().run(
            poll_interval=args.poll_interval,
            run_once=args.once,
        )
        return

    from processor.trace.worker import TraceProcessorWorker  # noqa: PLC0415

    PROVIDER_WORKERS["trace"] = TraceProcessorWorker

    worker = _worker_for_provider(args.provider)
    await worker.run(poll_interval=args.poll_interval, run_once=args.once)


if __name__ == "__main__":
    asyncio.run(_main())
