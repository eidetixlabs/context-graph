"""
processor/trace/worker.py — Trace-specific processor worker.

Polls S3 for raw trace files, runs the three-phase trace pipeline on each,
then moves the file to completed/ (success) or failed/ (error).

S3 lifecycle per trace file:
    traces/{org}/{project}/raw/{date}/{trace_id}.json
        ↓ fetch (file stays in raw/ during processing)
        ↓ run_trace_pipeline()
        ↓ success → copy to traces/{org}/{project}/completed/{date}/{trace_id}.json,
                    delete from raw/
        ↓ failure → copy to traces/{org}/{project}/failed/{date}/{trace_id}.json,
                    delete from raw/

Traces are grouped by session_id before pipeline calls so related traces in
the same S3 batch share context.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from collections import defaultdict
from typing import NamedTuple

from base.neo4j_client import Neo4jClient
from base.tools.s3 import (
    _bucket,
    _s3_client,
    list_keys_async,
    read_json_async,
    write_json,
)
from processor.config import config
from processor.storage.ingest import _neo4j_client
from processor.trace.pipeline import run_trace_pipeline
from processor.worker import BaseProcessorWorker

_log = logging.getLogger(__name__)

_LLM_PROVIDER_HOSTS: dict[str, tuple[str, int]] = {
    "openai": ("api.openai.com", 443),
    "anthropic": ("api.anthropic.com", 443),
    "google-gla": ("generativelanguage.googleapis.com", 443),
    "google-vertex": ("us-central1-aiplatform.googleapis.com", 443),
}
_LLM_PROBE_CONNECT_TIMEOUT = 5.0
_LLM_PROBE_INTERVAL = 2.0


def _probe_host_for_model(model_str: str) -> tuple[str, int] | None:
    """Return (host, port) to TCP-probe for the given provider:model string."""
    provider = model_str.split(":", 1)[0].lower()
    return _LLM_PROVIDER_HOSTS.get(provider)


def _cycle_probe_targets() -> list[tuple[str, int]]:
    """Deduplicated set of (host, port) pairs that must be reachable this cycle."""
    seen: set[tuple[str, int]] = set()
    targets: list[tuple[str, int]] = []
    for model_str in (config.entity_model, config.trace_linker_model):
        target = _probe_host_for_model(model_str)
        if target and target not in seen:
            seen.add(target)
            targets.append(target)
    return targets


async def _tcp_probe(host: str, port: int) -> bool:
    """Return True if a TCP connection to host:port succeeds within the timeout."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=_LLM_PROBE_CONNECT_TIMEOUT,
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return True
    except Exception:
        return False


async def _wait_for_llm_reachable(probe_interval: float = _LLM_PROBE_INTERVAL) -> None:
    """
    Block until all configured LLM provider endpoints are TCP-reachable.

    Logs a WARNING for every failed probe attempt at the configured interval
    so that ops can see the pod is waiting rather than silently stalling.
    """
    targets = _cycle_probe_targets()
    if not targets:
        return

    attempt = 0
    while True:
        attempt += 1
        failures = []
        for host, port in targets:
            if not await _tcp_probe(host, port):
                failures.append(f"{host}:{port}")

        if not failures:
            if attempt > 1:
                _log.info(
                    "LLM probe: all endpoints reachable after %d attempt(s) — proceeding",
                    attempt,
                )
            return

        _log.warning(
            "LLM probe attempt %d: unreachable=%s — retrying in %.0fs",
            attempt,
            failures,
            probe_interval,
        )
        await asyncio.sleep(probe_interval)


class TraceCycleStats(NamedTuple):
    scanned: int
    processed_ok: int
    failed: int


def _raw_key_to_completed(key: str) -> str:
    """Replace the /raw/ segment with /completed/ in a trace S3 key."""
    parts = key.split("/raw/", 1)
    if len(parts) == 2:
        return parts[0] + "/completed/" + parts[1]
    return key.replace("/raw/", "/completed/", 1)


def _raw_key_to_failed(key: str) -> str:
    """Replace the /raw/ segment with /failed/ in a trace S3 key."""
    parts = key.split("/raw/", 1)
    if len(parts) == 2:
        return parts[0] + "/failed/" + parts[1]
    return key.replace("/raw/", "/failed/", 1)


def _move_s3_object(src_key: str, dst_key: str) -> None:
    """Copy src → dst then delete src (atomic from the caller's perspective)."""
    bucket = _bucket()
    s3 = _s3_client()
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=bucket, Key=src_key)


async def _move_s3_object_async(src_key: str, dst_key: str) -> None:
    await asyncio.to_thread(_move_s3_object, src_key, dst_key)


class TraceProcessorWorker(BaseProcessorWorker):
    """
    Worker that processes raw traces from S3 using the trace pipeline.

    Overrides _run_cycle() entirely — uses S3 prefix scan instead of
    Postgres claim_documents. score_document() and extract_document()
    are not used.
    """

    PROVIDER = "trace"

    def __init__(self) -> None:
        # Skip BaseProcessorWorker.__init__ scoring-cache setup; we don't score traces.
        if not self.PROVIDER:
            raise ValueError(f"{self.__class__.__name__} must set PROVIDER class attribute.")
        self._scoring_cache = {}

    def _get_neo4j_client(self) -> Neo4jClient:
        return _neo4j_client()

    def _log_cycle_complete(self, stats: TraceCycleStats) -> None:
        _log.info(
            "Cycle complete: provider=%s scanned=%d processed_ok=%d failed=%d",
            self.PROVIDER,
            stats.scanned,
            stats.processed_ok,
            stats.failed,
        )

    async def _run_cycle(self) -> TraceCycleStats:  # type: ignore[override]
        """
        Scan S3 for raw trace files, group by session_id, run the pipeline,
        and move each file to completed/ or failed/.
        """
        all_keys = await list_keys_async("traces/")
        raw_keys = [k for k in all_keys if "/raw/" in k]

        if not raw_keys:
            return TraceCycleStats(scanned=0, processed_ok=0, failed=0)

        _log.info("Trace worker cycle: found %d raw trace files", len(raw_keys))

        # Load all trace dicts first (read-only; files stay in raw/ during processing)
        key_to_trace: dict[str, dict] = {}
        for key in raw_keys:
            try:
                trace = await read_json_async(key)
                if isinstance(trace, dict):
                    key_to_trace[key] = trace
            except Exception as exc:
                _log.error("Failed to read trace file %s: %s", key, exc)

        # Group by session_id so related traces share context within a batch
        session_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for key, trace in key_to_trace.items():
            session_id = trace.get("session_id") or "__no_session__"
            session_groups[session_id].append((key, trace))

        await _wait_for_llm_reachable()

        neo4j = self._get_neo4j_client()
        processed_ok = 0
        failed = 0

        # TODO[COST] : Batching of traces is not implemented. One LLM call per trace at this point.
        for session_id, items in session_groups.items():
            for s3_key, trace in items:
                trace_id = trace.get("trace_id") or s3_key
                try:
                    await run_trace_pipeline(trace=trace, neo4j_client=neo4j)
                    dst_key = _raw_key_to_completed(s3_key)
                    await _move_s3_object_async(s3_key, dst_key)
                    _log.info(
                        "Trace processed: trace_id=%s → %s", trace_id, dst_key
                    )
                    processed_ok += 1
                except Exception:
                    _log.error(
                        "Trace pipeline failed: trace_id=%s\n%s",
                        trace_id,
                        traceback.format_exc(),
                    )
                    try:
                        dst_key = _raw_key_to_failed(s3_key)
                        await _move_s3_object_async(s3_key, dst_key)
                    except Exception as move_exc:
                        _log.error(
                            "Failed to move trace to failed/: trace_id=%s error=%s",
                            trace_id,
                            move_exc,
                        )
                    failed += 1

        return TraceCycleStats(
            scanned=len(raw_keys),
            processed_ok=processed_ok,
            failed=failed,
        )
