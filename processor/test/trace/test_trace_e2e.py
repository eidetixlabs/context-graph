"""
End-to-end trace pipeline test.

Infrastructure:
    docker-compose "full" profile — neo4j, postgres, minio, eidetix webserver.
    (Started by conftest.py; containers are NOT removed after the run.)

Processor:
    TraceProcessorWorker runs locally inside the pytest process, pointed at the
    docker-compose services via environment variables.

Test sequence
─────────────
Step 1-2  Postgres seeding and user creation — handled by conftest fixtures
          (seeded_user, bearer_token).

Step 3    POST /ingest/traces  (3 traces with mentions, origin=cursor)
          → webserver injects author mention + _author_entity_id from session
            (see main.py lines 468-498).

Step 4    Verify MinIO users/{employee_id}/raw/ contains the 3 trace objects with:
          - author person-mention present
          - author name injected into keywords
          - _author_entity_id set

Step 5    Run TraceProcessorWorker (LLM = claude or cursor via env)
          → pipeline: entity-resolution → context-prefetch → transform → Neo4j.

Step 6a   MinIO users/{employee_id}/raw/  → empty  (all moved to completed/)
Step 6b   MinIO users/{employee_id}/completed/ → ≥3 objects
Step 6c   Neo4j has the user Entity node (entity:person:trace-test)

Step 7    Second worker run → scanned=0, processed_ok=0 (idempotent).

Run:
    env -u DATABASE_URL EIDETIX_URL=http://localhost:8080 API_KEY=dev-api-key \\
        OPENAI_API_KEY=sk-... pytest processor/test/trace/test_trace_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

_TRACE_DIR = Path(__file__).resolve().parent
if str(_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACE_DIR))

from trace_constants import (  # noqa: E402
    API_KEY,
    EIDETIX_URL,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PG_DSN,
    S3_BUCKET,
    TEST_ORG_ID,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for trace e2e test",
)


# ---------------------------------------------------------------------------
# Processor env setup fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def setup_trace_processor_env(pg_pool, s3_client):
    """Configure processor and base S3/DB modules to use the docker stack.

    Sets env vars, resets the lazy S3 singletons so they reconnect to MinIO,
    and opens the global db_pool so entity-resolution can acquire connections.
    """
    os.environ.update(
        {
            "DATABASE_URL": PG_DSN,
            "DATABASE_SSL": "0",
            "AWS_S3_BUCKET": S3_BUCKET,
            "PROCESSOR_AWS_S3_BUCKET": S3_BUCKET,
            "AWS_REGION": "us-east-1",
            "PROCESSOR_AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": MINIO_ACCESS_KEY,
            "AWS_SECRET_ACCESS_KEY": MINIO_SECRET_KEY,
            "S3_ENDPOINT_URL": MINIO_ENDPOINT,
            "NEO4J_URI": NEO4J_URI,
            "NEO4J_USER": NEO4J_USER,
            "NEO4J_PASSWORD": NEO4J_PASSWORD,
            "PROCESSOR_SCORING_MODEL": "openai:gpt-4.1-mini-2025-04-14",
            "PROCESSOR_ENTITY_MODEL": "openai:gpt-4.1-mini-2025-04-14",
            "PROCESSOR_EXTRACTION_MODEL": "openai:gpt-4.1-mini-2025-04-14",
            "PROCESSOR_SIGNAL_SCORE_THRESHOLD": "0.1",
            "PROCESSOR_EIDETIX_URL": EIDETIX_URL,
            "PROCESSOR_EIDETIX_API_KEY": API_KEY,
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "PROCESSOR_TRACER_LINKER_MODEL": "openai:gpt-4.1-mini-2025-04-14"
        }
    )

    # ── Refresh processor config ────────────────────────────────────────────
    from processor.config import get_config

    get_config.cache_clear()
    fresh_config = get_config()

    # ── Point S3 singletons at the test boto3 client (already points to MinIO)
    import base.tools.s3 as base_s3
    import processor.storage.s3 as proc_s3

    base_s3._s3 = s3_client
    proc_s3._s3 = s3_client
    proc_s3.config = fresh_config

    # ── Clear Neo4j client cache so it reconnects with updated env vars ──────
    from processor.storage.ingest import _neo4j_client

    _neo4j_client.cache_clear()

    # ── Open the global db_pool for entity resolution ────────────────────────
    from base.db.client import db_pool

    if db_pool._pool is None:
        await db_pool.open()

    # ── Reset trace linker agent so it rebuilds with fresh config ────────────
    import processor.trace.linker as linker_mod

    linker_mod._linker_agent = None

    yield fresh_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_trace(idx: int, session_id: str) -> dict:
    """Build a single trace payload dict (node-embedded format, Decision)."""
    return {
        "trace_id": f"trace-e2e-{idx}",
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "node": {
            "type": "Decision",
            "label": f"E2E test decision #{idx}: chose approach A over B",
            "rationale": (
                f"After evaluating options the team decided approach A is better "
                f"for e2e test trace #{idx}, considering performance and maintainability trade-offs."
            ),
            "trigger": f"E2E test #{idx} triggered this decision.",
            "alternatives": [
                {"option": "approach B", "rejected_because": "Higher maintenance cost."}
            ],
            "confidence": "high",
            "origin": "cursor",
            "tag": "checkpoint:stop",
        },
        "keywords": ["e2e-test", "approach-a", f"trace-{idx}"],
        "mentions": [
            {"mention_text": "approach A", "mention_type": "object"},
            {"mention_text": "e2e-service", "mention_type": "service"},
        ],
    }


def _list_s3_keys(s3_client, prefix: str) -> list[str]:
    """Return all S3 keys under prefix (handles empty results)."""
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


# ---------------------------------------------------------------------------
# Test 1 — ingest traces via webserver (step 3 & 4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_ingest_traces_and_verify_minio(
    bearer_token: str,
    s3_client,
    seeded_user: dict,
):
    """Step 3: POST /ingest/traces; Step 4: verify raw objects in MinIO."""
    session_id = f"session-e2e-{uuid.uuid4().hex[:12]}"
    traces = [_build_trace(i, session_id) for i in range(3)]

    # ── Step 3: ingest ──────────────────────────────────────────────────────
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{EIDETIX_URL}/ingest/traces",
            json={"traces": traces},
            headers={"Authorization": f"Bearer {bearer_token}"},
        )

    assert resp.status_code == 200, (
        f"Expected 200 from /ingest/traces, got {resp.status_code}: {resp.text}"
    )
    body = resp.json()
    assert body.get("ok") is True, f"Expected ok=true, got: {body}"
    assert body.get("accepted") == 3, f"Expected accepted=3, got: {body}"

    # ── Step 4: verify raw objects in MinIO ────────────────────────────────
    project = TEST_ORG_ID.removeprefix("org:")
    employee_id = seeded_user["employee_id"]
    raw_prefix = f"traces/{TEST_ORG_ID}/{project}/users/{employee_id}/raw/"
    raw_keys = _list_s3_keys(s3_client, raw_prefix)

    assert len(raw_keys) >= 3, (
        f"Expected ≥3 raw trace objects in MinIO under {raw_prefix!r}, "
        f"found {len(raw_keys)}: {raw_keys}"
    )

    # Verify author enrichment applied by webserver (main.py lines 468-498)
    author_name = seeded_user["canonical_name"]
    for key in raw_keys[:3]:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        trace_data = json.loads(obj["Body"].read())

        mentions = trace_data.get("mentions", [])
        person_mentions = [m for m in mentions if m.get("mention_type") == "person"]
        assert len(person_mentions) >= 1, (
            f"Expected at least one person-type mention (author) in {key}, "
            f"got mentions={mentions}"
        )
        assert any(m.get("mention_text") == author_name for m in person_mentions), (
            f"Author {author_name!r} not found in person mentions of {key}: {person_mentions}"
        )

        keywords = trace_data.get("keywords", [])
        assert author_name in keywords, (
            f"Author {author_name!r} not injected into keywords of {key}: {keywords}"
        )


# ---------------------------------------------------------------------------
# Test 2 — processor run (step 5 & 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_processor_processes_traces(
    setup_trace_processor_env,
    s3_client,
    neo4j_driver,
    seeded_user: dict,
):
    """Step 5: run TraceProcessorWorker; Step 6: validate S3 + Neo4j state."""
    from processor.trace.worker import TraceProcessorWorker

    # ── Step 5: run the processor ───────────────────────────────────────────
    worker = TraceProcessorWorker()
    await worker.run(poll_interval=0.1, run_once=True)

    project = TEST_ORG_ID.removeprefix("org:")
    employee_id = seeded_user["employee_id"]

    # ── Step 6a: no raw traces remain ───────────────────────────────────────
    raw_prefix = f"traces/{TEST_ORG_ID}/{project}/users/{employee_id}/raw/"
    remaining_raw = _list_s3_keys(s3_client, raw_prefix)
    assert remaining_raw == [], (
        f"Expected no raw traces after processing, still present: {remaining_raw}"
    )

    # ── Step 6b: all traces moved to completed/ ──────────────────────────────
    completed_prefix = (
        f"traces/{TEST_ORG_ID}/{project}/users/{employee_id}/completed/"
    )
    completed_keys = _list_s3_keys(s3_client, completed_prefix)
    assert len(completed_keys) >= 3, (
        f"Expected ≥3 completed traces, found {len(completed_keys)}: {completed_keys}"
    )

    # ── Step 6c: Neo4j has the user Entity node ──────────────────────────────
    with neo4j_driver.session() as session:
        rows = session.run(
            "MATCH (e:Entity {id: $id}) RETURN e.id AS id, e.label AS label LIMIT 1",
            id=seeded_user["entity_id"],
        ).data()

    assert len(rows) >= 1, (
        f"Expected user Entity node in Neo4j for id={seeded_user['entity_id']!r}, "
        f"but no matching node found. "
        f"Confirm entity resolution Phase 1 wrote the stub entity during pipeline run."
    )


# ---------------------------------------------------------------------------
# Test 3 — second processor run picks nothing (step 7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="module")
async def test_processor_second_run_is_noop(setup_trace_processor_env):
    """Step 7: running the processor again finds zero raw traces (idempotent)."""
    from processor.trace.worker import TraceProcessorWorker

    worker = TraceProcessorWorker()
    stats = await worker._run_cycle()

    assert stats.scanned == 0, (
        f"Expected scanned=0 on second run (no raw traces left), got {stats.scanned}"
    )
    assert stats.processed_ok == 0, (
        f"Expected processed_ok=0 on second run, got {stats.processed_ok}"
    )
