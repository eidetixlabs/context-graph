"""
E2E test: Processor pipeline (raw → scored → extracted → Neo4j).

Prerequisites (docker-compose.dev.yml must be up):
  - Postgres on localhost:5432
  - MinIO on localhost:9000
  - Neo4j on localhost:7687

Run from repo root:
  env -u DATABASE_URL EIDETIX_URL=http://localhost:8080 API_KEY=dev-api-key pytest processor/test/test_processor_pipeline.py -v -s

The test:
  1. Seeds a sample document into S3 + Postgres (state='raw')
  2. Runs scoring pipeline → verifies transition to 'scored'
  3. Runs extraction pipeline → verifies transition to 'extracted'
  4. Asserts that at least one Neo4j node exists for this document
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import AsyncGenerator

import asyncpg
import boto3
import pytest
import pytest_asyncio
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Environment defaults (docker-compose.dev.yml)
# ---------------------------------------------------------------------------

PG_DSN = os.environ.get(
    "DATABASE_URL", "postgresql://eidetix:devpassword@localhost:5432/eidetix"
)
MINIO_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "eidetix")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "devpassword")

# Unique test identifiers (avoid collision with other tests)
TEST_RUN_ID = uuid.uuid4().hex[:8]
ORG_ID = f"org:e2e-processor-test-{TEST_RUN_ID}"
ORG_SLUG = f"e2e-processor-test-{TEST_RUN_ID}"
CONNECTOR_ID = f"conn:manual:{ORG_SLUG}"
DOC_ID = f"doc:manual:test-doc-{TEST_RUN_ID}"
SOURCE_SCOPE = f"scope:manual:{ORG_SLUG}"
SOURCE_REF = f"test-doc-{TEST_RUN_ID}"

# Sample document content designed to score HIGH and produce extractable nodes.
# Contains explicit decision, constraint, and insight patterns.
SAMPLE_DOCUMENT = {
    "title": "Architecture Decision: Use PostgreSQL for Session Storage",
    "content": """
# Architecture Decision Record: Session Storage

## Context
We need to choose a session storage backend for our authentication service.
The team evaluated three options over the past sprint.

## Decision
We decided to use PostgreSQL for session storage instead of Redis.

### Alternatives Considered
1. **Redis** - Rejected because our ops team has limited Redis experience
   and we already run Postgres for user data. Adding Redis would increase
   operational complexity without significant performance benefit for our
   scale (< 10k concurrent users).

2. **In-memory (Node.js)** - Rejected because it doesn't survive restarts
   and can't scale horizontally.

## Constraints
- Sessions MUST survive server restarts (hard requirement from security team)
- Must support horizontal scaling (multiple app replicas)
- Session lookup latency must be < 50ms p99

## Implementation Notes
When implementing session cleanup, be aware that DELETE FROM sessions
WHERE expires_at < now() will lock the entire table if run during peak
hours. Use batched deletes with LIMIT 1000 and schedule during off-peak.

## Team
- Sarah Chen (Lead Engineer) made the final call
- @mike reviewed the performance benchmarks
- DevOps team (Alice, Bob) approved operational aspects

## Status
Accepted - 2024-01-15
""",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def pg() -> AsyncGenerator[asyncpg.Pool, None]:
    pool = await asyncpg.create_pool(
        PG_DSN,
        ssl=False,
        min_size=1,
        max_size=3,
        server_settings={"timezone": "UTC"},
    )
    yield pool
    await pool.close()


@pytest.fixture(scope="module")
def s3():
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if S3_BUCKET not in existing:
        client.create_bucket(Bucket=S3_BUCKET)
    return client


@pytest.fixture(scope="module")
def neo4j_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    yield driver
    driver.close()


@pytest_asyncio.fixture(scope="module", autouse=True)
async def seed(pg: asyncpg.Pool, s3, neo4j_driver) -> AsyncGenerator[None, None]:
    """Seed S3, Postgres with test document; clean up after."""

    # Compute S3 key and content hash
    content_json = json.dumps(SAMPLE_DOCUMENT)
    content_hash = hashlib.sha256(content_json.encode()).hexdigest()
    s3_key = f"{ORG_ID}/manual/raw/{SOURCE_REF}/v1.json"

    # Write sample document to S3
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=content_json.encode(),
        ContentType="application/json",
    )

    # Seed Postgres
    async with pg.acquire() as conn:
        # Organization
        await conn.execute(
            """
            INSERT INTO organizations (id, name, slug, created_at, updated_at)
            VALUES ($1, 'E2E Processor Test Org', $2, now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            ORG_ID,
            ORG_SLUG,
        )

        # Employee
        await conn.execute(
            """
            INSERT INTO employees (id, org_id, email, canonical_name, created_at, updated_at)
            VALUES ('emp:e2e-processor-test', $1, 'processor-test@test.local', 'Processor Test User', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            ORG_ID,
        )

        # Connector instance (manual provider for test)
        await conn.execute(
            """
            INSERT INTO connector_instances
                (id, org_id, provider, config, status, created_at, updated_at)
            VALUES ($1, $2, 'notion', '{}', 'active', now(), now())
            ON CONFLICT (id) DO UPDATE
                SET config        = EXCLUDED.config,
                    status        = CASE
                                        WHEN connector_instances.status IN ('revoked', 'paused')
                                        THEN connector_instances.status
                                        ELSE 'active'
                                    END,
                    error_message = CASE
                                        WHEN connector_instances.status = 'error' THEN NULL
                                        ELSE connector_instances.error_message
                                    END,
                    updated_at    = now()
            """,
            CONNECTOR_ID,
            ORG_ID,
        )

        # Document in 'raw' state
        await conn.execute(
            """
            INSERT INTO documents
                (id, org_id, source_provider, source_scope, source_ref, title,
                 current_version, content_hash, char_count, s3_key, processing_state,
                 created_at, updated_at, ingested_at)
            VALUES ($1, $2, 'manual', $3, $4, $5, 1, $6, $7, $8, 'raw', now(), now(), now())
            ON CONFLICT (id) DO UPDATE SET
                processing_state = 'raw',
                s3_key = EXCLUDED.s3_key,
                content_hash = EXCLUDED.content_hash
            """,
            DOC_ID,
            ORG_ID,
            SOURCE_SCOPE,
            SOURCE_REF,
            SAMPLE_DOCUMENT["title"],
            content_hash,
            len(content_json),
            s3_key,
        )

    yield

    # Cleanup
    async with pg.acquire() as conn:
        await conn.execute(
            "DELETE FROM document_versions WHERE doc_id = $1", DOC_ID
        )
        await conn.execute("DELETE FROM documents WHERE id = $1", DOC_ID)
        await conn.execute(
            "DELETE FROM connector_checkpoints WHERE connector_id = $1", CONNECTOR_ID
        )
        await conn.execute("DELETE FROM connector_instances WHERE id = $1", CONNECTOR_ID)
        await conn.execute(
            "DELETE FROM entity_aliases WHERE org_id = $1", ORG_ID
        )
        await conn.execute(
            "DELETE FROM entity_registry WHERE org_id = $1", ORG_ID
        )
        await conn.execute("DELETE FROM employees WHERE id = 'emp:e2e-processor-test'")
        await conn.execute("DELETE FROM organizations WHERE id = $1", ORG_ID)

    # Clean up S3
    try:
        s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception:
        pass

    # Clean up Neo4j nodes for this project
    # with neo4j_driver.session() as session:
    #     session.run(
    #         "MATCH (n) WHERE n.project = $project DETACH DELETE n",
    #         project=ORG_SLUG,
    #     )


# ---------------------------------------------------------------------------
# Test: Full pipeline raw → scored → extracted → Neo4j
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def setup_env(pg: asyncpg.Pool, s3, neo4j_driver):
    """Set up environment variables and patch modules for processor tests."""
    os.environ.update({
        "DATABASE_URL": PG_DSN,
        "DATABASE_SSL": "0",
        # Both bare and prefixed names — bare is read directly by s3._bucket(),
        # prefixed is what ProcessorConfig picks up via env_prefix="PROCESSOR_".
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
        # Use OpenAI for this test (user provides OPENAI_API_KEY)
        "PROCESSOR_SCORING_MODEL": "openai:gpt-4o-mini",
        "PROCESSOR_EXTRACTION_MODEL": "openai:gpt-4o-mini",
        # Low threshold to ensure document passes scoring
        "PROCESSOR_SIGNAL_SCORE_THRESHOLD": "0.1",
        # Eidetix webserver exposed by docker-compose.dev.yml on port 8080
        "PROCESSOR_EIDETIX_URL": os.environ.get("EIDETIX_URL", "http://localhost:8080"),
        # Local onprem/dev stack protects /api/* with API_KEY (default dev-api-key).
        "PROCESSOR_EIDETIX_API_KEY": os.environ.get("API_KEY", "dev-api-key"),
    })

    # Clear cached config BEFORE any processor imports so the fresh instance
    # picks up the env vars set above.  Also rebind the module-level singletons
    # that were created at import time (before our env vars existed).
    from processor.config import get_config
    get_config.cache_clear()
    import processor.config as config_mod
    import processor.storage.s3 as s3_module
    config_mod.config = get_config()
    s3_module.config = config_mod.config

    # Patch S3 client to use MinIO
    s3_module._s3 = s3

    # Initialize db_pool with our test pool
    from base.db.client import db_pool
    db_pool._pool = pg

    # Recreate the shared MCP server so it connects to the docker-compose
    # webserver on port 8080 (the URL set in the test config above).
    import httpx
    import processor.tools.eidetix as eidetix_module
    eidetix_module.config = config_mod.config
    await eidetix_module.close_shared_client()
    eidetix_module._mcp_http_client = httpx.AsyncClient(
        headers={"Authorization": f"Bearer {config_mod.config.eidetix_api_key}"}
        if config_mod.config.eidetix_api_key
        else {},
        timeout=eidetix_module._DEFAULT_TIMEOUT,
    )
    eidetix_module.mcp_server = eidetix_module._MCPServerHTTPClass(
        f"{config_mod.config.eidetix_url.rstrip('/')}/mcp/sse",
        http_client=eidetix_module._mcp_http_client,
    )

    yield


@pytest.mark.asyncio(loop_scope="module")
async def test_01_raw_to_scored(pg: asyncpg.Pool, setup_env):
    """
    Test Phase 1: raw → scoring → scored.

    Simulates the worker claiming a 'raw' document, running scoring,
    and transitioning to 'scored' state.
    """
    from processor.state_machine import transition_scored
    from processor.agents.scoring import run_scoring
    from processor.storage.s3 import read_document_content

    # Verify document is in 'raw' state
    row = await pg.fetchrow("SELECT processing_state, s3_key FROM documents WHERE id = $1", DOC_ID)
    assert row is not None, f"Test document {DOC_ID} not found in Postgres"
    assert row["processing_state"] == "raw", f"Expected 'raw', got '{row['processing_state']}'"

    print(f"\n✓ Document {DOC_ID} in 'raw' state")

    # Simulate worker: claim document (raw → scoring)
    await pg.execute(
        "UPDATE documents SET processing_state = 'scoring', updated_at = now() WHERE id = $1",
        DOC_ID,
    )

    # Read content from S3
    content = await read_document_content(row["s3_key"])
    assert content, "Failed to read document content from S3"
    print(f"✓ Read document content from S3 ({len(content)} chars)")

    # Run scoring
    scoring_result = await run_scoring(content, title=SAMPLE_DOCUMENT["title"])
    print(f"✓ Scoring complete: score={scoring_result.signal_score:.3f}, types={scoring_result.signal_types}")

    # Assert high score (document is designed to score well)
    assert scoring_result.signal_score >= 0.3, (
        f"Expected high score (>= 0.3), got {scoring_result.signal_score}. "
        f"Reasoning: {scoring_result.reasoning}"
    )

    # Transition: scoring → scored (or skipped if below threshold)
    threshold = 0.1
    new_state = await transition_scored(
        DOC_ID,
        document_version=1,
        score=scoring_result.signal_score,
        threshold=threshold,
    )
    assert new_state == "scored", f"Expected 'scored' state, got '{new_state}'"

    # Verify state in DB
    row = await pg.fetchrow("SELECT processing_state, signal_score FROM documents WHERE id = $1", DOC_ID)
    assert row["processing_state"] == "scored"
    assert row["signal_score"] is not None
    print(f"✓ Document transitioned: raw → scoring → scored (signal_score={row['signal_score']:.3f})")


@pytest.mark.asyncio(loop_scope="module")
async def test_02_scored_to_extracted(pg: asyncpg.Pool, neo4j_driver, setup_env):
    """
    Test Phase 2: scored → extracting → extracted → Neo4j.

    Simulates the worker claiming a 'scored' document, running the full
    extraction pipeline (identity → context → extraction), writing to Neo4j,
    and verifying node existence.
    """
    from processor.state_machine import DocumentRecord, transition_extracted
    from processor.agents.scoring import run_scoring
    from processor.agents.extraction.pipeline import run_extraction_pipeline
    from processor.storage.s3 import read_document_content
    from processor.storage.ingest import write_extraction_result
    from processor.config import config

    # Verify document is in 'scored' state (from previous test)
    row = await pg.fetchrow("SELECT processing_state, s3_key FROM documents WHERE id = $1", DOC_ID)
    assert row is not None, f"Test document {DOC_ID} not found in Postgres"
    assert row["processing_state"] == "scored", f"Expected 'scored', got '{row['processing_state']}'"

    print(f"\n✓ Document {DOC_ID} in 'scored' state")

    # Simulate worker: claim document (scored → extracting)
    await pg.execute(
        "UPDATE documents SET processing_state = 'extracting', updated_at = now() WHERE id = $1",
        DOC_ID,
    )

    # Build DocumentRecord
    doc_row = await pg.fetchrow(
        """
        SELECT id, org_id, source_provider, source_scope, source_ref, title,
               s3_key, processing_state, signal_score, char_count, current_version
        FROM documents WHERE id = $1
        """,
        DOC_ID,
    )

    doc = DocumentRecord(
        id=doc_row["id"],
        org_id=doc_row["org_id"],
        source_provider=doc_row["source_provider"],
        source_scope=doc_row["source_scope"],
        source_ref=doc_row["source_ref"],
        title=doc_row["title"],
        s3_key=doc_row["s3_key"],
        processing_state=doc_row["processing_state"],
        signal_score=doc_row["signal_score"],
        char_count=doc_row["char_count"],
        current_version=doc_row["current_version"],
    )

    # Read content and re-run scoring for topic hints
    content = await read_document_content(doc.s3_key)
    scoring_result = await run_scoring(content, title=SAMPLE_DOCUMENT["title"])
    print(f"✓ Scoring for topic hints: {scoring_result.suggested_topics}")

    # Run extraction pipeline (Phase 1 → Phase 2 → Phase 3)
    extraction_result = await run_extraction_pipeline(
        doc=doc,
        content=content,
        scoring_result=scoring_result,
    )

    print(f"✓ Extraction complete: {len(extraction_result.nodes)} nodes, {len(extraction_result.edges)} edges")

    # Assert we got some nodes
    assert len(extraction_result.nodes) > 0, "Expected at least one extracted node"
    assert len(extraction_result.edges) > 0, "Expected at least one extracted edge"

    # Verify entity/alias persistence in Postgres
    entity_count = await pg.fetchval(
        "SELECT count(*) FROM entity_registry WHERE org_id = $1",
        doc.org_id,
    )
    alias_count = await pg.fetchval(
        "SELECT count(*) FROM entity_aliases WHERE org_id = $1 AND source = 'extraction'",
        doc.org_id,
    )
    print(f"✓ Postgres entity_registry rows for org: {entity_count}")
    print(f"✓ Postgres entity_aliases rows (source='extraction') for org: {alias_count}")
    assert entity_count > 0, (
        f"Expected at least one entity_registry row for org_id={doc.org_id} after extraction"
    )
    assert alias_count > 0, (
        f"Expected at least one entity_aliases row with source='extraction' for org_id={doc.org_id}"
    )

    # Write to Neo4j
    summary = await write_extraction_result(doc, extraction_result)
    print(f"✓ Neo4j write: {summary.get('nodes_upserted', 0)} nodes, {summary.get('rels_upserted', 0)} rels")

    # Transition: extracting → extracted
    await transition_extracted(
        DOC_ID,
        document_version=1,
        model=config.extraction_model,
    )

    # Verify state in DB
    row = await pg.fetchrow("SELECT processing_state, extraction_model FROM documents WHERE id = $1", DOC_ID)
    assert row["processing_state"] == "extracted"
    print(f"✓ Document transitioned: scored → extracting → extracted (model={row['extraction_model']})")

    # Verify Neo4j node existence
    with neo4j_driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE n.project = $project
            RETURN count(n) as node_count, collect(distinct labels(n)[0]) as labels
            """,
            project=ORG_SLUG,
        )
        record = result.single()
        node_count = record["node_count"]
        labels = record["labels"]

    assert node_count > 0, f"Expected at least one Neo4j node for project '{ORG_SLUG}'"
    print(f"✓ Neo4j verified: {node_count} nodes with labels {labels}")

    score_str = f"{doc.signal_score:.3f}" if doc.signal_score else "N/A"
    print(f"\n{'='*60}")
    print(f"✓ FULL PIPELINE TEST PASSED")
    print(f"  Document: {DOC_ID}")
    print(f"  Signal Score: {score_str}")
    print(f"  Extracted Nodes: {len(extraction_result.nodes)}")
    print(f"  Neo4j Nodes: {node_count}")
    print(f"  Node Types: {labels}")
    print(f"{'='*60}")
