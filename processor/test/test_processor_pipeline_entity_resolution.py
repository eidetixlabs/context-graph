"""
Canonical entity-resolution e2e tests (3 scenarios in one file).

Run:
  env -u DATABASE_URL EIDETIX_URL=http://localhost:8080 API_KEY=dev-api-key \
  pytest processor/test/test_processor_pipeline_entity_resolution.py -v -s
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from processor.state_machine import DocumentRecord
from processor.storage.s3 import read_document_content
from processor.test.test_processor_pipeline import S3_BUCKET, neo4j_driver, pg, s3, setup_env

RUN_ID = uuid.uuid4().hex[:8]
ORG_ID = f"org:e2e-canonical-resolution-{RUN_ID}"
ORG_SLUG = f"e2e-canonical-resolution-{RUN_ID}"
CONNECTOR_ID = f"conn:notion:{ORG_SLUG}"
SOURCE_SCOPE = f"scope:notion:{ORG_SLUG}"

RAJESH_ENTITY_ID = f"entity:person:{ORG_SLUG}:rajesh-kumar"
SARAH_ENTITY_ID = f"entity:person:{ORG_SLUG}:sarah-chen"
ACME_ENTITY_ID = f"entity:organization:{ORG_SLUG}:acme"

DOC_MERGE_ID = f"doc:notion:canonical-merge-{RUN_ID}"
DOC_DEFER_ID = f"doc:notion:canonical-deferred-{RUN_ID}"
DOC_PRONOUN_ID = f"doc:notion:canonical-pronoun-{RUN_ID}"


@dataclass(frozen=True)
class SeededDoc:
    doc_id: str
    title: str
    source_ref: str
    content: dict[str, str]

    @property
    def s3_key(self) -> str:
        return f"{ORG_ID}/notion/raw/{self.source_ref}/v1.json"


DOCS: list[SeededDoc] = [
    SeededDoc(
        doc_id=DOC_MERGE_ID,
        title="Meeting: Acme Integration Sync",
        source_ref=f"canonical-merge-{RUN_ID}",
        content={
            "title": "Meeting: Acme Integration Sync",
            "content": (
                "Date: 2026-05-20\n"
                "Attendees: Raj, Jennifer\n"
                "\n"
                "## Decision\n"
                "We decided to gate the Q2 Acme pilot on Raj's integration demo next week; "
                "raj@acme.com owns the timeline. "
                "Rejected the standalone sandbox because it lacks VP Eng sign-off on API documentation.\n"
                "\n"
                "## Constraint\n"
                "Acme API documentation MUST arrive before Friday or the integration release blocks.\n"
                "\n"
                "Transcript:\n"
                "Raj mentioned he'd demo next week. raj@acme.com confirmed the timeline.\n"
                "Their VP Eng agreed to send the API docs before Friday.\n"
            ),
        },
    ),
    SeededDoc(
        doc_id=DOC_DEFER_ID,
        title="Program review with broad attendee list",
        source_ref=f"canonical-deferred-{RUN_ID}",
        content={
            "title": "Program review with broad attendee list",
            "content": (
                "Attendees: Alice, Bob, Carol, Dave, Erin, Frank, Grace, Heidi, Ivan, Judy.\n"
                "Observers: Mallory, Niaj, Olivia, Peggy, Trent, Victor.\n"
                "Discussion touched many teams and tools, but one concrete outcome:\n"
                "Decision: proceed with Zephyr API rollout and assign Alice to own migration."
            ),
        },
    ),
    SeededDoc(
        doc_id=DOC_PRONOUN_ID,
        title="Meeting: Acme API planning",
        source_ref=f"canonical-pronoun-{RUN_ID}",
        content={
            "title": "Meeting: Acme API planning",
            "content": (
                "Attendees: Sarah Chen, Bob\n"
                "\n"
                "## Decision\n"
                "We decided to freeze the Acme API schema on Friday with Sarah Chen owning the final review, "
                "rejecting an open-ended draft cycle because Acme leadership already aligned on timeline.\n"
                "\n"
                "## Constraint\n"
                "Schema freeze MUST hold on Friday; Lead Engineer migration notes are required before rollout.\n"
                "\n"
                "Transcript:\n"
                "Sarah said the schema freeze is Friday. She promised final review by Thursday.\n"
                "Acme leadership aligned on timeline. Their Lead Engineer agreed to publish migration notes.\n"
            ),
        },
    ),
]


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _entries_for_entity(entries: dict[str, dict[str, Any]], entity_id: str) -> list[dict[str, Any]]:
    return [v for v in entries.values() if v.get("entity_id") == entity_id]


def _has_mention(entries: dict[str, dict[str, Any]], token: str) -> bool:
    needle = token.lower()
    for key, value in entries.items():
        mention = str(value.get("mention_text", ""))
        if needle in key.lower() or needle in mention.lower():
            return True
    return False


async def _load_doc(pool: asyncpg.Pool, doc_id: str) -> DocumentRecord:
    row = await pool.fetchrow(
        """
        SELECT id, org_id, source_provider, source_scope, source_ref, title,
               s3_key, processing_state, signal_score, char_count, current_version
        FROM documents
        WHERE id = $1
        """,
        doc_id,
    )
    assert row is not None, f"Missing doc {doc_id}"
    return DocumentRecord(
        id=row["id"],
        org_id=row["org_id"],
        source_provider=row["source_provider"],
        source_scope=row["source_scope"],
        source_ref=row["source_ref"],
        title=row["title"],
        s3_key=row["s3_key"],
        processing_state=row["processing_state"],
        signal_score=row["signal_score"],
        char_count=row["char_count"],
        current_version=row["current_version"],
    )


async def _run_processor_for_doc(
    pool: asyncpg.Pool,
    doc_id: str,
    *,
    neo4j_client=None,
) -> None:
    from processor.agents.extraction.pipeline import run_extraction_pipeline
    from processor.agents.scoring import run_scoring
    from processor.state_machine import transition_scored
    from processor.storage.ingest import write_extraction_result

    row = await pool.fetchrow(
        "SELECT processing_state, s3_key, title FROM documents WHERE id = $1",
        doc_id,
    )
    assert row is not None
    assert row["processing_state"] == "raw"

    await pool.execute(
        "UPDATE documents SET processing_state = 'scoring', updated_at = now() WHERE id = $1",
        doc_id,
    )

    content = await read_document_content(row["s3_key"])
    assert content, f"S3 content missing for {doc_id}"

    scoring_result = await run_scoring(content, title=row["title"])
    new_state = await transition_scored(
        doc_id,
        document_version=1,
        score=scoring_result.signal_score,
        threshold=0.1,
    )
    assert new_state == "scored"

    doc = await _load_doc(pool, doc_id)
    extraction_result = await run_extraction_pipeline(
        doc=doc,
        content=content,
        scoring_result=scoring_result,
        neo4j_client=neo4j_client,
    )
    assert len(extraction_result.nodes) > 0
    assert len(extraction_result.edges) > 0

    if neo4j_client is None:
        await write_extraction_result(doc, extraction_result)


async def _seed_entity(
    conn: asyncpg.Connection,
    *,
    entity_id: str,
    entity_type: str,
    canonical_name: str,
    aliases: list[str],
    metadata: dict[str, str],
) -> None:
    await conn.execute(
        """
        INSERT INTO entity_registry (id, org_id, entity_type, canonical_name, status)
        VALUES ($1, $2, $3, $4, 'active')
        ON CONFLICT (id) DO NOTHING
        """,
        entity_id,
        ORG_ID,
        entity_type,
        canonical_name,
    )

    for alias in aliases:
        normalized = "".join(ch for ch in alias.lower() if ch.isalnum() or ch == " ").strip()
        normalized = " ".join(normalized.split())
        if not normalized:
            continue
        await conn.execute(
            """
            INSERT INTO entity_aliases (alias_normalized, entity_id, org_id, alias_raw, source)
            VALUES ($1, $2, $3, $4, 'manual')
            ON CONFLICT (org_id, alias_normalized) DO NOTHING
            """,
            normalized,
            entity_id,
            ORG_ID,
            alias,
        )

    for key, value in metadata.items():
        await conn.execute(
            """
            INSERT INTO entity_metadata (entity_id, key, value, source)
            VALUES ($1, $2, $3, 'manual')
            ON CONFLICT (entity_id, key) DO UPDATE
                SET value = EXCLUDED.value,
                    updated_at = now()
            """,
            entity_id,
            key,
            value,
        )


@pytest_asyncio.fixture(scope="module")
async def setup_canonical_env(pg: asyncpg.Pool, s3, setup_env):

    from processor.config import get_config
    import processor.config as config_mod
    import processor.storage.s3 as s3_module
    from base.db.client import db_pool

    get_config.cache_clear()
    config_mod.config = get_config()
    s3_module.config = config_mod.config
    s3_module._s3 = s3
    db_pool._pool = pg
    yield
    get_config.cache_clear()
    config_mod.config = get_config()
    s3_module.config = config_mod.config


@pytest_asyncio.fixture(scope="module")
async def seeded_pipeline_data(pg: asyncpg.Pool, s3, setup_canonical_env):
    async with pg.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO organizations (id, name, slug, created_at, updated_at)
            VALUES ($1, 'E2E Canonical Resolution Org', $2, now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            ORG_ID,
            ORG_SLUG,
        )
        await conn.execute(
            """
            INSERT INTO connector_instances
                (id, org_id, provider, config, status, created_at, updated_at)
            VALUES ($1, $2, 'notion', '{}', 'active', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            CONNECTOR_ID,
            ORG_ID,
        )

        await _seed_entity(
            conn,
            entity_id=ACME_ENTITY_ID,
            entity_type="organization",
            canonical_name="Acme",
            aliases=["Acme", "Acme Inc", "Acme Integration"],
            metadata={},
        )
        await _seed_entity(
            conn,
            entity_id=RAJESH_ENTITY_ID,
            entity_type="person",
            canonical_name="Rajesh Kumar",
            aliases=["Raj", "Rajesh", "raj@acme.com"],
            metadata={
                "email": "raj@acme.com",
                "role": "VP Eng",
                "org": ACME_ENTITY_ID,
            },
        )
        await _seed_entity(
            conn,
            entity_id=SARAH_ENTITY_ID,
            entity_type="person",
            canonical_name="Sarah Chen",
            aliases=["Sarah", "Sarah Chen"],
            metadata={
                "role": "Lead Engineer",
                "org": ACME_ENTITY_ID,
            },
        )

        for seeded_doc in DOCS:
            body = json.dumps(seeded_doc.content)
            content_hash = hashlib.sha256(body.encode()).hexdigest()
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=seeded_doc.s3_key,
                Body=body.encode(),
                ContentType="application/json",
            )
            await conn.execute(
                """
                INSERT INTO documents
                    (id, org_id, source_provider, source_scope, source_ref, title,
                     current_version, content_hash, char_count, s3_key, processing_state,
                     created_at, updated_at, ingested_at)
                VALUES ($1, $2, 'notion', $3, $4, $5, 1, $6, $7, $8, 'raw', now(), now(), now())
                ON CONFLICT (id) DO UPDATE SET
                    processing_state = 'raw',
                    s3_key = EXCLUDED.s3_key,
                    content_hash = EXCLUDED.content_hash,
                    title = EXCLUDED.title
                """,
                seeded_doc.doc_id,
                ORG_ID,
                SOURCE_SCOPE,
                seeded_doc.source_ref,
                seeded_doc.title,
                content_hash,
                len(body),
                seeded_doc.s3_key,
            )

    yield

    async with pg.acquire() as conn:
        await conn.execute(
            "DELETE FROM document_resolution_maps WHERE doc_id = ANY($1::text[])",
            [d.doc_id for d in DOCS],
        )
        await conn.execute(
            "DELETE FROM document_chunks WHERE doc_id = ANY($1::text[])",
            [d.doc_id for d in DOCS],
        )
        await conn.execute(
            "DELETE FROM document_versions WHERE doc_id = ANY($1::text[])",
            [d.doc_id for d in DOCS],
        )
        await conn.execute("DELETE FROM documents WHERE id = ANY($1::text[])", [d.doc_id for d in DOCS])
        await conn.execute("DELETE FROM entity_metadata WHERE entity_id IN ($1, $2, $3)", RAJESH_ENTITY_ID, SARAH_ENTITY_ID, ACME_ENTITY_ID)
        await conn.execute("DELETE FROM entity_aliases WHERE org_id = $1", ORG_ID)
        await conn.execute("DELETE FROM entity_registry WHERE org_id = $1", ORG_ID)
        await conn.execute("DELETE FROM connector_instances WHERE id = $1", CONNECTOR_ID)
        await conn.execute("DELETE FROM organizations WHERE id = $1", ORG_ID)

    for seeded_doc in DOCS:
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=seeded_doc.s3_key)
        except Exception:
            pass


@pytest.mark.asyncio(loop_scope="module")
async def test_01_existing_entities_merge_alias_email_role_pronoun(
    pg: asyncpg.Pool,
    seeded_pipeline_data,
    neo4j_driver,
):
    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    from base.neo4j_client import Neo4jClient

    handler = _ListHandler()
    dual_write_logger = logging.getLogger("processor.dual_write")
    dual_write_logger.addHandler(handler)

    neo_client = Neo4jClient()
    neo_client.verify()

    before_rajesh = await pg.fetchval(
        "SELECT count(*) FROM entity_registry WHERE id = $1",
        RAJESH_ENTITY_ID,
    )
    assert before_rajesh == 1

    try:
        await _run_processor_for_doc(pg, DOC_MERGE_ID, neo4j_client=neo_client)
    finally:
        neo_client.close()
        dual_write_logger.removeHandler(handler)

    after_rajesh = await pg.fetchval(
        "SELECT count(*) FROM entity_registry WHERE id = $1",
        RAJESH_ENTITY_ID,
    )
    assert after_rajesh == 1

    row = await pg.fetchrow(
        """
        SELECT entries, audit
        FROM document_resolution_maps
        WHERE doc_id = $1 AND doc_version = 1
        """,
        DOC_MERGE_ID,
    )
    assert row is not None
    entries = _coerce_json(row["entries"])
    raj_mentions = [
        e for e in entries.values()
        if str(e.get("mention_text", "")).lower().strip(".,!?;:\"'") == "raj"
    ]
    email_mentions = [
        e for e in entries.values()
        if str(e.get("mention_text", "")).lower().strip(".,!?;:\"'") == "raj@acme.com"
    ]
    role_mentions = [
        e for e in entries.values()
        if "vp eng" in str(e.get("mention_text", "")).lower()
    ]
    pronoun_mentions = [
        e for e in entries.values()
        if str(e.get("mention_text", "")).lower().strip(".,!?;:\"'") == "he"
    ]

    assert len(raj_mentions) >= 1
    assert len(email_mentions) >= 1
    assert len(role_mentions) >= 1
    assert len(pronoun_mentions) >= 1
    assert all(m.get("entity_id") == RAJESH_ENTITY_ID for m in raj_mentions)
    assert all(m.get("entity_id") == RAJESH_ENTITY_ID for m in email_mentions)
    assert all(m.get("entity_id") == RAJESH_ENTITY_ID for m in role_mentions)
    assert all(m.get("entity_id") == RAJESH_ENTITY_ID for m in pronoun_mentions)

    assert not any("Dropping edge" in msg for msg in handler.messages)

    with neo4j_driver.session() as session:
        rel_count = session.run(
            "MATCH ()-[r:MENTIONS]->(e:Entity {id: $entity_id}) RETURN count(r) AS c",
            entity_id=RAJESH_ENTITY_ID,
        ).single()["c"]
    assert rel_count > 0


@pytest.mark.asyncio(loop_scope="module")
async def test_02_deferred_persistence_only_mentions_survive(
    pg: asyncpg.Pool,
    seeded_pipeline_data,
):
    before_total = await pg.fetchval(
        "SELECT count(*) FROM entity_registry WHERE org_id = $1",
        ORG_ID,
    )

    await _run_processor_for_doc(pg, DOC_DEFER_ID)

    after_total = await pg.fetchval(
        "SELECT count(*) FROM entity_registry WHERE org_id = $1",
        ORG_ID,
    )
    assert after_total > before_total

    row = await pg.fetchrow(
        """
        SELECT entries, resolved_entities, audit, counts
        FROM document_resolution_maps
        WHERE doc_id = $1 AND doc_version = 1
        """,
        DOC_DEFER_ID,
    )
    assert row is not None

    entries = _coerce_json(row["entries"])
    resolved_entities = _coerce_json(row["resolved_entities"])
    audit = _coerce_json(row["audit"])
    counts = _coerce_json(row["counts"])

    dropped = [a for a in audit if a.get("reason") == "no_mentions_edge_after_extraction"]
    persisted = [a for a in audit if str(a.get("reason", "")).startswith("persisted new entity ")]
    assert len(dropped) >= 1
    assert len(entries) >= 6
    assert counts.get("total", 0) >= 6
    assert len(persisted) >= 1
    assert len(dropped) >= len(persisted)
    assert len(persisted) <= len(resolved_entities)
    assert (after_total - before_total) <= counts.get("resolved", 0)


@pytest.mark.asyncio(loop_scope="module")
async def test_03_pronoun_and_role_reference_resolution(
    pg: asyncpg.Pool,
    seeded_pipeline_data,
):
    await _run_processor_for_doc(pg, DOC_PRONOUN_ID)

    row = await pg.fetchrow(
        """
        SELECT entries
        FROM document_resolution_maps
        WHERE doc_id = $1 AND doc_version = 1
        """,
        DOC_PRONOUN_ID,
    )
    assert row is not None
    entries = _coerce_json(row["entries"])

    pronoun_hits = [
        e for e in entries.values()
        if e.get("entity_id") == SARAH_ENTITY_ID and e.get("resolution_method") == "pronoun_lookback"
    ]
    role_hits = [
        e for e in entries.values()
        if e.get("entity_id") == SARAH_ENTITY_ID and e.get("resolution_method") == "role_reference"
    ]

    assert len(pronoun_hits) >= 1
    assert len(role_hits) >= 1
