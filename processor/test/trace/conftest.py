"""Shared fixtures for trace e2e tests.

Docker stack (full profile: neo4j, postgres, minio, eidetix webserver) is
started once per module and intentionally NOT torn down — containers remain
running for fast iteration between test runs.

Seeded user:
    org_id      : org:trace-e2e
    employee_id : emp:trace-test
    entity_id   : entity:person:trace-test
    email       : trace-test@eidetix.com
    password    : trace-test-password-456
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import asyncpg
import boto3
import httpx
import pytest
import pytest_asyncio
from neo4j import GraphDatabase

_TRACE_DIR = Path(__file__).resolve().parent
if str(_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACE_DIR))

from trace_constants import (  # noqa: E402
    EIDETIX_URL,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PG_DSN,
    S3_BUCKET,
    TEST_EMPLOYEE_ID,
    TEST_ENTITY_ID,
    TEST_ORG_ID,
    TEST_ORG_SLUG,
    TEST_USER_EMAIL,
    TEST_USER_NAME,
    TEST_USER_PASSWORD,
)

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from e2e_base.docker_stack import DockerStackError, ensure_stack_up  # noqa: E402


# ---------------------------------------------------------------------------
# Infrastructure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docker_stack():
    """Start the full docker-compose stack (neo4j, postgres, minio, eidetix).
    No teardown — containers stay running after the test session."""
    try:
        ensure_stack_up(
            profile="full",
            pg_dsn=PG_DSN,
            neo4j_uri=NEO4J_URI,
            minio_url=MINIO_ENDPOINT,
            eidetix_url=EIDETIX_URL,
        )
    except DockerStackError as exc:
        pytest.skip(str(exc))
    yield


@pytest_asyncio.fixture(scope="module")
async def pg_pool(docker_stack) -> AsyncGenerator[asyncpg.Pool, None]:
    pool = await asyncpg.create_pool(
        PG_DSN,
        ssl=False,
        min_size=1,
        max_size=5,
        server_settings={"timezone": "UTC"},
    )
    yield pool
    await pool.close()


@pytest.fixture(scope="module")
def s3_client(docker_stack):
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
def neo4j_driver(docker_stack):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    yield driver
    driver.close()


# ---------------------------------------------------------------------------
# Postgres seeding — org + entity + employee + credentials + alias
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def seeded_user(pg_pool: asyncpg.Pool) -> AsyncGenerator[dict, None]:
    """Seed postgres with an org, entity, employee, and local credentials.

    Idempotent: uses ON CONFLICT DO NOTHING / DO UPDATE so re-runs are safe.
    Returns a dict with all identity fields needed by the tests.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from base.auth.password import hash_password

    password_hash = hash_password(TEST_USER_PASSWORD)

    async with pg_pool.acquire() as conn:
        # 1. Organization
        await conn.execute(
            """
            INSERT INTO organizations (id, name, slug, created_at, updated_at)
            VALUES ($1, $2, $3, now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_ORG_ID,
            "Trace E2E Org",
            TEST_ORG_SLUG,
        )

        # 2. Entity registry (person entity for the test user)
        await conn.execute(
            """
            INSERT INTO entity_registry
                (id, org_id, entity_type, canonical_name, status, created_at, updated_at)
            VALUES ($1, $2, 'person', $3, 'active', now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_ENTITY_ID,
            TEST_ORG_ID,
            TEST_USER_NAME,
        )

        # 3. Employee (links to entity_id so webserver injects _author_entity_id)
        await conn.execute(
            """
            INSERT INTO employees
                (id, org_id, email, canonical_name, role, status,
                 auth_provider, entity_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, 'member', 'active', 'local', $5, now(), now())
            ON CONFLICT (id) DO NOTHING
            """,
            TEST_EMPLOYEE_ID,
            TEST_ORG_ID,
            TEST_USER_EMAIL,
            TEST_USER_NAME,
            TEST_ENTITY_ID,
        )

        # 4. Local credentials (argon2id hash)
        await conn.execute(
            """
            INSERT INTO local_credentials (employee_id, password_hash, created_at, updated_at)
            VALUES ($1, $2, now(), now())
            ON CONFLICT (employee_id) DO UPDATE
                SET password_hash = EXCLUDED.password_hash,
                    updated_at    = now()
            """,
            TEST_EMPLOYEE_ID,
            password_hash,
        )

        # 5. Entity alias so mention resolution can find the user by canonical name
        normalized_alias = TEST_USER_NAME.lower()
        await conn.execute(
            """
            INSERT INTO entity_aliases
                (alias_normalized, entity_id, org_id, alias_raw, source, created_at)
            VALUES ($1, $2, $3, $4, 'manual', now())
            ON CONFLICT (org_id, alias_normalized) DO NOTHING
            """,
            normalized_alias,
            TEST_ENTITY_ID,
            TEST_ORG_ID,
            TEST_USER_NAME,
        )

    yield {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD,
        "entity_id": TEST_ENTITY_ID,
        "employee_id": TEST_EMPLOYEE_ID,
        "canonical_name": TEST_USER_NAME,
        "org_id": TEST_ORG_ID,
    }


# ---------------------------------------------------------------------------
# Auth fixture — login via the running webserver
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def bearer_token(seeded_user: dict) -> str:
    """Log in via POST /auth/login and return a bearer token."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{EIDETIX_URL}/auth/login",
            json={
                "email": seeded_user["email"],
                "password": seeded_user["password"],
                "client_type": "desktop-app",
            },
        )
    resp.raise_for_status()
    return resp.json()["token"]
