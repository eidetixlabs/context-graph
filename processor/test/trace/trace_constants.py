"""Shared constants for the trace e2e test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from e2e_base.docker_stack import (  # noqa: E402, F401
    DEFAULT_EIDETIX_URL,
    DEFAULT_MINIO_URL,
    DEFAULT_NEO4J_URI,
    DEFAULT_PG_DSN,
)

# ---------------------------------------------------------------------------
# Connection defaults (overridable via env)
# ---------------------------------------------------------------------------

PG_DSN = os.environ.get("DATABASE_URL", DEFAULT_PG_DSN)
MINIO_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", DEFAULT_MINIO_URL)
MINIO_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "eidetix")
NEO4J_URI = os.environ.get("NEO4J_URI", DEFAULT_NEO4J_URI)
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "devpassword")
EIDETIX_URL = os.environ.get("EIDETIX_URL", DEFAULT_EIDETIX_URL)
API_KEY = os.environ.get("API_KEY", "dev-api-key")

# ---------------------------------------------------------------------------
# Test identity constants
# ---------------------------------------------------------------------------

TEST_ORG_ID = "org:trace-e2e"
TEST_ORG_SLUG = "trace-e2e"
TEST_EMPLOYEE_ID = "emp:trace-test"
TEST_ENTITY_ID = "entity:person:trace-test"
TEST_USER_EMAIL = "trace-test@eidetix.com"
TEST_USER_PASSWORD = "trace-test-password-456"
TEST_USER_NAME = "Trace Test User"
