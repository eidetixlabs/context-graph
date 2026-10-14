-- 003_connectors.sql — Connector infrastructure tables.
--
-- Adds:
--   connector_instances   — one row per connected external tool per org (config + runtime state)
--   connector_checkpoints — stateful cursor per connector (advances per document, not per run)
--   access_scopes         — registry of discovered external channels/spaces/repos
--   channel_members       — crawler-maintained RBAC source of truth
--   documents             — document metadata + processing state
--   document_versions     — append-only version history
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.

-- ---------------------------------------------------------------------------
-- connector_instances
-- One row per connected external tool per org.
-- Stores non-secret config as JSONB. Token is injected via CONNECTOR_TOKEN env var at runtime.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS connector_instances (
    id               TEXT PRIMARY KEY,
    -- Format: 'conn:{provider}:{org_slug}'
    -- e.g. 'conn:notion:acme-corp'

    org_id           TEXT NOT NULL REFERENCES organizations(id),

    provider         TEXT NOT NULL
                     CHECK (provider IN ('notion')),
    -- Extend CHECK as more providers are added.

    config           JSONB NOT NULL DEFAULT '{}',
    -- Non-secret configuration. Provider-specific.
    -- Notion example: {"teamspace_ids": ["TEAMSPACE-XYZ"], "exclude_page_ids": []}

    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'paused', 'error', 'revoked')),
    -- active  : normal operation
    -- paused  : admin-paused, workers skip this connector
    -- error   : consecutive auth failures; requires manual intervention
    -- revoked : token was revoked; connector disabled

    error_message    TEXT,
    -- Set when status = 'error'. Human-readable description.

    created_by       TEXT DEFAULT 'system',
    -- Nullable; workers self-register without a human actor (default 'system').

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (org_id, provider)
    -- One connector per provider per org.
);

CREATE INDEX IF NOT EXISTS idx_connector_instances_org
    ON connector_instances(org_id);

CREATE INDEX IF NOT EXISTS idx_connector_instances_status
    ON connector_instances(org_id, status);

CREATE OR REPLACE TRIGGER trg_connector_instances_updated_at
  BEFORE UPDATE ON connector_instances
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- connector_checkpoints
-- Stateful cursor per connector per run type.
-- Workers are long-lived (or resume on restart). After each item is fully
-- written to S3 + Postgres, the worker updates last_cursor to that item's
-- source timestamp. On restart, the worker resumes from last_cursor (>=).
--
-- No run-history table. Observability via logs + documents table state.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS connector_checkpoints (
    connector_id  TEXT NOT NULL REFERENCES connector_instances(id),

    run_type      TEXT NOT NULL
                  CHECK (run_type IN ('crawl', 'ingest')),

    last_cursor   TIMESTAMPTZ,
    -- Timestamp of the last successfully processed item.
    -- NULL = never run; worker fetches from the beginning.
    -- On resume: worker queries provider with last_edited_time >= last_cursor.
    -- Using >= is intentional: guarantees no data loss at the cost of
    -- re-fetching the boundary item; content_hash dedup skips it if unchanged.

    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (connector_id, run_type)
);

-- ---------------------------------------------------------------------------
-- access_scopes
-- Registry of known external channels/spaces/repos discovered by connectors.
-- The scope ID is the RBAC key stamped on documents and Neo4j nodes.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS access_scopes (
    id              TEXT PRIMARY KEY,
    -- Format: 'scope:{provider}:{external_id}'
    -- e.g. 'scope:notion:TEAMSPACE-XYZ'
    --      'scope:slack:C0123ABCD'

    org_id          TEXT NOT NULL REFERENCES organizations(id),

    source_provider TEXT NOT NULL,
    -- 'notion', 'slack', 'confluence', etc.

    source_ref      TEXT NOT NULL,
    -- The external system's own ID for this channel/space.

    source_label    TEXT NOT NULL,
    -- Human-readable name: 'Engineering', '#eng-decisions'

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (org_id, source_provider, source_ref)
);

CREATE INDEX IF NOT EXISTS idx_access_scopes_org
    ON access_scopes(org_id);

CREATE INDEX IF NOT EXISTS idx_access_scopes_provider
    ON access_scopes(org_id, source_provider);

-- ---------------------------------------------------------------------------
-- channel_members
-- Crawler-maintained: for every access scope, which employees are members.
-- Sole source of truth for data access (RBAC).
--
-- For Notion v1, identity mapping is manual: employee_id may be NULL.
-- external_user_id stores the raw provider user ID. Once an admin creates
-- the identity_mapping row, employee_id is backfilled.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS channel_members (
    channel_id        TEXT NOT NULL,
    -- Matches access_scopes.id: 'scope:notion:TEAMSPACE-XYZ'

    employee_id       TEXT REFERENCES employees(id),
    -- NULL if identity mapping has not yet been done.

    external_user_id  TEXT,
    -- Raw user ID from the external system.
    -- Notion: 'notion-user-abc123'  Slack: 'U0123ABCD'
    -- Kept even after employee_id is resolved, for audit.

    source_provider   TEXT NOT NULL,
    -- Redundant with channel_id prefix; useful for filtered queries.

    crawled_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (channel_id, external_user_id),
    -- Keyed on external_user_id because employee_id may be NULL.

    CHECK (employee_id IS NOT NULL OR external_user_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_channel_members_employee
    ON channel_members(employee_id)
    WHERE employee_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_channel_members_channel
    ON channel_members(channel_id);

CREATE INDEX IF NOT EXISTS idx_channel_members_provider
    ON channel_members(source_provider, channel_id);

-- ---------------------------------------------------------------------------
-- documents
-- One row per external document. Tracks metadata, current processing state,
-- and the S3 key for the current version. Raw content lives in S3 only.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id                  TEXT PRIMARY KEY,
    -- Format: 'doc:{provider}:{external_ref}'
    -- e.g. 'doc:notion:PAGE-abc123'

    org_id              TEXT NOT NULL REFERENCES organizations(id),

    source_provider     TEXT NOT NULL
                        CHECK (source_provider IN (
                            'notion', 'slack', 'confluence', 'github', 'jira',
                            'google-meet', 'cursor-chat', 'claude-chat',
                            'browser', 'manual', 'fireflies'
                        )),

    source_scope        TEXT NOT NULL,
    -- The access scope this document originated from.
    -- Matches access_scopes.id: 'scope:notion:TEAMSPACE-XYZ'
    -- RBAC key: user can see this doc iff in channel_members for this scope.

    source_ref          TEXT NOT NULL,
    -- External system's own ID: Notion page ID, Slack thread ts, etc.

    title               TEXT,

    author_employee_id  TEXT REFERENCES employees(id),
    -- Nullable. Resolved from external author ID via identity_mappings.

    -- ── Current version ──────────────────────────────────────────────────
    current_version     INTEGER NOT NULL DEFAULT 1,

    content_hash        TEXT NOT NULL,
    -- SHA-256 hex of the raw content at the current version.
    -- Used to detect edits without re-fetching.

    char_count          INTEGER,

    s3_key              TEXT NOT NULL,
    -- Path to the current version's raw content in S3.
    -- Format: '{org_id}/{provider}/raw/{source_ref}/v{version}.json'
    -- e.g. 'org:acme-corp/notion/raw/PAGE-abc123/v1.json'

    -- ── Processing state machine ──────────────────────────────────────────
    processing_state    TEXT NOT NULL DEFAULT 'raw'
                        CHECK (processing_state IN (
                            'raw',        -- ingested, not yet processed
                            'scoring',    -- signal scoring in progress
                            'scored',     -- scored; awaiting extraction decision
                            'extracting', -- LLM extraction in progress
                            'extracted',  -- extraction complete, nodes in Neo4j
                            'skipped',    -- score too low, not extracted
                            'failed'      -- processing error
                        )),

    processing_version  INTEGER NOT NULL DEFAULT 1,

    signal_score        REAL,
    extraction_model    TEXT,
    extraction_error    TEXT,

    source_created_at   TIMESTAMPTZ,
    -- When the document was originally created in the source system.

    source_updated_at   TIMESTAMPTZ,
    -- When the document was last edited in the source system.
    -- Used as the cursor value in connector_checkpoints.

    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (org_id, source_provider, source_ref)
);

CREATE INDEX IF NOT EXISTS idx_docs_org_scope
    ON documents(org_id, source_scope);

CREATE INDEX IF NOT EXISTS idx_docs_processing
    ON documents(org_id, processing_state, processing_version);

CREATE INDEX IF NOT EXISTS idx_docs_source_ref
    ON documents(org_id, source_provider, source_ref);

CREATE INDEX IF NOT EXISTS idx_docs_hash
    ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_docs_source_updated
    ON documents(org_id, source_provider, source_updated_at);

CREATE OR REPLACE TRIGGER trg_documents_updated_at
  BEFORE UPDATE ON documents
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- document_versions
-- Append-only history. documents always points to current version;
-- this table holds the full audit trail.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_versions (
    doc_id          TEXT NOT NULL REFERENCES documents(id),

    version         INTEGER NOT NULL,
    -- Monotonically increasing per document, starting at 1.

    content_hash    TEXT NOT NULL,

    s3_key          TEXT NOT NULL,
    -- Immutable S3 path for this specific version. Never overwritten.

    char_count      INTEGER,

    change_summary  TEXT,

    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (doc_id, version)
);

CREATE INDEX IF NOT EXISTS idx_doc_versions_hash
    ON document_versions(content_hash);

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('003_connectors')
    ON CONFLICT (id) DO NOTHING;
