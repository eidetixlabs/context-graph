-- 019_jira_connector_runtime.sql — Encrypted Jira credentials and durable run history.
--
-- Enables Portal-managed Jira connectors:
--   - connector_instances.encrypted_token for Fernet-encrypted API tokens
--   - connector_runs for queued/running/succeeded/failed sync attempts
--
-- Idempotent and safe to re-run.

ALTER TABLE connector_instances
    ADD COLUMN IF NOT EXISTS encrypted_token TEXT;

CREATE TABLE IF NOT EXISTS connector_runs (
    id              TEXT PRIMARY KEY,
    -- Format: 'run:{provider}:{uuid}'

    connector_id    TEXT NOT NULL REFERENCES connector_instances(id),
    org_id          TEXT NOT NULL REFERENCES organizations(id),

    trigger         TEXT NOT NULL
                    CHECK (trigger IN ('initial', 'scheduled', 'manual')),

    status          TEXT NOT NULL DEFAULT 'queued'
                    CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),

    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_connector_runs_connector_created
    ON connector_runs(connector_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_connector_runs_status
    ON connector_runs(status, created_at);

CREATE INDEX IF NOT EXISTS idx_connector_runs_active
    ON connector_runs(connector_id, status)
    WHERE status IN ('queued', 'running');

INSERT INTO schema_migrations (id) VALUES ('019_jira_connector_runtime')
ON CONFLICT (id) DO NOTHING;
