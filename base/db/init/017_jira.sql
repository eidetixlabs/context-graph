-- 017_jira.sql — Add Jira connector provider and durable Jira issue manifest.
--
-- Enables:
--   - connector_instances.provider = 'jira'
--   - jira_issue_items table used by crawler/ingestion coordination
--
-- Idempotent and safe to re-run.

ALTER TABLE connector_instances
    DROP CONSTRAINT IF EXISTS connector_instances_provider_check;

ALTER TABLE connector_instances
    ADD CONSTRAINT connector_instances_provider_check
    CHECK (provider IN ('notion', 'github', 'fireflies', 'gmail', 'jira'));

CREATE TABLE IF NOT EXISTS jira_issue_items (
    id                      TEXT PRIMARY KEY,
    -- Immutable identity:
    --   jira:{site_host}:{project_key}:{issue_key}

    org_id                  TEXT NOT NULL REFERENCES organizations(id),
    connector_id            TEXT NOT NULL REFERENCES connector_instances(id),
    source_scope            TEXT NOT NULL REFERENCES access_scopes(id),

    site_host               TEXT NOT NULL,
    project_key             TEXT NOT NULL,
    issue_key               TEXT NOT NULL,
    issue_id                TEXT NOT NULL,

    canonical_url           TEXT NOT NULL,
    title                   TEXT,
    issue_type              TEXT,
    status_name             TEXT,
    status_category         TEXT,

    source_created_at       TIMESTAMPTZ NOT NULL,
    source_updated_at       TIMESTAMPTZ NOT NULL,
    source_resolved_at      TIMESTAMPTZ,

    human_comment_count     INTEGER CHECK (human_comment_count IS NULL OR human_comment_count >= 0),
    qualification_reason    TEXT NOT NULL,
    observed_content_marker TEXT NOT NULL,

    order_created_at        TIMESTAMPTZ NOT NULL,

    ingestion_state         TEXT NOT NULL DEFAULT 'pending'
                            CHECK (ingestion_state IN (
                                'pending', 'in_progress', 'completed', 'failed', 'ineligible'
                            )),
    ingestion_error         TEXT,
    last_ingested_at        TIMESTAMPTZ,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (connector_id, project_key, issue_key)
);

CREATE INDEX IF NOT EXISTS idx_jira_issue_items_scope
    ON jira_issue_items(org_id, source_scope);

CREATE INDEX IF NOT EXISTS idx_jira_issue_items_pending_order
    ON jira_issue_items(connector_id, ingestion_state, order_created_at, issue_key)
    WHERE ingestion_state IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_jira_issue_items_source_update
    ON jira_issue_items(connector_id, source_updated_at DESC);

CREATE OR REPLACE TRIGGER trg_jira_issue_items_updated_at
  BEFORE UPDATE ON jira_issue_items
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (id) VALUES ('017_jira')
ON CONFLICT (id) DO NOTHING;
