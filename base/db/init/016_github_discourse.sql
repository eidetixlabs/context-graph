-- ---------------------------------------------------------------------------
-- 016_github_discourse.sql — Durable GitHub discourse crawl manifest.
--
-- Adds github_discourse_items, a GitHub-specific eligibility + ingestion queue
-- for completed Issues and merged pull requests with rich discussion.
--
-- This table intentionally stores metadata only (ordering, qualification,
-- provenance timestamps, retry state). Raw source content remains versioned in S3.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS github_discourse_items (
    id                      TEXT PRIMARY KEY,
    -- Immutable identity:
    --   github:{owner}/{repo}:issue:{number}
    --   github:{owner}/{repo}:pr:{number}

    org_id                  TEXT NOT NULL REFERENCES organizations(id),
    connector_id            TEXT NOT NULL REFERENCES connector_instances(id),
    source_scope            TEXT NOT NULL REFERENCES access_scopes(id),

    repository              TEXT NOT NULL,
    -- "{owner}/{repo}" (e.g. "acme/payments")

    item_kind               TEXT NOT NULL
                            CHECK (item_kind IN ('issue', 'pull_request')),
    item_number             INTEGER NOT NULL CHECK (item_number > 0),

    canonical_url           TEXT NOT NULL,
    title                   TEXT,

    source_created_at       TIMESTAMPTZ NOT NULL,
    source_updated_at       TIMESTAMPTZ NOT NULL,
    source_closed_at        TIMESTAMPTZ,
    source_merged_at        TIMESTAMPTZ,

    state_reason            TEXT,
    human_comment_count     INTEGER CHECK (human_comment_count IS NULL OR human_comment_count >= 0),
    qualification_reason    TEXT NOT NULL,

    -- Deterministic oldest-first ordering:
    --   1) order_created_at ASC
    --   2) order_kind_rank ASC (issue=0, pull_request=1)
    --   3) item_number ASC
    order_created_at        TIMESTAMPTZ NOT NULL,
    order_kind_rank         SMALLINT NOT NULL CHECK (order_kind_rank IN (0, 1)),

    observed_content_marker TEXT NOT NULL,
    -- Stable marker from source metadata (e.g. updated_at + discussion counts).
    -- Used to decide whether completed rows should be returned to pending.

    ingestion_state         TEXT NOT NULL DEFAULT 'pending'
                            CHECK (ingestion_state IN (
                                'pending', 'in_progress', 'completed', 'failed', 'ineligible'
                            )),
    ingestion_error         TEXT,
    last_ingested_at        TIMESTAMPTZ,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (connector_id, item_kind, repository, item_number)
);

CREATE INDEX IF NOT EXISTS idx_github_discourse_items_scope
    ON github_discourse_items(org_id, source_scope);

CREATE INDEX IF NOT EXISTS idx_github_discourse_items_pending_order
    ON github_discourse_items(
        connector_id,
        ingestion_state,
        order_created_at,
        order_kind_rank,
        item_number
    )
    WHERE ingestion_state IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_github_discourse_items_source_update
    ON github_discourse_items(connector_id, source_updated_at DESC);

CREATE OR REPLACE TRIGGER trg_github_discourse_items_updated_at
  BEFORE UPDATE ON github_discourse_items
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (id) VALUES ('016_github_discourse')
ON CONFLICT (id) DO NOTHING;
