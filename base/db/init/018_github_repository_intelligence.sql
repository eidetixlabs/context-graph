-- 018_github_repository_intelligence.sql
--
-- Repository intelligence profiles, attributed repository notes, and durable
-- cross-store indexing jobs. This migration also restores the complete set of
-- document providers after older migrations recreated the CHECK constraint.

ALTER TABLE connector_instances
    DROP CONSTRAINT IF EXISTS connector_instances_provider_check;

ALTER TABLE connector_instances
    ADD CONSTRAINT connector_instances_provider_check
    CHECK (provider IN ('notion', 'github', 'fireflies', 'gmail', 'jira'));

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_provider_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_source_provider_check
    CHECK (source_provider IN (
        'notion', 'slack', 'confluence', 'github', 'jira',
        'google-meet', 'cursor-chat', 'claude-chat',
        'browser', 'manual', 'fireflies', 'gmail',
        'github-pdf', 'github-links', 'github-markdown',
        'github-structure', 'github-profile'
    ));

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_processing_state_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_processing_state_check
    CHECK (processing_state IN (
        'raw', 'scoring', 'scored', 'extracting', 'indexing',
        'extracted', 'skipped', 'failed'
    ));

CREATE TABLE IF NOT EXISTS github_repository_profiles (
    id                  TEXT PRIMARY KEY,
    org_id              TEXT NOT NULL REFERENCES organizations(id),
    connector_id        TEXT NOT NULL REFERENCES connector_instances(id),
    source_scope        TEXT NOT NULL REFERENCES access_scopes(id),
    repository          TEXT NOT NULL,
    ref                 TEXT NOT NULL,
    commit_sha          TEXT NOT NULL,
    evidence_hash       TEXT NOT NULL,
    evidence_s3_key     TEXT NOT NULL,
    trigger_reason      TEXT NOT NULL DEFAULT 'scheduled',
    release_metadata    JSONB NOT NULL DEFAULT '{}',
    refresh_requested_at TIMESTAMPTZ,
    refresh_reason      TEXT,
    refresh_event_id    TEXT,
    refresh_force_generation BOOLEAN NOT NULL DEFAULT false,
    generation_state    TEXT NOT NULL DEFAULT 'pending'
                        CHECK (generation_state IN (
                            'pending', 'in_progress', 'completed', 'failed'
                        )),
    generation_error    TEXT,
    profile_doc_id      TEXT REFERENCES documents(id),
    last_generated_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (connector_id, repository, ref)
);

ALTER TABLE github_repository_profiles
    ADD COLUMN IF NOT EXISTS refresh_requested_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS refresh_reason TEXT,
    ADD COLUMN IF NOT EXISTS refresh_event_id TEXT,
    ADD COLUMN IF NOT EXISTS refresh_force_generation BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_github_repository_profiles_pending
    ON github_repository_profiles(connector_id, generation_state, updated_at)
    WHERE generation_state IN ('pending', 'failed');

CREATE INDEX IF NOT EXISTS idx_github_repository_profiles_refresh
    ON github_repository_profiles(connector_id, refresh_requested_at)
    WHERE refresh_requested_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_github_repository_profiles_scope
    ON github_repository_profiles(org_id, source_scope, repository);

CREATE OR REPLACE TRIGGER trg_github_repository_profiles_updated_at
  BEFORE UPDATE ON github_repository_profiles
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS repository_context_notes (
    id                  TEXT PRIMARY KEY,
    org_id              TEXT NOT NULL REFERENCES organizations(id),
    source_scope        TEXT NOT NULL REFERENCES access_scopes(id),
    repository          TEXT NOT NULL,
    author_employee_id  TEXT NOT NULL REFERENCES employees(id),
    category            TEXT NOT NULL DEFAULT 'insight'
                        CHECK (category IN (
                            'purpose', 'architecture', 'integration',
                            'testing', 'operations', 'contributing', 'insight'
                        )),
    body                TEXT NOT NULL CHECK (length(trim(body)) > 0),
    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'superseded')),
    supersedes_id       TEXT REFERENCES repository_context_notes(id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE repository_context_notes
    DROP CONSTRAINT IF EXISTS repository_context_notes_body_length_check;
ALTER TABLE repository_context_notes
    ADD CONSTRAINT repository_context_notes_body_length_check
    CHECK (length(body) <= 10000);

ALTER TABLE repository_context_notes
    DROP CONSTRAINT IF EXISTS repository_context_notes_category_length_check;
ALTER TABLE repository_context_notes
    ADD CONSTRAINT repository_context_notes_category_length_check
    CHECK (length(category) <= 64);

CREATE INDEX IF NOT EXISTS idx_repository_context_notes_active
    ON repository_context_notes(org_id, source_scope, repository, created_at DESC)
    WHERE status = 'active';

CREATE OR REPLACE TRIGGER trg_repository_context_notes_updated_at
  BEFORE UPDATE ON repository_context_notes
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS document_index_jobs (
    doc_id              TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version    INTEGER NOT NULL,
    payload_hash        TEXT NOT NULL,
    payload_s3_key      TEXT NOT NULL,
    vector_state        TEXT NOT NULL DEFAULT 'pending'
                        CHECK (vector_state IN ('pending', 'completed', 'failed')),
    graph_state         TEXT NOT NULL DEFAULT 'pending'
                        CHECK (graph_state IN ('pending', 'completed', 'failed')),
    vector_error        TEXT,
    graph_error         TEXT,
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_id, document_version, payload_hash)
);

CREATE INDEX IF NOT EXISTS idx_document_index_jobs_pending
    ON document_index_jobs(updated_at)
    WHERE vector_state != 'completed' OR graph_state != 'completed';

CREATE OR REPLACE TRIGGER trg_document_index_jobs_updated_at
  BEFORE UPDATE ON document_index_jobs
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (id)
VALUES ('018_github_repository_intelligence')
ON CONFLICT (id) DO NOTHING;
