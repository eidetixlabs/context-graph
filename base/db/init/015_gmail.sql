-- 015_gmail.sql — Gmail connector: OAuth purpose tracking + per-user credentials.
--
-- Adds:
--   oauth_attempts.purpose  — distinguishes the browser-based OAuth flow used
--                             for desktop login vs. authenticated Gmail consent.
--   gmail_connections       — one row per employee who has connected a Gmail
--                             account. Stores the encrypted refresh token, a
--                             cached access token, and the per-user crawl /
--                             ingestion checkpoints used by later phases.
--   Provider CHECK constraints on connector_instances, documents, and
--   identity_mappings gain 'gmail' so later phases can reference it.
--
-- Idempotent: safe to re-run.

-- ---------------------------------------------------------------------------
-- oauth_attempts.purpose
-- 'login' — the existing desktop Google-login flow (unchanged behaviour).
-- 'gmail' — Gmail consent initiated by an already-authenticated employee.
--           employee_id + org_id are populated at attempt-creation time
--           (not at callback time, since the browser has no session).
-- ---------------------------------------------------------------------------

ALTER TABLE oauth_attempts
    ADD COLUMN IF NOT EXISTS purpose TEXT NOT NULL DEFAULT 'login';

ALTER TABLE oauth_attempts
    DROP CONSTRAINT IF EXISTS oauth_attempts_purpose_check;

ALTER TABLE oauth_attempts
    ADD CONSTRAINT oauth_attempts_purpose_check
    CHECK (purpose IN ('login', 'gmail'));

-- oauth_attempts.org_id already exists (added by 012_org_env.sql).
-- For 'gmail' attempts it is populated at attempt-creation time (the
-- employee's org is already known); for 'login' attempts it is set once
-- the Workspace domain is resolved in the callback, exactly as today.

-- ---------------------------------------------------------------------------
-- gmail_connections
-- One row per employee who has connected a Gmail account. Lives in the
-- employee's organization environment database (same as `employees`,
-- `sessions`, etc.) — not the central database.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS gmail_connections (
    id                        TEXT PRIMARY KEY,
    -- Format: 'gmailconn:{employee_id}'

    org_id                    TEXT NOT NULL REFERENCES organizations(id),

    employee_id               TEXT NOT NULL REFERENCES employees(id),

    gmail_address              TEXT NOT NULL,
    -- The connected Google account's email address (from the id_token).

    encrypted_refresh_token    TEXT NOT NULL,
    -- Fernet-encrypted refresh token (base/crypto.py). Never stored plaintext.

    encrypted_access_token     TEXT,
    -- Fernet-encrypted cached access token. Nullable — refreshed on demand.

    access_token_expires_at    TIMESTAMPTZ,

    token_stored_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- When the current refresh token was captured. Also anchors the initial
    -- 24-hour crawl/ingestion window (token_stored_at - 24h) until a
    -- checkpoint exists.

    crawl_checkpoint           TIMESTAMPTZ,
    -- Per-user cursor for the Gmail crawler (Phase 3). NULL = never run.

    ingestion_checkpoint       TIMESTAMPTZ,
    -- Per-user cursor for the Gmail ingestion worker (Phase 4). NULL = never run.

    status                     TEXT NOT NULL DEFAULT 'active'
                               CHECK (status IN ('active', 'error', 'revoked')),
    -- active  : normal operation
    -- error   : consecutive auth/rate-limit failures; requires reconnect
    -- revoked : user denied/revoked consent; requires reconnect

    error_message              TEXT,
    -- Human-readable description when status != 'active'.

    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (org_id, employee_id)
    -- One Gmail connection per employee. Reconnecting overwrites this row.
);

CREATE INDEX IF NOT EXISTS idx_gmail_connections_org
    ON gmail_connections(org_id);

CREATE INDEX IF NOT EXISTS idx_gmail_connections_status
    ON gmail_connections(org_id, status);

-- A mailbox has exactly one authoritative owner within an organization.
-- Resolve pre-index duplicates deterministically so existing deployments can
-- adopt the invariant without exposing oscillating identity ownership.
WITH duplicate_active_connections AS (
    SELECT id,
           row_number() OVER (
               PARTITION BY org_id, lower(gmail_address)
               ORDER BY created_at, id
           ) AS owner_rank
    FROM gmail_connections
    WHERE status = 'active'
)
UPDATE gmail_connections gc
SET status = 'error',
    error_message = 'Mailbox is already connected by another employee'
FROM duplicate_active_connections duplicates
WHERE gc.id = duplicates.id AND duplicates.owner_rank > 1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_gmail_connections_active_mailbox
    ON gmail_connections(org_id, lower(gmail_address))
    WHERE status = 'active';

CREATE OR REPLACE TRIGGER trg_gmail_connections_updated_at
  BEFORE UPDATE ON gmail_connections
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Provider CHECK constraints — add 'gmail' for later phases.
-- ---------------------------------------------------------------------------

ALTER TABLE connector_instances
    DROP CONSTRAINT IF EXISTS connector_instances_provider_check;

ALTER TABLE connector_instances
    ADD CONSTRAINT connector_instances_provider_check
    CHECK (provider IN ('notion', 'github', 'fireflies', 'gmail'));

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_provider_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_source_provider_check
    CHECK (source_provider IN (
        'notion', 'slack', 'confluence', 'github', 'jira',
        'google-meet', 'cursor-chat', 'claude-chat',
        'browser', 'manual', 'fireflies',
        'github-pdf', 'github-links', 'gmail'
    ));

ALTER TABLE identity_mappings
    DROP CONSTRAINT IF EXISTS identity_mappings_provider_check;

ALTER TABLE identity_mappings
    ADD CONSTRAINT identity_mappings_provider_check
    CHECK (provider IN (
        'slack', 'confluence', 'github', 'jira',
        'google-meet', 'notion', 'linear', 'fireflies', 'gmail'
    ));

INSERT INTO schema_migrations (id) VALUES ('015_gmail')
ON CONFLICT (id) DO NOTHING;
