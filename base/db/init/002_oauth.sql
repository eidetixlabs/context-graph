-- 002_oauth.sql — OAuth attempt tracking for the browser-based OAuth flow.
--
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS)
-- so this file is safe to re-run at any time.
--
-- The oauth_attempts table tracks in-flight Google OAuth flows initiated by the
-- desktop app. Flow:
--   1. Desktop calls POST /auth/google/start → creates a pending row
--   2. User completes OAuth in browser → row updated to completed + bootstrap_token set
--   3. Desktop polls GET /auth/google/status → receives bootstrap_token on completion
--   4. Desktop calls POST /auth/google/exchange → bootstrap_token consumed, session issued

-- ---------------------------------------------------------------------------
-- oauth_attempts
-- One row per in-flight or completed OAuth flow from a desktop app.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS oauth_attempts (
    id              TEXT PRIMARY KEY,
    -- URL-safe random ID returned to the desktop app as attempt_id.

    state           TEXT NOT NULL UNIQUE,
    -- Random CSRF state parameter sent to Google and verified on callback.
    -- Must be unique; never reused.

    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'completed', 'failed', 'expired')),

    employee_id     TEXT REFERENCES employees(id),
    -- Set when the OAuth flow completes successfully.

    bootstrap_token TEXT,
    -- Short-lived (5 min) one-time token stored as plaintext.
    -- Returned to the desktop via GET /auth/google/status when status=completed.
    -- Security: only reachable by the holder of the opaque attempt_id, over HTTPS.
    -- After the desktop calls /auth/google/exchange, set to NULL (single-use).

    error_message   TEXT,
    -- Human-readable error if status = 'failed'.

    ip_address      INET,
    -- IP of the desktop app that initiated the attempt.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,
    -- Attempts expire after 10 minutes if not completed.

    completed_at    TIMESTAMPTZ
    -- Timestamp of successful OAuth completion.
);

CREATE INDEX IF NOT EXISTS idx_oauth_attempts_state
    ON oauth_attempts(state);

-- Partial index for active-attempt lookups (pending status poll path).
CREATE INDEX IF NOT EXISTS idx_oauth_attempts_pending
    ON oauth_attempts(id, expires_at)
    WHERE status = 'pending';

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('002_oauth')
    ON CONFLICT (id) DO NOTHING;
