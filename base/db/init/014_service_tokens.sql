-- ---------------------------------------------------------------------------
-- 014_service_tokens.sql — Scoped service tokens for machine-to-machine auth.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS service_tokens (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name                    TEXT NOT NULL,
    token_hash              TEXT NOT NULL UNIQUE,
    scopes                  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_by_employee_id  TEXT REFERENCES employees(id) ON DELETE SET NULL,
    expires_at              TIMESTAMPTZ NOT NULL,
    revoked_at              TIMESTAMPTZ,
    last_used_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_service_tokens_org
    ON service_tokens(org_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_service_tokens_active
    ON service_tokens(token_hash, expires_at)
    WHERE revoked_at IS NULL;

CREATE OR REPLACE TRIGGER trg_service_tokens_updated_at
  BEFORE UPDATE ON service_tokens
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (id) VALUES ('014_service_tokens')
ON CONFLICT (id) DO NOTHING;
