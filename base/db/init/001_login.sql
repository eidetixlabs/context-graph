-- 001_login.sql — Login-related tables only.
--
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS, CREATE INDEX IF NOT EXISTS)
-- so this file is safe to re-run on EC2 restarts. Postgres executes files in
-- /docker-entrypoint-initdb.d/ only on first initialisation, but the IF NOT EXISTS
-- guards protect against any manual re-runs.
--
-- Tables intentionally omitted here (added in future migrations):
--   identity_mappings, channel_members, access_scopes, documents,
--   document_versions, entity_registry, entity_aliases, audit_log

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

-- updated_at trigger function — reuse across all tables.
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

-- ---------------------------------------------------------------------------
-- schema_migrations
-- Tracks which SQL migration files have been applied.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    id          TEXT PRIMARY KEY,
    -- e.g. '001_login'
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- organizations
-- One row per tenant.  Required FK target for employees.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS organizations (
    id              TEXT PRIMARY KEY,
    -- slug-based: 'org:acme-corp'

    name            TEXT NOT NULL,
    -- Display name: 'Acme Corp'

    slug            TEXT NOT NULL UNIQUE,
    -- URL-safe: 'acme-corp'

    neo4j_database  TEXT NOT NULL DEFAULT 'neo4j',
    -- Neo4j database name for this tenant.
    -- Community Edition only supports 'neo4j'; Enterprise supports per-tenant DBs.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);

CREATE OR REPLACE TRIGGER trg_organizations_updated_at
  BEFORE UPDATE ON organizations
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- employees
-- One row per person with an Eidetix account.
-- auth_provider / auth_provider_id are NULL for local-password users now.
-- They will be populated when IDP is connected.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS employees (
    id                  TEXT PRIMARY KEY,
    -- slug-based: 'emp:sudesh-wasnik'

    org_id              TEXT NOT NULL REFERENCES organizations(id),

    email               TEXT NOT NULL,
    -- Primary email.  Unique per org.

    canonical_name      TEXT NOT NULL,
    -- Full display name: 'Sudesh Wasnik'

    role                TEXT NOT NULL DEFAULT 'member'
                        CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    -- owner  : org-level admin, billing, delete org
    -- admin  : configure connectors, manage employees, view audit log
    -- member : standard access, can use all MCP tools
    -- viewer : read-only, cannot write semantics.json

    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'suspended', 'deactivated')),

    -- IDP fields — NULL for local-password users, populated when IDP is added.
    auth_provider       TEXT
                        CHECK (auth_provider IN (
                            'local', 'google', 'github', 'okta', 'azure-ad', 'saml'
                        )),
    auth_provider_id    TEXT,
    -- Subject ID from IDP (e.g. Google sub, Okta user ID).
    -- Used to match incoming OAuth tokens when IDP is enabled.

    avatar_url          TEXT,

    entity_id           TEXT,
    -- Neo4j Entity node ID: 'entity:person:sudesh-wasnik'
    -- Populated on account creation; used to link Postgres identity to graph nodes.

    last_login_at       TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    deactivated_at      TIMESTAMPTZ,

    UNIQUE (org_id, email)
);

CREATE INDEX IF NOT EXISTS idx_employees_org
    ON employees(org_id);

CREATE INDEX IF NOT EXISTS idx_employees_status
    ON employees(org_id, status);

CREATE INDEX IF NOT EXISTS idx_employees_auth
    ON employees(auth_provider, auth_provider_id)
    WHERE auth_provider_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_employees_entity
    ON employees(entity_id)
    WHERE entity_id IS NOT NULL;

CREATE OR REPLACE TRIGGER trg_employees_updated_at
  BEFORE UPDATE ON employees
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- local_credentials
-- Stores argon2id password hashes for auth_provider='local' users.
-- Kept separate from employees so IDP users never touch this table.
-- When IDP is added: local-password login checks auth_provider first;
-- IDP users have no row here and are authenticated via OAuth/SAML instead.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS local_credentials (
    employee_id     TEXT PRIMARY KEY REFERENCES employees(id) ON DELETE CASCADE,

    password_hash   TEXT NOT NULL,
    -- argon2id hash produced by argon2-cffi.
    -- Never store the plaintext password anywhere.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE TRIGGER trg_local_credentials_updated_at
  BEFORE UPDATE ON local_credentials
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- sessions
-- Bearer tokens for authenticated API calls (MCP, REST API, Desktop App).
-- The raw token is never stored — only its SHA-256 hex digest.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    employee_id     TEXT NOT NULL REFERENCES employees(id) ON DELETE CASCADE,

    token_hash      TEXT NOT NULL UNIQUE,
    -- SHA-256 hex of the bearer token.
    -- Lookup: WHERE token_hash = encode(sha256($incoming_token::bytea), 'hex')

    client_type     TEXT NOT NULL DEFAULT 'desktop-app'
                    CHECK (client_type IN (
                        'desktop-app', 'cursor-mcp', 'claude-mcp', 'browser', 'api'
                    )),

    device_info     TEXT,
    -- Free-text device description: 'macOS 15.1 / Eidetix 0.2.0'

    ip_address      INET,
    -- Recorded at session creation for suspicious-activity detection.

    expires_at      TIMESTAMPTZ NOT NULL,
    -- desktop-app: now() + 30 days
    -- api tokens:  now() + 1 hour (future)

    revoked_at      TIMESTAMPTZ,
    -- NULL = active.  Set on logout or forced revocation.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_employee
    ON sessions(employee_id);

CREATE INDEX IF NOT EXISTS idx_sessions_token
    ON sessions(token_hash);

-- Partial index for active-session lookups (the hot path on every request).
CREATE INDEX IF NOT EXISTS idx_sessions_active
    ON sessions(token_hash, employee_id, expires_at)
    WHERE revoked_at IS NULL;

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('001_login')
    ON CONFLICT (id) DO NOTHING;
