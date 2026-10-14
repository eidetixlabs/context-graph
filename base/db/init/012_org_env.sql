-- 012_org_env.sql — Per-org environment routing tables.
--
-- Two new tables that live in the CENTRAL Postgres only:
--
--   org_env       : directory of all orgs -> their env connection coords/creds.
--                   The default org row points at the central DB itself.
--                   edicli writes rows here when adding a new env.
--
--   token_org     : thin routing index  token_hash -> org_id.
--                   Lets the webserver resolve which env to validate against
--                   without scanning every env's sessions table.
--                   Dual-written at session create / cleared at revoke.
--
-- Idempotent: all statements use IF NOT EXISTS / ON CONFLICT DO NOTHING.
-- Safe to apply on every env DB (tables will be empty on non-central envs).

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '012_org_env') THEN

        -- ── org_env ──────────────────────────────────────────────────────────

        CREATE TABLE IF NOT EXISTS org_env (
            org_id          TEXT PRIMARY KEY,
            -- e.g. 'org:acme-corp'

            slug            TEXT NOT NULL UNIQUE,
            -- URL-safe label, matches organizations.slug in each env's own DB.
            -- Used for Google OAuth hd-domain resolution.

            email_domain    TEXT,
            -- e.g. 'acme-corp.com'  — used to resolve org from email at login.
            -- NULL means this org does not claim a domain (manual routing only).
            -- UNIQUE so two orgs can't claim the same domain.

            status          TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'suspended', 'decommissioned')),

            -- ── Postgres connection ───────────────────────────────────────────
            pg_dsn          TEXT NOT NULL,
            -- Full asyncpg DSN, e.g. 'postgresql://eidetix:pass@host:5432/eidetix'
            -- For the default org this equals DATABASE_URL.

            -- ── Neo4j connection ──────────────────────────────────────────────
            neo4j_uri       TEXT NOT NULL,
            -- bolt:// URI, e.g. 'bolt://host:7687'
            neo4j_user      TEXT NOT NULL DEFAULT 'neo4j',
            neo4j_password  TEXT NOT NULL,
            neo4j_database  TEXT NOT NULL DEFAULT 'neo4j',

            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_org_env_slug
            ON org_env(slug);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_org_env_email_domain
            ON org_env(email_domain)
            WHERE email_domain IS NOT NULL;

        CREATE OR REPLACE TRIGGER trg_org_env_updated_at
          BEFORE UPDATE ON org_env
          FOR EACH ROW EXECUTE FUNCTION set_updated_at();

        -- ── token_org ─────────────────────────────────────────────────────────

        CREATE TABLE IF NOT EXISTS token_org (
            token_hash      TEXT PRIMARY KEY,
            -- SHA-256 hex of the bearer token (same hash used in sessions.token_hash).

            org_id          TEXT NOT NULL,
            -- Which org's env this token belongs to.
            -- Not a FK — org_env may not exist yet on non-central DBs.

            expires_at      TIMESTAMPTZ NOT NULL,
            -- Mirrors sessions.expires_at; used for lazy cleanup.

            revoked_at      TIMESTAMPTZ
            -- NULL = active.  Set when the session is revoked/logged out.
        );

        CREATE INDEX IF NOT EXISTS idx_token_org_org
            ON token_org(org_id);

        -- Partial index for the hot lookup path (active tokens only).
        CREATE INDEX IF NOT EXISTS idx_token_org_active
            ON token_org(token_hash, org_id)
            WHERE revoked_at IS NULL;

        -- ── oauth_attempts.org_id ─────────────────────────────────────────
        -- Stores which org the Google OAuth attempt resolved to, so
        -- /auth/google/exchange knows which env Postgres to issue the session in.
        ALTER TABLE oauth_attempts
            ADD COLUMN IF NOT EXISTS org_id TEXT;

        INSERT INTO schema_migrations (id) VALUES ('012_org_env');

    END IF;
END $$;
