-- 020_cloud_tenants.sql — Invite-only cloud control plane.
--
-- Applied on every database (central + env). Rows are used on the central
-- registry; env copies stay empty unless the same Postgres is shared.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '020_cloud_tenants') THEN

        CREATE TABLE IF NOT EXISTS cloud_tenants (
            org_id          TEXT PRIMARY KEY,
            slug            TEXT NOT NULL UNIQUE,
            name            TEXT NOT NULL,
            email_domain    TEXT,
            owner_email     TEXT NOT NULL,
            plan            TEXT NOT NULL DEFAULT 'beta',
            status          TEXT NOT NULL DEFAULT 'provisioning'
                            CHECK (status IN (
                                'provisioning', 'ready', 'degraded',
                                'suspended', 'decommissioned'
                            )),
            provision_error TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            ready_at        TIMESTAMPTZ
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_cloud_tenants_email_domain
            ON cloud_tenants(email_domain)
            WHERE email_domain IS NOT NULL;

        CREATE TABLE IF NOT EXISTS tenant_provision_ops (
            id              TEXT PRIMARY KEY,
            org_id          TEXT NOT NULL,
            action          TEXT NOT NULL,
            status          TEXT NOT NULL
                            CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
            detail          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_tenant_provision_ops_org
            ON tenant_provision_ops(org_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS org_invitations (
            id              TEXT PRIMARY KEY,
            org_id          TEXT NOT NULL,
            email           TEXT NOT NULL,
            role            TEXT NOT NULL DEFAULT 'member'
                            CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
            token_hash      TEXT NOT NULL UNIQUE,
            invited_by      TEXT,
            accepted_at     TIMESTAMPTZ,
            expires_at      TIMESTAMPTZ NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_org_invitations_org
            ON org_invitations(org_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_org_invitations_email
            ON org_invitations(org_id, email)
            WHERE accepted_at IS NULL;

        CREATE TABLE IF NOT EXISTS platform_operators (
            email           TEXT PRIMARY KEY,
            added_by        TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS platform_audit (
            id              BIGSERIAL PRIMARY KEY,
            org_id          TEXT,
            actor_id        TEXT,
            action          TEXT NOT NULL,
            target_type     TEXT,
            target_id       TEXT,
            detail          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_platform_audit_org
            ON platform_audit(org_id, created_at DESC);

        INSERT INTO schema_migrations (id) VALUES ('020_cloud_tenants');

    END IF;
END $$;
