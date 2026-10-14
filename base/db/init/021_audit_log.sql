-- 021_audit_log.sql — Per-org append-only audit trail.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '021_audit_log') THEN

        CREATE TABLE IF NOT EXISTS audit_log (
            id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            org_id          TEXT NOT NULL,
            event_type      TEXT NOT NULL,
            actor_id        TEXT NOT NULL,
            target_type     TEXT NOT NULL,
            target_id       TEXT NOT NULL,
            detail          JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_audit_log_org_time
            ON audit_log(org_id, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_audit_log_target
            ON audit_log(org_id, target_type, target_id);

        INSERT INTO schema_migrations (id) VALUES ('021_audit_log');

    END IF;
END $$;
