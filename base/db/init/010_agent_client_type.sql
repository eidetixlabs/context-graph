-- 010_agent_client_type.sql — add agent session client type.
--
-- Idempotent migration guarded by schema_migrations.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '010_agent_client_type') THEN
        ALTER TABLE sessions
            DROP CONSTRAINT IF EXISTS sessions_client_type_check;

        ALTER TABLE sessions
            ADD CONSTRAINT sessions_client_type_check
            CHECK (client_type IN (
                'desktop-app', 'cursor-mcp', 'claude-mcp', 'browser', 'api', 'agent'
            ));

        INSERT INTO schema_migrations (id) VALUES ('010_agent_client_type');
    END IF;
END $$;
