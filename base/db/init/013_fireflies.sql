-- ---------------------------------------------------------------------------
-- 013_fireflies.sql — Add 'fireflies' to provider CHECK constraints.
--
-- The connector_instances.provider, documents.source_provider, and
-- identity_mappings.provider CHECK constraints are defined in 003_connectors.sql,
-- 005_entities.sql, and patched by 011_github.sql. Those files are only applied
-- on fresh installs; this migration patches existing databases by dropping and
-- recreating the constraints.
-- ---------------------------------------------------------------------------

-- connector_instances.provider
ALTER TABLE connector_instances
    DROP CONSTRAINT IF EXISTS connector_instances_provider_check;

ALTER TABLE connector_instances
    ADD CONSTRAINT connector_instances_provider_check
    CHECK (provider IN ('notion', 'github', 'fireflies'));

-- documents.source_provider
ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_provider_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_source_provider_check
    CHECK (source_provider IN (
        'notion', 'slack', 'confluence', 'github', 'jira',
        'google-meet', 'cursor-chat', 'claude-chat',
        'browser', 'manual', 'fireflies',
        'github-pdf', 'github-links'
    ));

-- identity_mappings.provider
ALTER TABLE identity_mappings
    DROP CONSTRAINT IF EXISTS identity_mappings_provider_check;

ALTER TABLE identity_mappings
    ADD CONSTRAINT identity_mappings_provider_check
    CHECK (provider IN (
        'slack', 'confluence', 'github', 'jira',
        'google-meet', 'notion', 'linear', 'fireflies'
    ));

INSERT INTO schema_migrations (id) VALUES ('013_fireflies')
ON CONFLICT (id) DO NOTHING;
