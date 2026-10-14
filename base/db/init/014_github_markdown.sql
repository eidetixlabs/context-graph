-- ---------------------------------------------------------------------------
-- 014_github_markdown.sql — Add 'github-markdown' to documents.source_provider.
--
-- Existing provider CHECK constraints are managed via idempotent drop + recreate
-- migrations. This migration extends documents_source_provider_check so ingestion
-- can persist GitHub markdown artifacts.
-- ---------------------------------------------------------------------------

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_provider_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_source_provider_check
    CHECK (source_provider IN (
        'notion', 'slack', 'confluence', 'github', 'jira',
        'google-meet', 'cursor-chat', 'claude-chat',
        'browser', 'manual', 'fireflies',
        'github-pdf', 'github-links', 'github-markdown'
    ));

INSERT INTO schema_migrations (id) VALUES ('014_github_markdown')
ON CONFLICT (id) DO NOTHING;
