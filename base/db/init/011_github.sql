-- 011_github.sql — Add GitHub connector provider support.
--
-- Extends two existing CHECK constraints to allow the GitHub connector:
--   connector_instances.provider     += 'github'
--   documents.source_provider        += 'github-pdf', 'github-links'
--
-- This file is the only early CHECK rewrite that historically did not record
-- itself in schema_migrations. postgres-init applies it on first start, later
-- migrations (013+) expand the same CHECKs and do self-record, then the
-- migrate runner sees 011 as pending and re-applies it. A bare DROP + ADD of
-- the historically narrower set then fails on existing fireflies/gmail/jira
-- rows (CheckViolationError), or silently shrinks the already-widened
-- constraint on empty / notion-github-only databases. Apply the historical
-- rewrite only when no later CHECK-expanding migration has already run.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM schema_migrations
        WHERE id IN (
            '013_fireflies',
            '014_github_markdown',
            '015_gmail',
            '017_jira',
            '018_github_repository_intelligence'
        )
    ) THEN
        RAISE NOTICE '011_github: skip CHECK rewrite; later migrations already expanded the constraints';
    ELSE
        ALTER TABLE connector_instances
            DROP CONSTRAINT IF EXISTS connector_instances_provider_check;

        ALTER TABLE connector_instances
            ADD CONSTRAINT connector_instances_provider_check
            CHECK (provider IN ('notion', 'github'));

        ALTER TABLE documents
            DROP CONSTRAINT IF EXISTS documents_source_provider_check;

        ALTER TABLE documents
            ADD CONSTRAINT documents_source_provider_check
            CHECK (source_provider IN (
                'notion', 'slack', 'confluence', 'github', 'jira',
                'google-meet', 'cursor-chat', 'claude-chat',
                'browser', 'manual',
                'github-pdf', 'github-links'
            ));
    END IF;
END $$;

INSERT INTO schema_migrations (id) VALUES ('011_github')
ON CONFLICT (id) DO NOTHING;
