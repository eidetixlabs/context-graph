-- 009_document_parent.sql — add parent_ref to documents for Notion child-page support.
--
-- Adds:
--   documents.parent_ref — bare Notion page ID of the parent page (nullable, no FK)
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.

-- ---------------------------------------------------------------------------
-- documents.parent_ref
-- ---------------------------------------------------------------------------

ALTER TABLE documents ADD COLUMN IF NOT EXISTS parent_ref TEXT;

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('009_document_parent')
    ON CONFLICT (id) DO NOTHING;
