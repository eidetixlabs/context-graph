-- 007_drop_chunk_text.sql — Remove chunk_text column from document_chunks.
--
-- Raw text for each chunk is reconstructable from S3 using char_offset +
-- char_length stored on the row. Storing it in Postgres duplicates the entire
-- document corpus inside the DB unnecessarily.
--
-- Idempotent: guarded by schema_migrations table.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '007_drop_chunk_text') THEN
        ALTER TABLE document_chunks DROP COLUMN IF EXISTS chunk_text;
        INSERT INTO schema_migrations (id) VALUES ('007_drop_chunk_text');
    END IF;
END $$;
