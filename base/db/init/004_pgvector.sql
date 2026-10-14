-- 004_pgvector.sql — pgvector extension and document chunk embeddings table.
--
-- Adds:
--   vector extension      — enables the vector column type and ANN index operators
--   document_chunks       — stores chunk offsets and 1024-dim embeddings for
--                           semantic (cosine-similarity) retrieval over documents
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.

-- ---------------------------------------------------------------------------
-- pgvector extension
-- Must be created before the vector column type can be used.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- document_chunks
-- One row per chunk of a document. Each chunk stores character offsets into
-- the source document and a 1024-dimensional embedding vector used for
-- nearest-neighbour retrieval. Raw text lives in S3 only.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    doc_id       TEXT NOT NULL REFERENCES documents(id),
    -- The parent document this chunk belongs to.

    chunk_index  INTEGER NOT NULL,
    -- Zero-based position of this chunk within the document.

    char_offset  INTEGER NOT NULL,
    -- Start character position of this chunk in the source document.

    char_length  INTEGER NOT NULL,
    -- Number of characters in this chunk.

    embedding    vector(1024) NOT NULL,
    -- 1024-dimensional embedding vector (e.g. from text-embedding-3-large or
    -- a comparable 1024-dim model). Used for cosine-similarity ANN search.

    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (doc_id, chunk_index)
    -- Each chunk position is unique within a document.
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc
    ON document_chunks(doc_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
-- IVFFlat with 100 lists gives good recall/speed trade-off for typical corpus
-- sizes. Re-index with more lists if the table grows beyond ~1M rows.

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('004_pgvector')
    ON CONFLICT (id) DO NOTHING;
