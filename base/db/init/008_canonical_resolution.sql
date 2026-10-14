-- 008_canonical_resolution.sql — canonical entity resolution tables.
--
-- Adds:
--   document_resolution_maps — one resolution map per (doc_id, doc_version)
--   entity_metadata          — key/value metadata per entity (role, email, org, etc.)
--
-- All statements are idempotent (IF NOT EXISTS / DO NOTHING) — safe to re-run.

-- ---------------------------------------------------------------------------
-- document_resolution_maps
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS document_resolution_maps (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    doc_id          TEXT NOT NULL REFERENCES documents(id),
    doc_version     INTEGER NOT NULL DEFAULT 1,

    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'resolved', 'partial', 'failed')),

    entries         JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- mention_text -> ResolutionMapEntry dict

    resolved_entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- list of ResolvedEntity dicts

    audit           JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- list of audit event dicts {mention, reason, timestamp}

    counts          JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- {total, resolved, new, pronoun, failed}

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (doc_id, doc_version)
);

CREATE INDEX IF NOT EXISTS idx_resolution_maps_doc
    ON document_resolution_maps(doc_id);

CREATE INDEX IF NOT EXISTS idx_resolution_maps_status
    ON document_resolution_maps(status);

-- ---------------------------------------------------------------------------
-- entity_metadata
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_metadata (
    entity_id   TEXT NOT NULL REFERENCES entity_registry(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'extraction'
                CHECK (source IN ('connector', 'extraction', 'manual')),

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (entity_id, key)
);

CREATE INDEX IF NOT EXISTS idx_entity_metadata_email
    ON entity_metadata(value)
    WHERE key = 'email';

CREATE INDEX IF NOT EXISTS idx_entity_metadata_role
    ON entity_metadata(entity_id, value)
    WHERE key = 'role';

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('008_canonical_resolution')
    ON CONFLICT (id) DO NOTHING;
