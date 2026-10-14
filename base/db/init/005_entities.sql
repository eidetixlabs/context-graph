-- 005_entities.sql — Canonical entity registry and alias lookup tables.
--
-- Adds:
--   entity_registry  — canonical entities (person/service/tool/etc.)
--   entity_aliases   — org-scoped alias lookup table with unique normalized alias
--   documents.author_entity_id — optional entity-level author linkage
--
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------

-- Deterministic normalization used by alias write paths.
CREATE OR REPLACE FUNCTION normalize_entity_alias(raw_alias TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT lower(regexp_replace(trim(COALESCE(raw_alias, '')), '[^a-z0-9 ]', '', 'gi'));
$$;

-- ---------------------------------------------------------------------------
-- entity_registry
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_registry (
    id              TEXT PRIMARY KEY,
    -- Format: 'entity:{type}:{slug}'

    org_id          TEXT NOT NULL REFERENCES organizations(id),

    entity_type     TEXT NOT NULL
                    CHECK (entity_type IN (
                        'person', 'service', 'organization', 'repository', 'tool',
                        'object', 'location', 'event'
                    )),

    canonical_name  TEXT NOT NULL,

    employee_id     TEXT REFERENCES employees(id) ON DELETE SET NULL,
    -- NULL for non-person entities or unresolved external identities.
    -- ON DELETE SET NULL: removing a placeholder employee orphans the entity
    -- (preserves graph history) rather than cascading a delete.

    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'merged', 'deleted')),

    merged_into     TEXT REFERENCES entity_registry(id),
    -- NULL unless status='merged'.

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entity_org
    ON entity_registry(org_id);

CREATE INDEX IF NOT EXISTS idx_entity_emp
    ON entity_registry(employee_id)
    WHERE employee_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_entity_merged
    ON entity_registry(merged_into)
    WHERE merged_into IS NOT NULL;

-- ---------------------------------------------------------------------------
-- entity_aliases
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_normalized TEXT NOT NULL,
    -- Lowercased, punctuation-stripped alias.

    entity_id        TEXT NOT NULL REFERENCES entity_registry(id) ON DELETE CASCADE,
    org_id           TEXT NOT NULL REFERENCES organizations(id),

    alias_raw        TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'manual'
                     CHECK (source IN (
                        'onboarding',
                        'idp-sync',
                        'extraction',
                        'manual',
                        'merge'
                     )),

    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (org_id, alias_normalized)
);

CREATE INDEX IF NOT EXISTS idx_alias_entity
    ON entity_aliases(entity_id);

CREATE INDEX IF NOT EXISTS idx_alias_org_entity
    ON entity_aliases(org_id, entity_id);

-- ---------------------------------------------------------------------------
-- identity_mappings
-- Maps external system user IDs to internal employee IDs.
-- Populated by IDP sync, seeding, and the identity resolution helper.
-- org_id is part of the PK so the same external_id on different providers
-- can map to different employees in different orgs without collision.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS identity_mappings (
    org_id          TEXT NOT NULL REFERENCES organizations(id),

    employee_id     TEXT NOT NULL REFERENCES employees(id),

    provider        TEXT NOT NULL
                    CHECK (provider IN (
                        'slack', 'confluence', 'github', 'jira',
                        'google-meet', 'notion', 'linear', 'fireflies'
                    )),

    external_id     TEXT NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (org_id, provider, external_id)
    -- Within one org, a given external ID on a given provider maps to exactly one employee.
);

CREATE INDEX IF NOT EXISTS idx_identity_employee
    ON identity_mappings(employee_id);

-- ---------------------------------------------------------------------------
-- documents author entity linkage
-- ---------------------------------------------------------------------------

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS author_entity_id TEXT REFERENCES entity_registry(id);

CREATE INDEX IF NOT EXISTS idx_docs_author_entity
    ON documents(author_entity_id)
    WHERE author_entity_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Record this migration
-- ---------------------------------------------------------------------------

INSERT INTO schema_migrations (id) VALUES ('005_entities')
    ON CONFLICT (id) DO NOTHING;
