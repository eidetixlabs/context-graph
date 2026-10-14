-- 006_entity_types.sql — Extend entity_type CHECK constraint to include object/location/event.
--
-- The original constraint in 005_entities.sql only allowed:
--   person, service, organization, repository, tool
--
-- This migration drops and recreates the constraint to add:
--   object, location, event
--
-- Idempotent: guarded by schema_migrations table.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE id = '006_entity_types') THEN

        -- Drop the old constraint by name (Postgres auto-names it entity_registry_entity_type_check)
        ALTER TABLE entity_registry
            DROP CONSTRAINT IF EXISTS entity_registry_entity_type_check;

        -- Recreate with full type list
        ALTER TABLE entity_registry
            ADD CONSTRAINT entity_registry_entity_type_check
            CHECK (entity_type IN (
                'person', 'service', 'organization', 'repository', 'tool',
                'object', 'location', 'event'
            ));

        INSERT INTO schema_migrations (id) VALUES ('006_entity_types');

    END IF;
END
$$;
