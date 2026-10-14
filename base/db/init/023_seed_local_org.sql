-- Seed the single-org local deployment. Idempotent.

INSERT INTO organizations (id, name, slug, created_at, updated_at)
VALUES ('org:dev', 'Local Context Graph', 'dev', now(), now())
ON CONFLICT (id) DO NOTHING;

INSERT INTO employees (id, org_id, email, canonical_name, role, status, auth_provider)
VALUES (
    'emp:local-service',
    'org:dev',
    'service@local',
    'local-service',
    'member',
    'active',
    'local'
)
ON CONFLICT (id) DO NOTHING;
