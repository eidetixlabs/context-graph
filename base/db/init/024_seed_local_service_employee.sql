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
