-- 022_context_graph_documents.sql — Repo-scoped portal context documents.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS repo TEXT,
    ADD COLUMN IF NOT EXISTS doc_type TEXT;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_source_provider_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_source_provider_check
    CHECK (source_provider IN (
        'notion', 'slack', 'confluence', 'github', 'jira',
        'google-meet', 'cursor-chat', 'claude-chat',
        'browser', 'manual', 'fireflies', 'gmail',
        'github-pdf', 'github-links', 'github-markdown',
        'github-structure', 'github-profile', 'portal'
    ));

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_doc_type_check;

ALTER TABLE documents
    ADD CONSTRAINT documents_doc_type_check
    CHECK (
        doc_type IS NULL
        OR doc_type IN (
            'architecture', 'hld', 'lld', 'design', 'roadmap',
            'ticket_retrospective', 'context'
        )
    );

CREATE INDEX IF NOT EXISTS idx_docs_org_repo
    ON documents(org_id, repo)
    WHERE repo IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_docs_org_repo_type
    ON documents(org_id, repo, doc_type)
    WHERE repo IS NOT NULL;

INSERT INTO schema_migrations (id) VALUES ('022_context_graph_documents')
ON CONFLICT (id) DO NOTHING;
