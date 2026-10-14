// 001_schema.cypher — Neo4j schema: all indexes for Eidetix.
//
// Single source of truth — consumed by every deployment target:
//   Helm (eidetix + eidetix-env)  : neo4j-schema-job.yaml Helm hook
//   Docker Compose (on-prem)      : deployment/docker/scripts/neo4j-entrypoint.sh
//   Docker standalone envs        : deployment/scripts/sync_neo4j.sh (docker exec mode)
//   Minikube                      : edicli minikube neo4j schema
//   EC2                           : deployment/scripts/sync_neo4j.sh (bolt URL mode)
//
// All statements are idempotent (IF NOT EXISTS).

// ── Vector indexes (384-dim cosine, MiniLM) ──────────────────────────────────
CREATE VECTOR INDEX decision_vector IF NOT EXISTS
  FOR (n:Decision) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX constraint_vector IF NOT EXISTS
  FOR (n:Constraint) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX insight_vector IF NOT EXISTS
  FOR (n:Insight) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX signal_vector IF NOT EXISTS
  FOR (n:Signal) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX preference_vector IF NOT EXISTS
  FOR (n:Preference) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX fact_vector IF NOT EXISTS
  FOR (n:Fact) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX entity_vector IF NOT EXISTS
  FOR (n:Entity) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX action_item_vector IF NOT EXISTS
  FOR (n:ActionItem) ON (n.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};

// ── Fulltext indexes ──────────────────────────────────────────────────────────
CREATE FULLTEXT INDEX semantic_fulltext IF NOT EXISTS
  FOR (n:Decision|Constraint|Insight|Signal)
  ON EACH [n.label, n.description, n.body, n.observation, n.rationale, n.trigger,
           n.alternatives, n.symptom, n.failure_without, n.appears_to_mean,
           n.actually_means, n.conditions, n.evidence];

CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
  FOR (n:Entity) ON EACH [n.canonical_name, n.label, n.aliases];

CREATE FULLTEXT INDEX preference_fulltext IF NOT EXISTS
  FOR (n:Preference) ON EACH [n.category, n.preference, n.context];

CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
  FOR (n:Fact) ON EACH [n.statement, n.category];

CREATE FULLTEXT INDEX action_item_fulltext IF NOT EXISTS
  FOR (n:ActionItem) ON EACH [n.label, n.status, n.source];

// ── Scalar indexes (superseded_by IS NULL filter) ────────────────────────────
CREATE INDEX idx_decision_current   IF NOT EXISTS FOR (n:Decision)   ON (n.superseded_by);
CREATE INDEX idx_constraint_current IF NOT EXISTS FOR (n:Constraint) ON (n.superseded_by);
CREATE INDEX idx_insight_current    IF NOT EXISTS FOR (n:Insight)    ON (n.superseded_by);
CREATE INDEX idx_signal_current     IF NOT EXISTS FOR (n:Signal)     ON (n.superseded_by);

// ── Entity scalar indexes ─────────────────────────────────────────────────────
CREATE INDEX idx_entity_type    IF NOT EXISTS FOR (n:Entity) ON (n.entity_type);
CREATE INDEX idx_entity_project IF NOT EXISTS FOR (n:Entity) ON (n.project);

// ── Preference / Fact scalar indexes ─────────────────────────────────────────
CREATE INDEX idx_preference_category    IF NOT EXISTS FOR (n:Preference) ON (n.category);
CREATE INDEX idx_preference_valid_until IF NOT EXISTS FOR (n:Preference) ON (n.valid_until);
CREATE INDEX idx_preference_entity_id   IF NOT EXISTS FOR (n:Preference) ON (n.entity_id);
CREATE INDEX idx_fact_category          IF NOT EXISTS FOR (n:Fact)       ON (n.category);
CREATE INDEX idx_fact_valid_until       IF NOT EXISTS FOR (n:Fact)       ON (n.valid_until);

// ── ActionItem scalar indexes ─────────────────────────────────────────────────
CREATE CONSTRAINT action_item_id IF NOT EXISTS FOR (n:ActionItem) REQUIRE n.id IS UNIQUE;
CREATE INDEX idx_action_item_status IF NOT EXISTS FOR (n:ActionItem) ON (n.status);
CREATE INDEX idx_action_item_due_date IF NOT EXISTS FOR (n:ActionItem) ON (n.due_date);
