"""
base/db/neo4j_indexes.py — Ensure Neo4j vector, fulltext, and scalar indexes exist at startup.
"""

from __future__ import annotations

from base.neo4j_client import Neo4jClient

_VECTOR_INDEXES = [
    (
        "decision_vector",
        "Decision",
        "CREATE VECTOR INDEX decision_vector IF NOT EXISTS "
        "FOR (n:Decision) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "constraint_vector",
        "Constraint",
        "CREATE VECTOR INDEX constraint_vector IF NOT EXISTS "
        "FOR (n:Constraint) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "insight_vector",
        "Insight",
        "CREATE VECTOR INDEX insight_vector IF NOT EXISTS "
        "FOR (n:Insight) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "signal_vector",
        "Signal",
        "CREATE VECTOR INDEX signal_vector IF NOT EXISTS "
        "FOR (n:Signal) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "preference_vector",
        "Preference",
        "CREATE VECTOR INDEX preference_vector IF NOT EXISTS "
        "FOR (n:Preference) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "fact_vector",
        "Fact",
        "CREATE VECTOR INDEX fact_vector IF NOT EXISTS "
        "FOR (n:Fact) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "entity_vector",
        "Entity",
        "CREATE VECTOR INDEX entity_vector IF NOT EXISTS "
        "FOR (n:Entity) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
    (
        "action_item_vector",
        "ActionItem",
        "CREATE VECTOR INDEX action_item_vector IF NOT EXISTS "
        "FOR (n:ActionItem) ON (n.embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
    ),
]

_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX semantic_fulltext IF NOT EXISTS "
    "FOR (n:Decision|Constraint|Insight|Signal) ON EACH [n.label, n.description, n.body, "
    "n.observation, n.rationale, n.trigger, n.alternatives, n.symptom, n.failure_without, "
    "n.appears_to_mean, n.actually_means, n.conditions, n.evidence]"
)

_ENTITY_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
    "FOR (n:Entity) ON EACH [n.canonical_name, n.label, n.aliases]"
)

# Scalar indexes on superseded_by so IS NULL filter is fast
_SUPERSEDED_BY_INDEXES = [
    "CREATE INDEX idx_decision_current   IF NOT EXISTS FOR (n:Decision)   ON (n.superseded_by)",
    "CREATE INDEX idx_constraint_current IF NOT EXISTS FOR (n:Constraint) ON (n.superseded_by)",
    "CREATE INDEX idx_insight_current    IF NOT EXISTS FOR (n:Insight)    ON (n.superseded_by)",
    "CREATE INDEX idx_signal_current     IF NOT EXISTS FOR (n:Signal)     ON (n.superseded_by)",
]

# Scalar index on Entity.entity_type so filtering by type (person/service/object/etc.) is fast
_ENTITY_TYPE_INDEX = (
    "CREATE INDEX idx_entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)"
)

# Scalar index on Entity.project (ON CREATE SET — fast lookup per project)
_ENTITY_PROJECT_INDEX = (
    "CREATE INDEX idx_entity_project IF NOT EXISTS FOR (n:Entity) ON (n.project)"
)

_PREFERENCE_FACT_SCALAR_INDEXES = [
    "CREATE INDEX idx_preference_category   IF NOT EXISTS FOR (n:Preference) ON (n.category)",
    "CREATE INDEX idx_preference_valid_until IF NOT EXISTS FOR (n:Preference) ON (n.valid_until)",
    "CREATE INDEX idx_preference_entity_id  IF NOT EXISTS FOR (n:Preference) ON (n.entity_id)",
    "CREATE INDEX idx_fact_category         IF NOT EXISTS FOR (n:Fact)       ON (n.category)",
    "CREATE INDEX idx_fact_valid_until      IF NOT EXISTS FOR (n:Fact)       ON (n.valid_until)",
]

_PREFERENCE_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX preference_fulltext IF NOT EXISTS "
    "FOR (n:Preference) ON EACH [n.category, n.preference, n.context]"
)

_FACT_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS "
    "FOR (n:Fact) ON EACH [n.statement, n.category]"
)

_ACTION_ITEM_FULLTEXT_INDEX = (
    "CREATE FULLTEXT INDEX action_item_fulltext IF NOT EXISTS "
    "FOR (n:ActionItem) ON EACH [n.label, n.status, n.source]"
)

_ACTION_ITEM_SCALAR_INDEXES = [
    "CREATE CONSTRAINT action_item_id IF NOT EXISTS FOR (n:ActionItem) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX idx_action_item_status   IF NOT EXISTS FOR (n:ActionItem) ON (n.status)",
    "CREATE INDEX idx_action_item_due_date IF NOT EXISTS FOR (n:ActionItem) ON (n.due_date)",
]


def ensure_vector_indexes(client: Neo4jClient) -> None:
    """Create all vector, fulltext, and scalar indexes (idempotent — uses IF NOT EXISTS)."""
    for _name, _label, cypher in _VECTOR_INDEXES:
        client.run_write(cypher)
    client.run_write(_FULLTEXT_INDEX)
    client.run_write(_ENTITY_FULLTEXT_INDEX)
    for cypher in _SUPERSEDED_BY_INDEXES:
        client.run_write(cypher)
    client.run_write(_ENTITY_TYPE_INDEX)
    client.run_write(_ENTITY_PROJECT_INDEX)
    for cypher in _PREFERENCE_FACT_SCALAR_INDEXES:
        client.run_write(cypher)
    client.run_write(_PREFERENCE_FULLTEXT_INDEX)
    client.run_write(_FACT_FULLTEXT_INDEX)
    client.run_write(_ACTION_ITEM_FULLTEXT_INDEX)
    for cypher in _ACTION_ITEM_SCALAR_INDEXES:
        client.run_write(cypher)
