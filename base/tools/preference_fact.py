"""
base/tools/preference_fact.py — Cypher templates for Preference and Fact CRUD.

Temporal values (valid_from, valid_until) are stored as Neo4j datetime() types.
Python callers pass neo4j.time.DateTime objects as $as_of_dt — no wrapping needed.
"""

from __future__ import annotations

from typing import Any

from base.neo4j_client import Neo4jClient


# ---------------------------------------------------------------------------
# Preference Cypher templates
# ---------------------------------------------------------------------------

CREATE_PREFERENCE = """
MERGE (p:Preference {id: $id})
SET p.category    = $category,
    p.preference  = $preference,
    p.confidence  = $confidence,
    p.context     = $context,
    p.entity_id   = $entity_id,
    p.valid_from  = CASE WHEN p.valid_from IS NULL THEN coalesce($valid_from, datetime()) ELSE p.valid_from END,
    p.valid_until = CASE WHEN $valid_until IS NOT NULL THEN $valid_until ELSE p.valid_until END,
    p.created_at  = CASE WHEN p.created_at IS NULL THEN datetime() ELSE p.created_at END
"""

LINK_ENTITY_TO_PREFERENCE = """
MERGE (e:Entity {id: $entity_id})-[:HAS_PREFERENCE]->(p:Preference {id: $preference_id})
"""

LINK_PREFERENCE_TO_ENTITY = """
MERGE (p:Preference {id: $preference_id})-[:APPLIES_TO]->(e:Entity {id: $entity_id})
"""

SUPERSEDE_PREFERENCE = """
MATCH (old:Preference {id: $old_id})
SET old.valid_until = datetime()
"""

SEARCH_PREFERENCES_BY_EMBEDDING = """
CALL db.index.vector.queryNodes('preference_vector', $k, $embedding)
YIELD node AS p, score
WHERE score >= $threshold
RETURN properties(p) AS props, score
ORDER BY score DESC
"""

GET_PREFERENCES_FOR_ENTITY = """
MATCH (e:Entity {id: $entity_id})-[:HAS_PREFERENCE]->(p:Preference)
WHERE CASE
  WHEN $as_of_dt IS NOT NULL THEN
    (p.valid_from IS NULL OR p.valid_from <= $as_of_dt)
    AND (p.valid_until IS NULL OR p.valid_until > $as_of_dt)
  WHEN $active_only = true THEN
    p.valid_until IS NULL
  ELSE true
END
RETURN properties(p) AS props
ORDER BY p.created_at DESC
"""


# ---------------------------------------------------------------------------
# Fact Cypher templates
# ---------------------------------------------------------------------------

CREATE_FACT = """
MERGE (f:Fact {id: $id})
SET f.subject       = $subject,
    f.predicate     = $predicate,
    f.object        = $object,
    f.confidence    = $confidence,
    f.source_doc_id = $source_doc_id,
    f.valid_from    = CASE WHEN f.valid_from IS NULL THEN $valid_from ELSE f.valid_from END,
    f.valid_until   = CASE WHEN f.valid_until IS NULL THEN $valid_until ELSE f.valid_until END,
    f.created_at    = CASE WHEN f.created_at IS NULL THEN datetime() ELSE f.created_at END
"""

LINK_FACT_TO_ENTITY = """
MERGE (f:Fact {id: $fact_id})-[:ABOUT]->(e:Entity {id: $entity_id})
"""

SEARCH_FACTS_BY_EMBEDDING = """
CALL db.index.vector.queryNodes('fact_vector', $k, $embedding)
YIELD node AS f, score
WHERE score >= $threshold
RETURN properties(f) AS props, score
ORDER BY score DESC
"""

GET_FACTS_BY_SUBJECT = """
MATCH (f:Fact)
WHERE f.subject = $subject
  AND ($as_of_dt IS NULL
       OR ((f.valid_from IS NULL OR f.valid_from <= $as_of_dt)
           AND (f.valid_until IS NULL OR f.valid_until > $as_of_dt)))
RETURN properties(f) AS props
ORDER BY f.created_at DESC
"""

FIND_DUPLICATE_FACTS = """
CALL db.index.vector.queryNodes('fact_vector', $k, $embedding)
YIELD node AS f, score
WHERE score >= $threshold AND f.subject = $subject AND f.predicate = $predicate
RETURN properties(f) AS props, score
ORDER BY score DESC
"""

EXPIRE_FACT = """
MATCH (f:Fact {id: $fact_id})
SET f.valid_until = datetime()
"""


# ---------------------------------------------------------------------------
# Temporal filter helper
# ---------------------------------------------------------------------------

def as_of_filter(alias: str = "p") -> str:
    """Return a Cypher WHERE clause fragment for as_of temporal filtering.

    Values are already stored as Neo4j datetime — no wrapping needed.
    Caller must bind $as_of_dt as a neo4j.time.DateTime object.

    Usage:
        cypher = f"MATCH (p:Preference) WHERE {as_of_filter('p')} RETURN p"
        client.run(cypher, {"as_of_dt": neo4j_datetime_obj})
    """
    return (
        f"({alias}.valid_from IS NULL OR {alias}.valid_from <= $as_of_dt) "
        f"AND ({alias}.valid_until IS NULL OR {alias}.valid_until > $as_of_dt)"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_preferences_for_entity(
    client: Neo4jClient,
    entity_id: str,
    active_only: bool = True,
    as_of_dt=None,
) -> list[dict]:
    """Fetch preferences for a person entity.

    Args:
        entity_id: ID of the (:Entity {entity_type:"person"}) node.
        active_only: If True, only return preferences where valid_until IS NULL.
        as_of_dt: neo4j.time.DateTime for point-in-time query. If as_of_dt is
            provided, it overrides active_only and returns the full temporal
            snapshot at that point.
    """
    rows = client.run(GET_PREFERENCES_FOR_ENTITY, {
        "entity_id": entity_id,
        "active_only": active_only,
        "as_of_dt": as_of_dt,
    })
    return [r["props"] for r in rows]


def search_preferences_by_embedding(
    client: Neo4jClient,
    embedding: list[float],
    k: int = 10,
    threshold: float = 0.7,
) -> list[dict]:
    rows = client.run(SEARCH_PREFERENCES_BY_EMBEDDING, {
        "embedding": embedding,
        "k": k,
        "threshold": threshold,
    })
    return [{"score": r["score"], **r["props"]} for r in rows]


def search_facts_by_embedding(
    client: Neo4jClient,
    embedding: list[float],
    k: int = 10,
    threshold: float = 0.7,
) -> list[dict]:
    rows = client.run(SEARCH_FACTS_BY_EMBEDDING, {
        "embedding": embedding,
        "k": k,
        "threshold": threshold,
    })
    return [{"score": r["score"], **r["props"]} for r in rows]


def get_facts_by_subject(
    client: Neo4jClient,
    subject: str,
    as_of_dt=None,
) -> list[dict]:
    rows = client.run(GET_FACTS_BY_SUBJECT, {
        "subject": subject,
        "as_of_dt": as_of_dt,
    })
    return [r["props"] for r in rows]


def supersede_preference(client: Neo4jClient, old_id: str) -> None:
    """Set valid_until = datetime() on the given Preference node."""
    client.run_write(SUPERSEDE_PREFERENCE, {"old_id": old_id})


def expire_fact(client: Neo4jClient, fact_id: str) -> None:
    """Set valid_until = datetime() on the given Fact node."""
    client.run_write(EXPIRE_FACT, {"fact_id": fact_id})
