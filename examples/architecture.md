# Sample architecture

We store session state in Postgres, not Redis.

That decision was shaped by operational overhead: a single-node Redis cache
would have been another failover domain for a product that already had
Postgres. Session rows live in the `sessions` table with a hashed bearer
token and an expiry timestamp.

Constraint: Community Neo4j is a single database named `neo4j`. Do not
assume per-tenant Neo4j databases in local mode.

Insight: agents should call `recall` before asking a human about prior
decisions. The graph is the team's memory; the chat is not.
