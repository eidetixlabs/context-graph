#!/bin/bash
# neo4j-entrypoint.sh — Custom Neo4j entrypoint that applies schema on first start.
#
# Wraps the standard Neo4j image entrypoint (/startup/docker-entrypoint.sh):
#   1. Starts Neo4j in background
#   2. Waits for bolt to accept connections
#   3. Applies every /neo4j-init/*.cypher file (base/db/neo4j/ mounted read-only)
#   4. Keeps Neo4j in foreground
#
# Used by deployment/docker/docker-compose.yml.
# Schema source of truth: base/db/neo4j/001_schema.cypher

set -e

# ── Resolve password ──────────────────────────────────────────────────────────
# Docker Compose injects the secret at /run/secrets/neo4j_password.
# Fall back to NEO4J_AUTH env var ("neo4j/<pass>") for compatibility.
if [[ -f /run/secrets/neo4j_password ]]; then
    NEO4J_PASS="$(cat /run/secrets/neo4j_password)"
else
    NEO4J_PASS="${NEO4J_AUTH#neo4j/}"
fi

# ── Start Neo4j in background ─────────────────────────────────────────────────
echo "[neo4j-entrypoint] Starting Neo4j..."
/startup/docker-entrypoint.sh neo4j &
NEO4J_PID=$!

# ── Wait for bolt ─────────────────────────────────────────────────────────────
echo "[neo4j-entrypoint] Waiting for bolt on localhost:7687..."
ELAPSED=0
MAX_WAIT=300
until cypher-shell -a "bolt://localhost:7687" \
                   -u neo4j -p "$NEO4J_PASS" \
                   "RETURN 1" > /dev/null 2>&1; do
    if ! kill -0 $NEO4J_PID 2>/dev/null; then
        echo "[neo4j-entrypoint] ERROR: Neo4j process exited unexpectedly."
        exit 1
    fi
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "[neo4j-entrypoint] ERROR: Neo4j not ready after ${MAX_WAIT}s."
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "[neo4j-entrypoint] Neo4j ready."

# ── Apply schema ──────────────────────────────────────────────────────────────
INIT_DIR="/neo4j-init"
if [[ -d "$INIT_DIR" ]]; then
    for f in $(ls "$INIT_DIR"/*.cypher 2>/dev/null | LC_ALL=C sort); do
        [[ -f "$f" ]] || continue
        echo "[neo4j-entrypoint] Applying $(basename "$f")..."
        cypher-shell -a "bolt://localhost:7687" \
                     -u neo4j -p "$NEO4J_PASS" \
                     --format plain \
                     --file "$f"
    done
    echo "[neo4j-entrypoint] Schema applied."
else
    echo "[neo4j-entrypoint] WARN: /neo4j-init not mounted — skipping schema init."
fi

# ── Keep Neo4j in foreground ──────────────────────────────────────────────────
wait $NEO4J_PID
