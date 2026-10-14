#!/usr/bin/env bash
# Compose smoke: health, document ingest, trace ingest, optional recall.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

URL="${CONTEXT_GRAPH_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-${CONTEXT_GRAPH_API_KEY:-dev-local-key}}"
AUTH="Authorization: Bearer ${API_KEY}"

echo "Waiting for ${URL}/health ..."
for i in $(seq 1 60); do
  if curl -sf "${URL}/health" >/dev/null; then
    break
  fi
  if [[ "$i" -eq 60 ]]; then
    echo "health check timed out" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "${URL}/health" | grep -q '"ok"'

INIT=$(curl -sf -X POST "${URL}/mcp/sse" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}')
echo "$INIT" | grep -q 'context-graph'

DOC=$(curl -sf -X POST "${URL}/ingest/documents" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"title":"Smoke architecture","content":"We store sessions in Postgres, not Redis.","doc_type":"architecture","source_ref":"smoke-architecture"}')
echo "$DOC" | grep -q 'doc:portal'

TRACE=$(curl -sf -X POST "${URL}/ingest/traces" \
  -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"traces":[{"trace_id":"smoke-trace-1","ts":"2026-10-14T00:00:00Z","node":{"type":"Decision","label":"Postgres for sessions","body":"Avoid a separate Redis failover domain."}}]}')
echo "$TRACE" | grep -q '"ok"'

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY set — waiting for processor then recall"
  HIT=0
  for i in $(seq 1 40); do
    RECALL=$(curl -sf -X POST "${URL}/mcp/sse" \
      -H "$AUTH" -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall","arguments":{"query":"Postgres sessions"}}}')
    if echo "$RECALL" | grep -Eqi 'postgres|session|Decision'; then
      HIT=1
      break
    fi
    sleep 5
  done
  if [[ "$HIT" -ne 1 ]]; then
    echo "recall did not return a citation after waiting" >&2
    echo "$RECALL" >&2
    exit 1
  fi
else
  echo "OPENAI_API_KEY unset — skipped processor/recall assertion"
fi

echo "smoke ok"
