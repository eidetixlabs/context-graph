#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

ensure_key() {
  python3 - <<'PY'
from pathlib import Path
import re, secrets

path = Path(".env")
text = path.read_text() if path.exists() else ""
match = re.search(r"^API_KEY=(.*)$", text, re.M)
current = match.group(1).strip() if match else ""
if current:
    print(current)
    raise SystemExit(0)
key = secrets.token_urlsafe(32)
if match:
    text = re.sub(r"^API_KEY=.*$", f"API_KEY={key}", text, count=1, flags=re.M)
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += f"API_KEY={key}\n"
path.write_text(text)
print(key)
PY
}

API_KEY="$(ensure_key)"

mkdir -p "${HOME}/.context-graph"
python3 - <<PY
import json
from pathlib import Path
path = Path.home() / ".context-graph" / "local.json"
path.write_text(json.dumps({
    "url": "http://localhost:8080",
    "api_key": "${API_KEY}",
}, indent=2) + "\n")
PY

cat <<EOF

Context Graph MCP config (Cursor ~/.cursor/mcp.json):

{
  "mcpServers": {
    "context-graph": {
      "url": "http://localhost:8080/mcp/sse",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}

Claude Desktop can use plugins/claude/mcp-wrapper.mjs, or mcp-remote with the
same Bearer token. Wrote ~/.context-graph/local.json

Next:
  docker compose up --build
  python3 -m tools.ingest_folder ./examples --api-key "${API_KEY}"

EOF
