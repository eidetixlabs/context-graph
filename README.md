# Context Graph

Local long-term memory for coding agents. It stores decisions, constraints, insights, and signals in a knowledge graph and exposes them over MCP so Cursor, Claude, and other agents can **ingest** and **recall** as they work.

## Prerequisites

- Docker Engine 24+ with Compose v2
- About 8 GB RAM for Neo4j + Postgres + the processors
- An OpenAI API key (embeddings + extraction). Optional: local MiniLM via `docker compose --profile inference up`

## Start the stack

```bash
cp .env.example .env
# paste OPENAI_API_KEY=sk-...
./scripts/bootstrap.sh          # writes API_KEY and prints MCP config
docker compose up --build
```

MCP is at `http://localhost:8080/mcp/sse`. Health: `curl http://localhost:8080/health`.

## Connect an agent

Cursor — `~/.cursor/mcp.json` (bootstrap prints this with your key):

```json
{
  "mcpServers": {
    "context-graph": {
      "url": "http://localhost:8080/mcp/sse",
      "headers": {
        "Authorization": "Bearer <API_KEY>"
      }
    }
  }
}
```

Claude Desktop — stdio via `mcp-remote` (Node.js required):

```json
{
  "mcpServers": {
    "context-graph": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "http://localhost:8080/mcp/sse",
        "--header",
        "Authorization: Bearer <API_KEY>"
      ]
    }
  }
}
```

Optional Cursor/Claude extraction rules live in `plugins/`.

## Ingest, then recall

Index a folder of markdown/text:

```bash
python3 -m tools.ingest_folder ./examples --api-key "$API_KEY"
```

Agents can also call MCP `ingest_document` (notes/docs) and `ingest_traces` (Decision / Constraint / Insight / Signal). After the `processor-portal` / `processor-trace` workers run, use:

1. `recall` — ranked documents + graph hits
2. `get_graph` — neighbors of promising node ids
3. `get_content` — original source text

License: Apache-2.0. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
