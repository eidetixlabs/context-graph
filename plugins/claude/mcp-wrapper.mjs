#!/usr/bin/env node
/**
 * Stdio bridge for Claude Desktop → local Context Graph MCP (HTTP SSE).
 *
 * Reads URL + API key from, in order:
 *   CONTEXT_GRAPH_URL / CONTEXT_GRAPH_API_KEY
 *   ~/.context-graph/local.json  { "url": "http://localhost:8080", "api_key": "..." }
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const home = process.env.USERPROFILE || process.env.HOME || os.homedir();
const configPath = path.join(home, ".context-graph", "local.json");
const npxBin = process.argv[2] || "npx";

function readLocalConfig() {
  try {
    return JSON.parse(fs.readFileSync(configPath, "utf8"));
  } catch {
    return {};
  }
}

const local = readLocalConfig();
const url = (process.env.CONTEXT_GRAPH_URL || local.url || "http://localhost:8080").replace(/\/+$/, "");
const apiKey = process.env.CONTEXT_GRAPH_API_KEY || local.api_key || "";

if (!apiKey) {
  process.stderr.write(
    "context-graph: missing API key. Run ./scripts/bootstrap.sh or set CONTEXT_GRAPH_API_KEY.\n",
  );
  process.exit(1);
}

const mcpUrl = `${url}/mcp/sse`;
const child = spawn(
  npxBin,
  [
    "-y",
    "mcp-remote",
    mcpUrl,
    "--header",
    `Authorization: Bearer ${apiKey}`,
  ],
  { stdio: "inherit" },
);

child.on("error", (error) => {
  process.stderr.write(`context-graph: failed to start mcp-remote: ${error.message}\n`);
  process.exit(1);
});
child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
