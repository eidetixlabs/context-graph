/**
 * Rollout/debugging hook log — ~/.context-graph/logs/claude-hooks.log
 *
 * Append-only, fail-open, never blocks Claude.
 * Low-volume: one line per hook invocation.
 * Format: <ISO> hook=<name> event=<event> duration_ms=<n> <detail>
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";

function logFilePath() {
  const home = process.env.USERPROFILE || process.env.HOME || os.homedir();
  return path.join(home, ".context-graph", "logs", "claude-hooks.log");
}

/**
 * @param {string} hookName
 * @param {string} detail  - short single-line outcome (e.g. "exit=0", "block", "fail-open")
 * @param {{ durationMs?: number, event?: string }} [meta]
 */
export function appendHookLog(hookName, detail, meta = {}) {
  try {
    const file = logFilePath();
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const ts = new Date().toISOString();
    const dur =
      meta.durationMs !== undefined ? ` duration_ms=${meta.durationMs}` : "";
    const ev = meta.event ? ` event=${meta.event}` : "";
    const line = `${ts} hook=${hookName}${ev}${dur} ${detail}\n`;
    fs.appendFileSync(file, line, "utf8");
  } catch {
    // fail-open: logging errors never affect hook exit code
  }
}

export { logFilePath };
