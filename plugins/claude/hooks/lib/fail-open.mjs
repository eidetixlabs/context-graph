/**
 * Fail-open scaffolding for Eidetix Claude hooks.
 *
 * All hooks MUST exit 0 on error (Claude continues).
 * The only intentional non-zero-ish path is the Stop block JSON on stdout —
 * which still uses exit 0 with a `decision: block` payload.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { appendHookLog } from "./hook-log.mjs";
import { isCwdAllowed } from "./paths.mjs";

const PLUGIN_SETTINGS_KEY = "eidetix@eidetix";

function isPluginDisabledInClaudeSettings() {
  try {
    const home = process.env.USERPROFILE || process.env.HOME || os.homedir();
    const configDir = process.env.CLAUDE_CONFIG_DIR || path.join(home, ".claude");
    const settingsPath = path.join(configDir, "settings.json");
    if (!fs.existsSync(settingsPath)) return false;
    const settings = JSON.parse(fs.readFileSync(settingsPath, "utf8"));
    return settings?.enabledPlugins?.[PLUGIN_SETTINGS_KEY] === false;
  } catch {
    return false;
  }
}

/** Read full stdin synchronously and parse JSON. Used by non-Stop hooks. */
export function readStdinJson() {
  try {
    const buf = fs.readFileSync(0, "utf8");
    if (!buf || !buf.trim()) return { ok: true, data: {} };
    return { ok: true, data: JSON.parse(buf) };
  } catch (err) {
    return { ok: false, error: err.message, data: {} };
  }
}

/** Read stdin asynchronously and parse JSON. Preferred for all hooks. */
export async function readStdinJsonAsync() {
  return new Promise((resolve) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => {
      try {
        if (!data.trim()) resolve({ ok: true, data: {} });
        else resolve({ ok: true, data: JSON.parse(data) });
      } catch (err) {
        resolve({ ok: false, error: err.message, data: {} });
      }
    });
    process.stdin.on("error", (err) => {
      resolve({ ok: false, error: err.message, data: {} });
    });
  });
}

/** Write JSON to stdout (hook response channel). */
export function emitJson(obj) {
  process.stdout.write(JSON.stringify(obj));
}

/** Write a warning to stderr. Never throws. */
export function warn(msg) {
  try {
    process.stderr.write(`eidetix: ${msg}\n`);
  } catch {
    // ignore
  }
}

/**
 * Wrap a hook body function.
 * - Measures duration; logs to claude-hooks.log; warns if >250ms.
 * - Any thrown error → fail-open exit 0.
 * - fn() may return a numeric exit code (0) or undefined.
 */
export function runHook(name, fn) {
  if (isPluginDisabledInClaudeSettings()) {
    process.exit(0);
  }

  const started = Date.now();
  Promise.resolve()
    .then(() => fn())
    .then((code) => {
      const durationMs = Date.now() - started;
      appendHookLog(name, `exit=${code ?? 0}`, { durationMs });
      if (durationMs > 250) {
        warn(`slow hook ${name} ${durationMs}ms (target <250ms except intentional Stop block)`);
      }
      process.exit(typeof code === "number" ? code : 0);
    })
    .catch((err) => {
      const durationMs = Date.now() - started;
      appendHookLog(name, `fail-open error=${err.message}`, { durationMs });
      warn(`${name} error (fail-open): ${err.message}`);
      process.exit(0);
    });
}

export function exitIfCwdNotAllowed(hookName, cwd) {
  if (isCwdAllowed(cwd)) return false;
  appendHookLog(hookName, "skip disallowed cwd", { cwd });
  process.exit(0);
}
