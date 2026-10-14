#!/usr/bin/env node
/**
 * Eidetix — SessionStart hook
 *
 * Responsibilities:
 *   1. Init (or re-init on resume) session.json with current session_id.
 *   2. Clear scratch files (tools-ran, stop-blocked-once).
 *   3. Emit additionalContext encouraging one MCP recall at session start.
 *
 * Fail-open: exit 0 on any error; Claude is never blocked.
 * Invalid semantics.json: preserve prior last_checkpoint_len; do not advance.
 */

import fs from "node:fs";
import { readStdinJsonAsync, emitJson, warn, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";
import {
  projectSlugFromCwd,
  projectDir,
  eidetixDir,
  sessionPath,
  semanticsPath,
  toolsRanPath,
  stopBlockedOncePath,
  ensureDir,
} from "./lib/paths.mjs";
import { readSemantics } from "./lib/semantics.mjs";
import { readSession, writeSession } from "./lib/session.mjs";

runHook("SessionStart", () => {
  return readStdinJsonAsync().then((stdin) => {
    if (!stdin.ok) warn(`stdin parse failed: ${stdin.error}`);

    const input = stdin.data;
    const cwd = input.cwd || process.cwd();
    if (exitIfCwdNotAllowed("SessionStart", cwd)) return 0;
    const slug = projectSlugFromCwd(cwd);
    const sessionId = input.session_id || "unknown-session";

    ensureDir(projectDir(slug));
    ensureDir(eidetixDir(slug));

    const sem = readSemantics(semanticsPath(slug));
    if (sem.warning) warn(sem.warning);

    // If semantics is valid, snapshot current length as checkpoint baseline.
    // If invalid, carry forward the last known good checkpoint to avoid resetting.
    let checkpointLen = sem.valid ? sem.length : 0;
    let checkpointAt = new Date().toISOString();

    if (!sem.valid) {
      const { session: prev, existed, corrupted } = readSession(sessionPath(slug));
      if (existed && !corrupted) {
        checkpointLen = prev.last_checkpoint_len;
        checkpointAt = prev.last_checkpoint_at;
      }
    }

    writeSession(sessionPath(slug), {
      session_id: sessionId,
      last_checkpoint_len: checkpointLen,
      last_checkpoint_at: checkpointAt,
    });

    // Clear scratch files from any previous session
    for (const p of [toolsRanPath(slug), stopBlockedOncePath(slug)]) {
      try {
        if (fs.existsSync(p)) fs.unlinkSync(p);
      } catch (e) {
        warn(`clear scratch ${p}: ${e.message}`);
      }
    }

    emitJson({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: `Eidetix context graph is active for project "${slug}". `
          + `At session start, run one MCP recall using search(query) `
          + `if the task is substantive — do not recall every turn. `
          + `Session id: ${sessionId}. `
          + `Journal high-signal semantics (Decisions, Constraints, Insights, Signals) incrementally; skip low-signal noise.`,
      },
    });
    return 0;
  });
});
