#!/usr/bin/env node
/**
 * Eidetix — Stop hook
 *
 * Decision logic:
 *
 *   A. New entries since last checkpoint (semantics valid, len > last_checkpoint_len)
 *      → update session.json; clear scratch; allow stop (exit 0, no block).
 *
 *   B. No tools ran this turn (no tools-ran file)
 *      → allow stop (nothing to journal).
 *
 *   C. Already blocked once this turn (stop-blocked-once file exists)
 *      → allow stop (no infinite loop).
 *
 *   D. Tools ran, length unchanged, not yet blocked
 *      → write stop-blocked-once; emit block once asking for checkpoint.
 *
 * Invalid semantics.json: compareLen = 0; do NOT update last_checkpoint_len
 * until file is valid again.
 *
 * Fail-open: any I/O error → exit 0 (Claude continues).
 */

import fs from "node:fs";
import { readStdinJsonAsync, emitJson, warn, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";
import {
  projectSlugFromCwd,
  eidetixDir,
  sessionPath,
  semanticsPath,
  toolsRanPath,
  stopBlockedOncePath,
  ensureDir,
} from "./lib/paths.mjs";
import { readSemantics } from "./lib/semantics.mjs";
import { readSession, writeSession, updateCheckpointIfValid } from "./lib/session.mjs";

const BLOCK_REASON =
  "Eidetix checkpoint: this turn used tools. "
  + "If it produced a Decision, Constraint, Insight, Signal, or abandoned approach, "
  + "append up to 3 entries to semantics.json now. "
  + "If nothing is high-signal, make no changes — \"no signal\" is a valid outcome.";

runHook("Stop", () => {
  return readStdinJsonAsync().then((stdin) => {
    if (!stdin.ok) warn(`stdin parse failed: ${stdin.error}`);

    const input = stdin.data;
    const cwd = input.cwd || process.cwd();
    if (exitIfCwdNotAllowed("Stop", cwd)) return 0;
    const slug = projectSlugFromCwd(cwd);
    ensureDir(eidetixDir(slug));

    const sem = readSemantics(semanticsPath(slug));
    if (sem.warning) warn(sem.warning);

    const compareLen = sem.length; // 0 if invalid (per contract)
    const { session } = readSession(sessionPath(slug));

    const toolsRan = fs.existsSync(toolsRanPath(slug));
    const alreadyBlocked = fs.existsSync(stopBlockedOncePath(slug));

    // Path A: new valid entries since last checkpoint
    if (sem.valid && compareLen > session.last_checkpoint_len) {
      const updated = updateCheckpointIfValid(sessionPath(slug), session, compareLen, true);
      writeSession(sessionPath(slug), updated);
      for (const p of [toolsRanPath(slug), stopBlockedOncePath(slug)]) {
        try {
          if (fs.existsSync(p)) fs.unlinkSync(p);
        } catch (e) {
          warn(`clear scratch: ${e.message}`);
        }
      }
      warn(`Stop: checkpoint advanced to len=${compareLen}`);
      return 0;
    }

    // Path B: no tools ran — nothing to prompt about
    if (!toolsRan) {
      warn("Stop: no tools-ran; allowing stop");
      return 0;
    }

    // Path C: already blocked once this turn — never loop
    if (alreadyBlocked) {
      warn("Stop: stop-blocked-once set; allowing stop");
      return 0;
    }

    // Path D: single intentional block
    try {
      fs.writeFileSync(stopBlockedOncePath(slug), "", "utf8");
    } catch (e) {
      // If we can't write the guard file we must not block — infinite loop risk
      warn(`cannot write stop-blocked-once (fail-open, not blocking): ${e.message}`);
      return 0;
    }

    warn("Stop: blocking once for semantic checkpoint");
    emitJson({
      decision: "block",
      reason: BLOCK_REASON,
    });
    return 0;
  });
});
