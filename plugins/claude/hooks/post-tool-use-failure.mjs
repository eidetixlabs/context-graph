#!/usr/bin/env node
/**
 * Eidetix — PostToolUseFailure hook
 *
 * Responsibilities:
 *   1. Touch tools-ran (same as PostToolUse — failure is tool activity).
 *   2. Emit a short additionalContext nudge: tool failures often reveal Constraints or Signals.
 *
 * The nudge is precision-focused — it asks Claude to consider whether the failure
 * is worth capturing, not to capture it unconditionally.
 *
 * Fail-open: exit 0 on any error.
 */

import fs from "node:fs";
import { readStdinJsonAsync, emitJson, warn, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";
import {
  projectSlugFromCwd,
  eidetixDir,
  toolsRanPath,
  ensureDir,
} from "./lib/paths.mjs";

runHook("PostToolUseFailure", () => {
  return readStdinJsonAsync().then((stdin) => {
    if (!stdin.ok) warn(`stdin parse failed: ${stdin.error}`);
    const input = stdin.data;
    const cwd = input.cwd || process.cwd();
    if (exitIfCwdNotAllowed("PostToolUseFailure", cwd)) return 0;
    const slug = projectSlugFromCwd(cwd);
    ensureDir(eidetixDir(slug));

    try {
      fs.writeFileSync(toolsRanPath(slug), "", "utf8");
    } catch (e) {
      warn(`tools-ran write: ${e.message}`);
    }

    emitJson({
      hookSpecificOutput: {
        hookEventName: "PostToolUseFailure",
        additionalContext:
          "Tool failure noted. If this reveals a clear Constraint or Signal "
          + "(e.g. platform limitation, API behaviour, environment requirement), "
          + "capture it in semantics.json at the next checkpoint. "
          + "Skip if the failure is transient or self-evident.",
      },
    });
    return 0;
  });
});
