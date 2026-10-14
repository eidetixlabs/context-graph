#!/usr/bin/env node
/**
 * Eidetix — PostToolUse hook (minimal)
 *
 * Touches tools-ran to signal that tool activity occurred this turn.
 * The Stop hook reads this flag to decide whether to prompt for a checkpoint.
 *
 * No stdout JSON required. No additionalContext injected.
 * Fail-open: exit 0 on any error.
 */

import fs from "node:fs";
import { readStdinJsonAsync, warn, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";
import {
  projectSlugFromCwd,
  eidetixDir,
  toolsRanPath,
  ensureDir,
} from "./lib/paths.mjs";

runHook("PostToolUse", () => {
  return readStdinJsonAsync().then((stdin) => {
    if (!stdin.ok) warn(`stdin parse failed: ${stdin.error}`);
    const input = stdin.data;
    const cwd = input.cwd || process.cwd();
    if (exitIfCwdNotAllowed("PostToolUse", cwd)) return 0;
    const slug = projectSlugFromCwd(cwd);
    ensureDir(eidetixDir(slug));
    try {
      fs.writeFileSync(toolsRanPath(slug), "", "utf8");
    } catch (e) {
      warn(`tools-ran write: ${e.message}`);
    }
    return 0;
  });
});
