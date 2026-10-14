#!/usr/bin/env node
/**
 * Eidetix — SessionEnd hook  (async: true in settings.json)
 *
 * Clears scratch files so the next session starts clean.
 * Does not modify session.json — last_checkpoint_len is preserved for
 * the next session's baseline.
 *
 * Async: Claude does not wait for this hook to finish.
 * Fail-open: exit 0 on any error.
 */

import fs from "node:fs";
import { readStdinJsonAsync, warn, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";
import {
  projectSlugFromCwd,
  eidetixDir,
  toolsRanPath,
  stopBlockedOncePath,
  ensureDir,
} from "./lib/paths.mjs";

runHook("SessionEnd", () => {
  return readStdinJsonAsync().then((stdin) => {
    if (!stdin.ok) warn(`stdin parse failed: ${stdin.error}`);
    const input = stdin.data;
    const cwd = input.cwd || process.cwd();
    if (exitIfCwdNotAllowed("SessionEnd", cwd)) return 0;
    const slug = projectSlugFromCwd(cwd);
    ensureDir(eidetixDir(slug));

    for (const p of [toolsRanPath(slug), stopBlockedOncePath(slug)]) {
      try {
        if (fs.existsSync(p)) fs.unlinkSync(p);
      } catch (e) {
        warn(`SessionEnd clear ${p}: ${e.message}`);
      }
    }
    return 0;
  });
});
