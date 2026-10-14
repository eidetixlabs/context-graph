#!/usr/bin/env node
/**
 * Eidetix — UserPromptSubmit hook
 *
 * Injects recall guidance for every user prompt. Unknown entities should be
 * looked up via eidetix search(query) before asking clarification questions.
 *
 * Fail-open: exit 0 on any error.
 */

import { readStdinJsonAsync, emitJson, runHook, exitIfCwdNotAllowed } from "./lib/fail-open.mjs";

runHook("UserPromptSubmit", () => {
  return readStdinJsonAsync().then((stdin) => {
    const cwd = stdin.data.cwd || process.cwd();
    if (exitIfCwdNotAllowed("UserPromptSubmit", cwd)) return 0;

    emitJson({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext:
          "MANDATORY: if prompt references unknown term, name, entity, product, "
          + "or prior decision, call eidetix search(query) before replying or "
          + "asking for clarification. Do not say 'not familiar' before search. "
          + "Only ask user if search(query) returns no relevant context.",
      },
    });
    return 0;
  });
});
