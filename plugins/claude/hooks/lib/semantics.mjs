/**
 * semantics.json read helpers — invalid-JSON contract.
 *
 * - Missing file → { entries: [], length: 0, valid: true }  (empty, not corrupt)
 * - Invalid JSON or non-array → { valid: false, warning }
 * - valid: false → caller must NOT update last_checkpoint_len
 */

import fs from "node:fs";

/**
 * @returns {{ entries: unknown[], length: number, valid: boolean, warning?: string }}
 */
export function readSemantics(filePath) {
  if (!fs.existsSync(filePath)) {
    return { entries: [], length: 0, valid: true };
  }
  let raw;
  try {
    raw = fs.readFileSync(filePath, "utf8");
  } catch (err) {
    return {
      entries: [],
      length: 0,
      valid: false,
      warning: `eidetix: cannot read semantics.json: ${err.message}`,
    };
  }
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return {
        entries: [],
        length: 0,
        valid: false,
        warning: "eidetix: semantics.json is not a JSON array",
      };
    }
    return { entries: parsed, length: parsed.length, valid: true };
  } catch (err) {
    return {
      entries: [],
      length: 0,
      valid: false,
      warning: `eidetix: invalid semantics.json (${err.message}); treating as [] for length checks`,
    };
  }
}
