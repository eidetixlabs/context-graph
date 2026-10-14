/**
 * session.json — 3 fields only.
 *
 * Fields: session_id, last_checkpoint_len, last_checkpoint_at
 * Invalid JSON → return defaults (fail-open, corrupted flag set).
 */

import fs from "node:fs";

const DEFAULTS = () => ({
  session_id: "",
  last_checkpoint_len: 0,
  last_checkpoint_at: new Date().toISOString(),
});

export function readSession(filePath) {
  if (!fs.existsSync(filePath)) {
    return { session: DEFAULTS(), existed: false };
  }
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    const parsed = JSON.parse(raw);
    return {
      session: {
        session_id: String(parsed.session_id ?? ""),
        last_checkpoint_len: Number(parsed.last_checkpoint_len ?? 0),
        last_checkpoint_at: String(
          parsed.last_checkpoint_at ?? new Date().toISOString()
        ),
      },
      existed: true,
    };
  } catch {
    return { session: DEFAULTS(), existed: false, corrupted: true };
  }
}

export function writeSession(filePath, session) {
  fs.writeFileSync(filePath, JSON.stringify(session, null, 2) + "\n", "utf8");
}

/**
 * Update checkpoint only when semantics file is valid JSON.
 * Invalid semantics → preserve existing last_checkpoint_len until file recovers.
 */
export function updateCheckpointIfValid(filePath, session, currentLen, semanticsValid) {
  if (!semanticsValid) {
    return session;
  }
  return {
    ...session,
    last_checkpoint_len: currentLen,
    last_checkpoint_at: new Date().toISOString(),
  };
}
