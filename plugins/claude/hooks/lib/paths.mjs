/**
 * Path resolution for Eidetix Claude hooks.
 * Path resolution for hooks. Scope configuration defaults to deny: an absent,
 * malformed, or empty allowlist must not activate Eidetix in any project.
 */

import fs from "node:fs";
import path from "node:path";

export function projectSlugFromCwd(cwd) {
  if (!cwd || typeof cwd !== "string") return "unknown-project";
  const base = path.basename(cwd);
  return base || "unknown-project";
}

export function contextGraphRoot() {
  const home = process.env.USERPROFILE || process.env.HOME || ".";
  return path.join(home, ".context-graph");
}

export function claudeAllowedDirsPath() {
  return path.join(contextGraphRoot(), "claude-allowed-dirs.json");
}

export function loadAllowedDirs() {
  try {
    const configPath = claudeAllowedDirsPath();
    if (!fs.existsSync(configPath)) return [];
    const raw = fs.readFileSync(configPath, "utf8");
    if (!raw || !raw.trim()) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed?.dirs)) return [];
    return parsed.dirs
      .filter((dir) => typeof dir === "string" && dir.trim().length > 0)
      .map((dir) => fs.realpathSync.native(path.resolve(dir)));
  } catch {
    return [];
  }
}

function normaliseForComparison(p) {
  const resolved = fs.realpathSync.native(path.resolve(p));
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

export function isCwdAllowed(cwd) {
  try {
    const allowedDirs = loadAllowedDirs();
    if (allowedDirs.length === 0) return false;

    if (!cwd || typeof cwd !== "string") return false;
    const resolvedCwd = normaliseForComparison(cwd);
    for (const dir of allowedDirs) {
      const allowedRoot = normaliseForComparison(dir);
      if (resolvedCwd === allowedRoot) return true;
      const relative = path.relative(allowedRoot, resolvedCwd);
      if (relative && !relative.startsWith("..") && !path.isAbsolute(relative)) {
        return true;
      }
    }
  } catch {
    return false;
  }
  return false;
}

export function projectDir(slug) {
  return path.join(contextGraphRoot(), "projects", slug);
}

export function semanticsPath(slug) {
  return path.join(projectDir(slug), "semantics.json");
}

export function eidetixDir(slug) {
  return path.join(projectDir(slug), ".eidetix");
}

export function sessionPath(slug) {
  return path.join(eidetixDir(slug), "session.json");
}

export function toolsRanPath(slug) {
  return path.join(eidetixDir(slug), "tools-ran");
}

export function stopBlockedOncePath(slug) {
  return path.join(eidetixDir(slug), "stop-blocked-once");
}

export function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}
