import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { isCwdAllowed } from "./paths.mjs";

function withTempHome(run) {
  const tempHome = fs.mkdtempSync(path.join(os.tmpdir(), "eidetix-allowed-dirs-"));
  const previousHome = process.env.HOME;
  const previousUserProfile = process.env.USERPROFILE;

  process.env.HOME = tempHome;
  process.env.USERPROFILE = tempHome;
  try {
    run(tempHome);
  } finally {
    if (previousHome === undefined) delete process.env.HOME;
    else process.env.HOME = previousHome;

    if (previousUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = previousUserProfile;

    fs.rmSync(tempHome, { recursive: true, force: true });
  }
}

function writeAllowedDirs(home, dirs) {
  const root = path.join(home, ".context-graph");
  fs.mkdirSync(root, { recursive: true });
  fs.writeFileSync(
    path.join(root, "claude-allowed-dirs.json"),
    JSON.stringify({ dirs }, null, 2),
    "utf8",
  );
}

test("isCwdAllowed denies all when list is empty", () => {
  withTempHome((home) => {
    writeAllowedDirs(home, []);
    assert.equal(isCwdAllowed(path.join(home, "anywhere")), false);
  });
});

test("isCwdAllowed denies all when the allowlist is missing", () => {
  withTempHome((home) => {
    assert.equal(isCwdAllowed(path.join(home, "anywhere")), false);
  });
});

test("isCwdAllowed denies all when the allowlist is malformed", () => {
  withTempHome((home) => {
    const root = path.join(home, ".context-graph");
    fs.mkdirSync(root, { recursive: true });
    fs.writeFileSync(path.join(root, "claude-allowed-dirs.json"), "{", "utf8");

    assert.equal(isCwdAllowed(path.join(home, "anywhere")), false);
  });
});

test("isCwdAllowed allows configured root and descendants only", () => {
  withTempHome((home) => {
    const nodeops = path.join(home, "root", "nodeops");
    fs.mkdirSync(path.join(nodeops, "dirA", "dirA1"), { recursive: true });
    fs.mkdirSync(path.join(home, "root", "someotherdir"), { recursive: true });
    writeAllowedDirs(home, [nodeops]);

    assert.equal(isCwdAllowed(nodeops), true, "exact root should be allowed");
    assert.equal(
      isCwdAllowed(path.join(nodeops, "dirA", "dirA1")),
      true,
      "descendant should be allowed",
    );
    assert.equal(
      isCwdAllowed(path.join(home, "root", "someotherdir")),
      false,
      "sibling should be denied",
    );
    assert.equal(
      isCwdAllowed(path.join(home, "root")),
      false,
      "parent of allowed root should be denied",
    );
  });
});

test("isCwdAllowed denies symlinks that escape an allowed root", () => {
  withTempHome((home) => {
    const allowed = path.join(home, "allowed");
    const excluded = path.join(home, "excluded");
    fs.mkdirSync(allowed, { recursive: true });
    fs.mkdirSync(excluded, { recursive: true });
    fs.symlinkSync(excluded, path.join(allowed, "escape"), "dir");
    writeAllowedDirs(home, [allowed]);

    assert.equal(isCwdAllowed(path.join(allowed, "escape")), false);
  });
});
