from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

TargetCategory = Literal[
    "cursor-rule",
    "cursor-skill",
    "claude-rule",
    "claude-hook",
    "claude-hook-lib",
]


class PluginPackFile(BaseModel):
    id: str
    sourcePath: str
    targetCategory: TargetCategory
    sha256: str
    size: int


class PluginPackManifest(BaseModel):
    pluginPackVersion: str
    files: list[PluginPackFile]


_PLUGIN_ASSETS_DIR = Path(os.environ.get("PLUGIN_PACK_ASSETS_DIR", "/app/plugin-pack"))
_PLUGIN_PACK_VERSION_FILE = Path(
    os.environ.get("PLUGIN_PACK_VERSION_FILE", "/app/plugin-pack.version")
)


def plugin_pack_version() -> str:
    env_version = os.environ.get("PLUGIN_PACK_VERSION", "").strip()
    if env_version:
        return env_version
    if _PLUGIN_PACK_VERSION_FILE.exists():
        return _PLUGIN_PACK_VERSION_FILE.read_text(encoding="utf-8").strip() or "dev"
    return "dev"


def plugin_pack_assets_dir() -> Path:
    return _PLUGIN_ASSETS_DIR


def _target_category_for_path(rel_path: str) -> TargetCategory | None:
    if rel_path == "cursor-plugin/rules/semantics.mdc":
        return "cursor-rule"
    if rel_path.startswith("cursor-plugin/skills/") and rel_path.endswith("/SKILL.md"):
        return "cursor-skill"
    if rel_path == "claude-plugin/eidetix-semantics.md":
        return "claude-rule"
    if rel_path.startswith("claude-plugin/hooks/lib/") and rel_path.endswith(".mjs"):
        return "claude-hook-lib"
    if rel_path.startswith("claude-plugin/hooks/") and rel_path.endswith(".mjs"):
        return "claude-hook"
    return None


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 64), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_plugin_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    patterns = [
        "cursor-plugin/rules/semantics.mdc",
        "cursor-plugin/skills/*/SKILL.md",
        "claude-plugin/eidetix-semantics.md",
        "claude-plugin/hooks/*.mjs",
        "claude-plugin/hooks/lib/*.mjs",
    ]
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted({path.resolve() for path in paths if path.is_file()})


def build_manifest() -> PluginPackManifest:
    root = plugin_pack_assets_dir()
    files: list[PluginPackFile] = []

    for path in _iter_plugin_files(root):
        rel_path = path.relative_to(root).as_posix()
        target = _target_category_for_path(rel_path)
        if target is None:
            continue
        files.append(
            PluginPackFile(
                id=rel_path,
                sourcePath=rel_path,
                targetCategory=target,
                sha256=_hash_file(path),
                size=path.stat().st_size,
            )
        )

    return PluginPackManifest(
        pluginPackVersion=plugin_pack_version(),
        files=files,
    )


def resolve_plugin_file_path(file_id: str) -> Path | None:
    normalized = file_id.strip().lstrip("/")
    if not normalized:
        return None
    candidate = (plugin_pack_assets_dir() / normalized).resolve()
    root = plugin_pack_assets_dir().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    if _target_category_for_path(normalized) is None:
        return None
    return candidate
