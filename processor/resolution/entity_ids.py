"""Shared helpers for canonical entity ID generation."""

from __future__ import annotations

import re


def slugify_entity_component(value: str) -> str:
    """Convert arbitrary text into a stable lowercase slug."""
    raw = (value or "").strip().lower()
    if raw.startswith("org:"):
        raw = raw.removeprefix("org:")
    slug = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return slug or "unknown"


def build_entity_id(entity_type: str, org_id: str, canonical_name: str) -> str:
    """Build canonical ID: entity:{entity_type}:{org_slug}:{name_slug}."""
    return (
        f"entity:{slugify_entity_component(entity_type)}:"
        f"{slugify_entity_component(org_id)}:"
        f"{slugify_entity_component(canonical_name)}"
    )
