from __future__ import annotations

import re
import uuid


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug


def org_id_for_slug(slug: str) -> str:
    cleaned = slugify(slug)
    if not cleaned:
        raise ValueError("Organization slug is required")
    return f"org:{cleaned}"


def employee_id_for_name(name: str) -> str:
    slug = slugify(name) or "member"
    return f"emp:{slug}"


def new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def email_domain(email: str) -> str:
    return email.strip().lower().rsplit("@", 1)[-1] if "@" in email else ""


def normalize_org_id(org_id: str) -> str:
    cleaned = org_id.strip()
    return cleaned if cleaned.startswith("org:") else f"org:{cleaned}"
