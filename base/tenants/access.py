from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from base.tenants.ids import normalize_org_id
from base.tenants.service import is_platform_operator


async def ensure_org_admin_or_operator(
    employee: dict[str, Any],
    org_id: str,
    central_db: Any,
    *,
    operator_emails: list[str],
    cross_org_detail: str = "Admin access required",
) -> None:
    """Require owner/admin of the org, or a platform operator for cross-org access."""
    normalized = normalize_org_id(org_id)
    operator = await is_platform_operator(
        central_db,
        str(employee.get("email") or ""),
        operator_emails,
    )
    if employee.get("org_id") != normalized:
        if not operator:
            raise HTTPException(status_code=403, detail=cross_org_detail)
        return
    if employee.get("role") not in {"owner", "admin"}:
        raise HTTPException(status_code=403, detail="Admin access required")


async def ensure_org_owner_or_operator(
    employee: dict[str, Any],
    org_id: str,
    central_db: Any,
    *,
    operator_emails: list[str],
    cross_org_detail: str = "Owner access required",
) -> None:
    """Require an owner of the org, or a platform operator for cross-org access."""
    normalized = normalize_org_id(org_id)
    operator = await is_platform_operator(
        central_db,
        str(employee.get("email") or ""),
        operator_emails,
    )
    if employee.get("org_id") != normalized:
        if not operator:
            raise HTTPException(status_code=403, detail=cross_org_detail)
        return
    if employee.get("role") != "owner":
        raise HTTPException(status_code=403, detail="Owner access required")


def ensure_can_grant_role(employee: dict[str, Any], org_id: str, role: str) -> None:
    """Admins may invite members and viewers. Only owners may grant admin or owner."""
    if role not in {"owner", "admin"}:
        return
    normalized = normalize_org_id(org_id)
    if employee.get("org_id") != normalized:
        return
    if employee.get("role") != "owner":
        raise HTTPException(
            status_code=403,
            detail="Owner access required to grant admin or owner access",
        )
