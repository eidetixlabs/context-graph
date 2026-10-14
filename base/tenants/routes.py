from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field

from base.auth.deps import CurrentEmployee
from base.db.client import DbConn, get_db
from base.tenants.access import (
    ensure_can_grant_role,
    ensure_org_admin_or_operator,
    ensure_org_owner_or_operator,
)
from base.tenants.ids import normalize_org_id
from base.tenants.service import (
    TenantConflict,
    TenantError,
    TenantNotFound,
    TenantValidationError,
    accept_invitation,
    create_invitation,
    create_tenant,
    get_invitation_meta,
    get_tenant,
    is_platform_operator,
    list_invitations,
    list_tenants,
    remove_member,
    revoke_invitation,
    suspend_tenant,
    update_invitation_role,
    update_member_role,
)

router = APIRouter(tags=["tenants"])


class CreateTenantRequest(BaseModel):
    name: str
    slug: str | None = None
    owner_email: EmailStr
    owner_name: str
    owner_password: str = Field(min_length=10)
    email_domain: str | None = None
    plan: str = "beta"


class InviteRequest(BaseModel):
    email: EmailStr
    role: str = "member"


class MemberRoleRequest(BaseModel):
    role: str


class InviteRoleRequest(BaseModel):
    role: str


class AcceptInviteRequest(BaseModel):
    token: str
    name: str
    password: str = Field(min_length=10)
    email: EmailStr | None = None


def _operator_emails() -> list[str]:
    raw = os.environ.get("EIDETIX_OPERATOR_EMAILS", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _shared_env_coords() -> dict[str, str]:
    return {
        "pg_dsn": os.environ.get("DATABASE_URL", ""),
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("NEO4J_PASSWORD", "password"),
        "neo4j_database": os.environ.get("NEO4J_DATABASE", "neo4j"),
    }


async def require_operator(
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    email = str(employee.get("email") or "")
    if await is_platform_operator(central_db, email, _operator_emails()):
        return employee
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operator access required")


def _raise_tenant_error(exc: TenantError) -> None:
    if isinstance(exc, TenantValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, TenantConflict):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, TenantNotFound):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/operator", include_in_schema=False)
async def operator_console(_employee: dict[str, Any] = Depends(require_operator)) -> HTMLResponse:
    html = (Path(__file__).resolve().parent / "operator.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@router.get("/tenants/me")
async def tenant_operator_status(
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    email = str(employee.get("email") or "")
    operator = await is_platform_operator(central_db, email, _operator_emails())
    return {
        "operator": operator,
        "org_id": employee.get("org_id"),
        "email": email,
        "role": employee.get("role"),
    }


@router.get("/tenants")
async def api_list_tenants(
    _employee: dict[str, Any] = Depends(require_operator),
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    return {"tenants": await list_tenants(central_db)}


@router.post("/tenants")
async def api_create_tenant(
    body: CreateTenantRequest,
    employee: dict[str, Any] = Depends(require_operator),
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    coords = _shared_env_coords()
    if not coords["pg_dsn"]:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")
    try:
        return await create_tenant(
            central_db,
            name=body.name,
            slug=body.slug or body.name,
            owner_email=str(body.owner_email),
            owner_name=body.owner_name,
            owner_password=body.owner_password,
            email_domain_override=body.email_domain,
            plan=body.plan,
            actor_id=str(employee.get("id") or employee.get("email")),
            **coords,
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.post("/tenants/{org_id}/suspend")
async def api_suspend_tenant(
    org_id: str,
    employee: dict[str, Any] = Depends(require_operator),
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await suspend_tenant(
            central_db,
            org_id=normalize_org_id(org_id),
            actor_id=str(employee.get("id") or employee.get("email")),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.get("/tenants/{org_id}")
async def api_get_tenant(
    org_id: str,
    _employee: dict[str, Any] = Depends(require_operator),
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await get_tenant(central_db, normalize_org_id(org_id))
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.get("/tenants/{org_id}/invites")
async def api_list_invites(
    org_id: str,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_admin_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
    )
    return {"invites": await list_invitations(central_db, normalized)}


@router.post("/tenants/{org_id}/invites")
async def api_create_invite(
    org_id: str,
    body: InviteRequest,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_admin_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
        cross_org_detail="Cannot invite to another organization",
    )
    ensure_can_grant_role(employee, normalized, body.role)
    try:
        return await create_invitation(
            central_db,
            org_id=normalized,
            email=str(body.email),
            role=body.role,
            invited_by=str(employee.get("id") or employee.get("email")),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


def _actor_id(employee: dict[str, Any]) -> str:
    return str(employee.get("id") or employee.get("email") or "")


@router.patch("/tenants/{org_id}/members/{employee_id}")
async def api_update_member_role(
    org_id: str,
    employee_id: str,
    body: MemberRoleRequest,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_owner_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
    )
    try:
        return await update_member_role(
            central_db,
            org_id=normalized,
            employee_id=employee_id,
            role=body.role,
            actor_id=_actor_id(employee),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.delete("/tenants/{org_id}/members/{employee_id}")
async def api_remove_member(
    org_id: str,
    employee_id: str,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_owner_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
    )
    try:
        return await remove_member(
            central_db,
            org_id=normalized,
            employee_id=employee_id,
            actor_id=_actor_id(employee),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.patch("/tenants/{org_id}/invites/{invite_id}")
async def api_update_invite_role(
    org_id: str,
    invite_id: str,
    body: InviteRoleRequest,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_owner_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
    )
    ensure_can_grant_role(employee, normalized, body.role)
    try:
        return await update_invitation_role(
            central_db,
            org_id=normalized,
            invite_id=invite_id,
            role=body.role,
            actor_id=_actor_id(employee),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.delete("/tenants/{org_id}/invites/{invite_id}")
async def api_revoke_invite(
    org_id: str,
    invite_id: str,
    employee: CurrentEmployee,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    normalized = normalize_org_id(org_id)
    await ensure_org_owner_or_operator(
        employee,
        normalized,
        central_db,
        operator_emails=_operator_emails(),
    )
    try:
        return await revoke_invitation(
            central_db,
            org_id=normalized,
            invite_id=invite_id,
            actor_id=_actor_id(employee),
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.get("/invites/{invite_id}/meta")
async def api_invite_meta(
    invite_id: str,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await get_invitation_meta(central_db, invite_id)
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise


@router.post("/invites/accept")
async def api_accept_invite(
    body: AcceptInviteRequest,
    central_db: DbConn = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await accept_invitation(
            central_db,
            token=body.token,
            name=body.name,
            password=body.password,
            email=str(body.email) if body.email else None,
        )
    except TenantError as exc:
        _raise_tenant_error(exc)
        raise
