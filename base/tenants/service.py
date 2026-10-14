from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

_log = logging.getLogger(__name__)

from base.auth.password import hash_password
from base.tenants.ids import email_domain, employee_id_for_name, new_id, org_id_for_slug, slugify


class TenantError(Exception):
    pass


class TenantValidationError(TenantError):
    pass


class TenantConflict(TenantError):
    pass


class TenantNotFound(TenantError):
    pass


ASSIGNABLE_ROLES = {"owner", "admin", "member", "viewer"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _serialize(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key, value in list(out.items()):
        if hasattr(value, "isoformat"):
            out[key] = value.isoformat()
    return out


async def is_platform_operator(conn: Any, email: str, extra_emails: list[str] | None = None) -> bool:
    normalized = email.strip().lower()
    extras = {item.strip().lower() for item in (extra_emails or []) if item.strip()}
    if normalized in extras:
        return True
    return bool(await conn.fetchval("SELECT 1 FROM platform_operators WHERE email = $1", normalized))


async def _write_audit(
    conn: Any,
    *,
    org_id: str | None,
    actor_id: str | None,
    action: str,
    target_type: str,
    target_id: str,
    detail: dict[str, Any] | None = None,
) -> None:
    payload = json.dumps(detail or {})
    await conn.execute(
        """
        INSERT INTO platform_audit (org_id, actor_id, action, target_type, target_id, detail)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        """,
        org_id,
        actor_id,
        action,
        target_type,
        target_id,
        payload,
    )
    try:
        await conn.execute(
            """
            INSERT INTO audit_log (org_id, event_type, actor_id, target_type, target_id, detail)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            org_id or "",
            action,
            actor_id or "system:cloud",
            target_type,
            target_id,
            payload,
        )
    except Exception as exc:
        _log.warning("audit_log insert failed for action=%s: %s", action, exc)


async def _unique_employee_id(conn: Any, name: str) -> str:
    base = employee_id_for_name(name)
    candidate = base
    suffix = 2
    while await conn.fetchval("SELECT 1 FROM employees WHERE id = $1", candidate):
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


async def create_tenant(
    conn: Any,
    *,
    name: str,
    slug: str,
    owner_email: str,
    owner_name: str,
    owner_password: str,
    pg_dsn: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    plan: str = "beta",
    actor_id: str | None = None,
    email_domain_override: str | None = None,
) -> dict[str, Any]:
    cleaned_name = name.strip()
    cleaned_slug = slugify(slug or cleaned_name)
    owner_email_norm = owner_email.strip().lower()
    owner_name_clean = owner_name.strip()
    if not cleaned_name:
        raise TenantValidationError("Organization name is required")
    if not cleaned_slug:
        raise TenantValidationError("Organization slug is required")
    if "@" not in owner_email_norm:
        raise TenantValidationError("Owner email is required")
    if not owner_name_clean:
        raise TenantValidationError("Owner name is required")
    if len(owner_password) < 10:
        raise TenantValidationError("Owner password must be at least 10 characters")

    org_id = org_id_for_slug(cleaned_slug)
    domain = (email_domain_override or email_domain(owner_email_norm)).strip().lower() or None

    if await conn.fetchval("SELECT 1 FROM cloud_tenants WHERE slug = $1", cleaned_slug):
        raise TenantConflict(f"Tenant slug '{cleaned_slug}' already exists")
    if await conn.fetchval("SELECT 1 FROM organizations WHERE slug = $1", cleaned_slug):
        raise TenantConflict(f"Organization slug '{cleaned_slug}' already exists")
    if domain:
        if await conn.fetchval("SELECT 1 FROM org_env WHERE email_domain = $1", domain):
            raise TenantConflict(f"Email domain '{domain}' is already registered")
        if await conn.fetchval("SELECT 1 FROM cloud_tenants WHERE email_domain = $1", domain):
            raise TenantConflict(f"Email domain '{domain}' is already registered")

    employee_id: str
    try:
        async with conn.transaction():
            op_id = new_id("op_")
            await conn.execute(
                """
                INSERT INTO cloud_tenants (
                    org_id, slug, name, email_domain, owner_email, plan, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                org_id,
                cleaned_slug,
                cleaned_name,
                domain,
                owner_email_norm,
                plan,
                "provisioning",
            )
            await conn.execute(
                """
                INSERT INTO tenant_provision_ops (id, org_id, action, status, detail)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                op_id,
                org_id,
                "provision",
                "running",
                json.dumps({"mode": "shared_env"}),
            )
            await conn.execute(
                """
                INSERT INTO organizations (id, name, slug)
                VALUES ($1, $2, $3)
                """,
                org_id,
                cleaned_name,
                cleaned_slug,
            )
            employee_id = await _unique_employee_id(conn, owner_name_clean)
            await conn.execute(
                """
                INSERT INTO employees (
                    id, org_id, email, canonical_name, role, status, auth_provider
                )
                VALUES ($1, $2, $3, $4, $5, 'active', 'local')
                """,
                employee_id,
                org_id,
                owner_email_norm,
                owner_name_clean,
                "owner",
            )
            await conn.execute(
                """
                INSERT INTO local_credentials (employee_id, password_hash)
                VALUES ($1, $2)
                """,
                employee_id,
                hash_password(owner_password),
            )
            await conn.execute(
                """
                INSERT INTO org_env (
                    org_id, slug, email_domain, status, pg_dsn,
                    neo4j_uri, neo4j_user, neo4j_password, neo4j_database
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                org_id,
                cleaned_slug,
                domain,
                "active",
                pg_dsn,
                neo4j_uri,
                neo4j_user,
                neo4j_password,
                neo4j_database,
            )
            ready_at = _now()
            await conn.execute(
                """
                UPDATE cloud_tenants
                SET status = $1, ready_at = $2, updated_at = now(), provision_error = NULL
                WHERE org_id = $3
                """,
                "ready",
                ready_at,
                org_id,
            )
            await conn.execute(
                """
                UPDATE tenant_provision_ops
                SET status = $1, updated_at = now()
                WHERE id = $2
                """,
                "succeeded",
                op_id,
            )
    except TenantError:
        raise
    except Exception as exc:
        raise TenantError(f"Tenant provisioning failed: {exc}") from exc

    await _write_audit(
        conn,
        org_id=org_id,
        actor_id=actor_id,
        action="tenant_created",
        target_type="tenant",
        target_id=org_id,
        detail={"slug": cleaned_slug, "owner_email": owner_email_norm},
    )
    row = await conn.fetchrow("SELECT * FROM cloud_tenants WHERE org_id = $1", org_id)
    result = _serialize(row) or {"org_id": org_id, "slug": cleaned_slug, "status": "ready"}
    result["owner_employee_id"] = employee_id
    result["owner_email"] = owner_email_norm
    return result


async def list_tenants(conn: Any) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT org_id, slug, name, email_domain, owner_email, plan, status,
               provision_error, created_at, updated_at, ready_at
        FROM cloud_tenants
        ORDER BY created_at DESC
        """
    )
    return [_serialize(dict(row)) or {} for row in rows]


async def get_tenant(conn: Any, org_id: str) -> dict[str, Any]:
    row = await conn.fetchrow("SELECT * FROM cloud_tenants WHERE org_id = $1", org_id)
    if not row:
        raise TenantNotFound(f"Unknown tenant {org_id}")
    return _serialize(dict(row)) or {}


async def suspend_tenant(conn: Any, *, org_id: str, actor_id: str | None = None) -> dict[str, Any]:
    existing = await get_tenant(conn, org_id)
    await conn.execute(
        """
        UPDATE cloud_tenants
        SET status = $1, updated_at = now()
        WHERE org_id = $2
        """,
        "suspended",
        org_id,
    )
    await conn.execute(
        """
        UPDATE org_env
        SET status = $1, updated_at = now()
        WHERE org_id = $2
        """,
        "suspended",
        org_id,
    )
    await _write_audit(
        conn,
        org_id=org_id,
        actor_id=actor_id,
        action="tenant_suspended",
        target_type="tenant",
        target_id=org_id,
        detail={"previous_status": existing.get("status")},
    )
    return await get_tenant(conn, org_id)


async def create_invitation(
    conn: Any,
    *,
    org_id: str,
    email: str,
    role: str = "member",
    invited_by: str | None = None,
    ttl_hours: int = 168,
) -> dict[str, Any]:
    if role not in {"owner", "admin", "member", "viewer"}:
        raise TenantValidationError("Invalid role")
    email_norm = email.strip().lower()
    if "@" not in email_norm:
        raise TenantValidationError("Invite email is required")
    existing = await conn.fetchrow(
        "SELECT id, email, auth_provider, status FROM employees WHERE org_id = $1 AND email = $2",
        org_id,
        email_norm,
    )
    from base.db.identity import is_identity_stub

    if (
        existing
        and not is_identity_stub(
            employee_id=existing["id"],
            email=existing.get("email"),
            auth_provider=existing.get("auth_provider"),
        )
        and str(existing.get("status") or "active") != "deactivated"
    ):
        raise TenantConflict("That email already belongs to this organization")
    token = secrets.token_urlsafe(32)
    invite_id = new_id("inv_")
    expires_at = _now() + timedelta(hours=ttl_hours)
    await conn.execute(
        """
        INSERT INTO org_invitations (
            id, org_id, email, role, token_hash, invited_by, expires_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        invite_id,
        org_id,
        email_norm,
        role,
        _hash_token(token),
        invited_by,
        expires_at,
    )
    await _write_audit(
        conn,
        org_id=org_id,
        actor_id=invited_by,
        action="member_invited",
        target_type="invitation",
        target_id=invite_id,
        detail={"email": email_norm, "role": role},
    )
    return {
        "id": invite_id,
        "org_id": org_id,
        "email": email_norm,
        "role": role,
        "token": token,
        "expires_at": expires_at.isoformat(),
    }


async def get_invitation_meta(conn: Any, invite_id: str) -> dict[str, Any]:
    row = await conn.fetchrow(
        """
        SELECT id, org_id, email, role, expires_at, accepted_at
        FROM org_invitations
        WHERE id = $1
        """,
        invite_id,
    )
    if not row:
        raise TenantNotFound(f"Unknown invitation {invite_id}")
    invite = dict(row)
    expires_at = invite["expires_at"]
    if getattr(expires_at, "tzinfo", None) is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return {
        "id": invite["id"],
        "org_id": invite["org_id"],
        "email": invite["email"],
        "role": invite["role"],
        "expired": expires_at < _now(),
        "accepted": bool(invite.get("accepted_at")),
    }


async def list_invitations(conn: Any, org_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id, org_id, email, role, invited_by, accepted_at, expires_at, created_at
        FROM org_invitations
        WHERE org_id = $1
        ORDER BY created_at DESC
        """,
        org_id,
    )
    return [_serialize(dict(row)) or {} for row in rows]


async def accept_invitation(
    conn: Any,
    *,
    token: str,
    name: str,
    password: str,
    email: str | None = None,
) -> dict[str, Any]:
    if len(password) < 10:
        raise TenantValidationError("Password must be at least 10 characters")
    name_clean = name.strip()
    if not name_clean:
        raise TenantValidationError("Name is required")
    row = await conn.fetchrow(
        "SELECT * FROM org_invitations WHERE token_hash = $1",
        _hash_token(token),
    )
    if not row:
        raise TenantValidationError("Invitation is invalid")
    invite = dict(row)
    expires_at = invite["expires_at"]
    if getattr(expires_at, "tzinfo", None) is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if invite.get("accepted_at"):
        raise TenantValidationError("Invitation was already accepted")
    if expires_at < _now():
        raise TenantValidationError("Invitation has expired")

    env = await conn.fetchrow(
        "SELECT status FROM org_env WHERE org_id = $1",
        invite["org_id"],
    )
    if not env or env["status"] != "active":
        raise TenantValidationError("Organization is not accepting invitations")

    tenant = await conn.fetchrow(
        "SELECT status FROM cloud_tenants WHERE org_id = $1",
        invite["org_id"],
    )
    if tenant and tenant["status"] in {"suspended", "decommissioned"}:
        raise TenantValidationError("Organization is not accepting invitations")

    email_norm = (email or invite["email"]).strip().lower()
    if "@" not in email_norm:
        raise TenantValidationError("Email is required")

    existing = await conn.fetchrow(
        "SELECT id, email, auth_provider, status FROM employees WHERE org_id = $1 AND email = $2",
        invite["org_id"],
        email_norm,
    )
    from base.db.identity import is_identity_stub

    if (
        existing
        and not is_identity_stub(
            employee_id=existing["id"],
            email=existing.get("email"),
            auth_provider=existing.get("auth_provider"),
        )
        and str(existing.get("status") or "active") != "deactivated"
    ):
        raise TenantConflict("That email already belongs to this organization")

    if existing:
        employee_id = existing["id"]
        await conn.execute(
            """
            UPDATE employees
            SET canonical_name = $1,
                role = $2,
                status = 'active',
                auth_provider = 'local',
                deactivated_at = NULL,
                updated_at = now()
            WHERE id = $3
            """,
            name_clean,
            invite["role"],
            employee_id,
        )
    else:
        employee_id = await _unique_employee_id(conn, name_clean)
        await conn.execute(
            """
            INSERT INTO employees (
                id, org_id, email, canonical_name, role, status, auth_provider
            )
            VALUES ($1, $2, $3, $4, $5, 'active', 'local')
            """,
            employee_id,
            invite["org_id"],
            email_norm,
            name_clean,
            invite["role"],
        )
    await conn.execute(
        """
        INSERT INTO local_credentials (employee_id, password_hash)
        VALUES ($1, $2)
        ON CONFLICT (employee_id) DO UPDATE
            SET password_hash = EXCLUDED.password_hash,
                updated_at = now()
        """,
        employee_id,
        hash_password(password),
    )
    await conn.execute(
        """
        UPDATE org_invitations
        SET accepted_at = now()
        WHERE id = $1
        """,
        invite["id"],
    )
    await _write_audit(
        conn,
        org_id=invite["org_id"],
        actor_id=employee_id,
        action="invite_accepted",
        target_type="employee",
        target_id=employee_id,
        detail={"email": email_norm, "role": invite["role"]},
    )
    return {
        "employee_id": employee_id,
        "org_id": invite["org_id"],
        "email": email_norm,
        "name": name_clean,
        "role": invite["role"],
    }


def _member_payload(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": member.get("id"),
        "org_id": member.get("org_id"),
        "email": member.get("email"),
        "name": member.get("canonical_name") or member.get("name"),
        "role": member.get("role"),
        "status": member.get("status"),
    }


async def _lock_active_owners(conn: Any, org_id: str) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT id FROM employees
        WHERE org_id = $1 AND role = 'owner' AND status = 'active'
        FOR UPDATE
        """,
        org_id,
    )
    return list(rows or [])


async def _get_org_member(conn: Any, org_id: str, employee_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        "SELECT id, org_id, email, canonical_name, role, status, auth_provider "
        "FROM employees WHERE id = $1 AND org_id = $2",
        employee_id,
        org_id,
    )
    return dict(row) if row else None


def _reject_identity_stub(member: dict[str, Any]) -> None:
    from base.db.identity import is_identity_stub

    if is_identity_stub(
        employee_id=member.get("id"),
        email=member.get("email"),
        auth_provider=member.get("auth_provider"),
    ):
        raise TenantValidationError("Connector identities are not team members")


async def update_member_role(
    conn: Any,
    *,
    org_id: str,
    employee_id: str,
    role: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    if role not in ASSIGNABLE_ROLES:
        raise TenantValidationError("Invalid role")
    if actor_id and actor_id == employee_id:
        raise TenantValidationError("You can't change your own access")
    member = await _get_org_member(conn, org_id, employee_id)
    if not member:
        raise TenantNotFound("Team member not found")
    _reject_identity_stub(member)
    if member.get("status") != "active":
        raise TenantValidationError("Team member is not active")
    if member.get("role") == role:
        return _member_payload(member)

    # Role is read from employees on every request, so open sessions pick it up
    # without a new login.
    async with conn.transaction():
        if member.get("role") == "owner" and role != "owner":
            owners = await _lock_active_owners(conn, org_id)
            if len(owners) <= 1:
                raise TenantValidationError("The organization needs at least one owner")
        updated = await conn.fetchrow(
            """
            UPDATE employees
            SET role = $3, updated_at = now()
            WHERE id = $1 AND org_id = $2 AND status = 'active'
            RETURNING id, org_id, email, canonical_name, role, status
            """,
            employee_id,
            org_id,
            role,
        )
        if not updated:
            raise TenantConflict("Team member changed. Refresh and try again")
        await _write_audit(
            conn,
            org_id=org_id,
            actor_id=actor_id,
            action="member_role_changed",
            target_type="employee",
            target_id=employee_id,
            detail={"email": member.get("email"), "from": member.get("role"), "to": role},
        )
    return _member_payload(dict(updated))


async def _revoke_personal_access(conn: Any, employee_id: str) -> list[str]:
    """Sign the person out and drop credentials that belong to them, not the org."""
    sessions = await conn.fetch(
        """
        UPDATE sessions
        SET revoked_at = now()
        WHERE employee_id = $1 AND revoked_at IS NULL
        RETURNING token_hash
        """,
        employee_id,
    )
    token_hashes = [str(row["token_hash"]) for row in sessions if row.get("token_hash")]
    if token_hashes:
        await conn.execute(
            "UPDATE token_org SET revoked_at = now() WHERE token_hash = ANY($1::text[])",
            token_hashes,
        )
    await conn.execute("DELETE FROM local_credentials WHERE employee_id = $1", employee_id)
    await conn.execute(
        """
        UPDATE gmail_connections
        SET status = 'revoked',
            encrypted_refresh_token = '',
            encrypted_access_token = NULL,
            access_token_expires_at = NULL,
            error_message = 'Member removed',
            updated_at = now()
        WHERE employee_id = $1
        """,
        employee_id,
    )
    await conn.execute("DELETE FROM channel_members WHERE employee_id = $1", employee_id)
    await conn.execute(
        """
        UPDATE oauth_attempts
        SET status = 'expired',
            bootstrap_token = NULL,
            error_message = 'Member removed'
        WHERE employee_id = $1 AND status = 'pending'
        """,
        employee_id,
    )
    try:
        from base.db.access import invalidate_scope_cache

        invalidate_scope_cache(employee_id)
    except Exception:
        _log.warning("Scope cache invalidation failed for %s", employee_id, exc_info=True)
    return token_hashes


async def remove_member(
    conn: Any,
    *,
    org_id: str,
    employee_id: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    """Deactivate a member and clear personal access.

    Organization records stay: tickets, comments, usage, audit history, identity
    mappings, and service tokens. Those are shared history or org credentials.
    """
    if actor_id and actor_id == employee_id:
        raise TenantValidationError("You can't remove your own account")
    member = await _get_org_member(conn, org_id, employee_id)
    if not member:
        raise TenantNotFound("Team member not found")
    _reject_identity_stub(member)
    if member.get("status") == "deactivated":
        return _member_payload(member)
    email_norm = str(member.get("email") or "").strip().lower()
    async with conn.transaction():
        if member.get("role") == "owner" and member.get("status") == "active":
            owners = await _lock_active_owners(conn, org_id)
            if len(owners) <= 1:
                raise TenantValidationError("The organization needs at least one owner")
        updated = await conn.fetchrow(
            """
            UPDATE employees
            SET status = 'deactivated', deactivated_at = now(), updated_at = now()
            WHERE id = $1 AND org_id = $2 AND status <> 'deactivated'
            RETURNING id
            """,
            employee_id,
            org_id,
        )
        if not updated:
            raise TenantConflict("Team member changed. Refresh and try again")
        await _revoke_personal_access(conn, employee_id)
        if email_norm:
            await conn.execute(
                """
                DELETE FROM org_invitations
                WHERE org_id = $1 AND email = $2 AND accepted_at IS NULL
                """,
                org_id,
                email_norm,
            )
        await _write_audit(
            conn,
            org_id=org_id,
            actor_id=actor_id,
            action="member_removed",
            target_type="employee",
            target_id=employee_id,
            detail={"email": email_norm, "role": member.get("role")},
        )
    member["status"] = "deactivated"
    return _member_payload(member)


async def update_invitation_role(
    conn: Any,
    *,
    org_id: str,
    invite_id: str,
    role: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    if role not in ASSIGNABLE_ROLES:
        raise TenantValidationError("Invalid role")
    updated = await conn.fetchrow(
        """
        UPDATE org_invitations
        SET role = $3
        WHERE id = $1 AND org_id = $2 AND accepted_at IS NULL
        RETURNING id, org_id, email, role, invited_by, accepted_at, expires_at, created_at
        """,
        invite_id,
        org_id,
        role,
    )
    if not updated:
        existing = await conn.fetchrow(
            "SELECT id, accepted_at FROM org_invitations WHERE id = $1 AND org_id = $2",
            invite_id,
            org_id,
        )
        if existing and existing.get("accepted_at"):
            raise TenantValidationError("Invitation was already accepted")
        raise TenantNotFound("Invitation not found")
    await _write_audit(
        conn,
        org_id=org_id,
        actor_id=actor_id,
        action="invitation_role_changed",
        target_type="invitation",
        target_id=invite_id,
        detail={"email": updated.get("email"), "role": role},
    )
    return _serialize(dict(updated)) or {}


async def revoke_invitation(
    conn: Any,
    *,
    org_id: str,
    invite_id: str,
    actor_id: str | None = None,
) -> dict[str, Any]:
    existing = await conn.fetchrow(
        """
        SELECT id, email, role, accepted_at
        FROM org_invitations
        WHERE id = $1 AND org_id = $2
        """,
        invite_id,
        org_id,
    )
    if not existing:
        raise TenantNotFound("Invitation not found")
    if existing.get("accepted_at"):
        raise TenantValidationError("Invitation was already accepted")
    await conn.execute(
        """
        DELETE FROM org_invitations
        WHERE id = $1 AND org_id = $2 AND accepted_at IS NULL
        """,
        invite_id,
        org_id,
    )
    await _write_audit(
        conn,
        org_id=org_id,
        actor_id=actor_id,
        action="invitation_revoked",
        target_type="invitation",
        target_id=invite_id,
        detail={"email": existing.get("email"), "role": existing.get("role")},
    )
    return {"id": invite_id, "revoked": True}
