"""Ordered, transactional PostgreSQL migrations for existing and fresh databases."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence


_log = logging.getLogger(__name__)
_MIGRATION_FILENAME = re.compile(r"^(?P<number>[0-9]{3})_(?P<name>[a-z0-9][a-z0-9_]*)\.sql$")
_MIGRATION_LOCK_ID = 19510567985744216  # stable, application-specific bigint
_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
""".strip()

Connect = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class Migration:
    number: int
    id: str
    path: Path
    sql: str


def migration_directory() -> Path:
    """Return the migration directory packaged with the base module."""
    return Path(__file__).resolve().parent / "init"


def discover_migrations(directory: Path | str | None = None) -> list[Migration]:
    """Load valid SQL migrations in deterministic numeric/name order."""
    root = Path(directory) if directory is not None else migration_directory()
    migrations: list[Migration] = []
    seen_ids: set[str] = set()
    for path in root.glob("*.sql"):
        match = _MIGRATION_FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unsafe migration filename: {path.name}")
        migration_id = path.stem
        if migration_id in seen_ids:
            raise ValueError(f"Duplicate migration id: {migration_id}")
        seen_ids.add(migration_id)
        migrations.append(
            Migration(
                number=int(match.group("number")),
                id=migration_id,
                path=path,
                sql=path.read_text(encoding="utf-8"),
            )
        )
    if not migrations:
        raise RuntimeError(f"No SQL migrations found in {root}")
    return sorted(migrations, key=lambda migration: (migration.number, migration.id))


async def _default_connect(dsn: str, **kwargs: Any) -> Any:
    import asyncpg

    return await asyncpg.connect(dsn, **kwargs)


async def run_migrations(
    dsn: str,
    *,
    migrations: Sequence[Migration] | None = None,
    connect: Connect = _default_connect,
    database_label: str = "database",
) -> list[str]:
    """Apply pending migrations while holding a cross-replica advisory lock."""
    if not dsn.strip():
        raise RuntimeError("Database DSN is required for migrations")

    ordered = list(migrations) if migrations is not None else discover_migrations()
    try:
        connection = await connect(dsn.strip(), command_timeout=300)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to connect to {database_label}: {type(exc).__name__}"
        ) from None
    locked = False
    try:
        await connection.execute("SELECT pg_advisory_lock($1::bigint)", _MIGRATION_LOCK_ID)
        locked = True
        await connection.execute(_TRACKING_TABLE_SQL)
        applied: list[str] = []
        for migration in ordered:
            already_applied = await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM schema_migrations WHERE id = $1)",
                migration.id,
            )
            if already_applied:
                continue
            _log.info("Applying PostgreSQL migration %s to %s", migration.id, database_label)
            async with connection.transaction():
                await connection.execute(migration.sql)
                await connection.execute(
                    "INSERT INTO schema_migrations (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
                    migration.id,
                )
            applied.append(migration.id)
        return applied
    finally:
        if locked:
            try:
                await connection.execute("SELECT pg_advisory_unlock($1::bigint)", _MIGRATION_LOCK_ID)
            except Exception:
                _log.exception("Failed to release PostgreSQL migration lock for %s", database_label)
        await connection.close()


def collect_org_databases(
    rows: Iterable[Mapping[str, Any]],
    default_dsn: str,
) -> list[tuple[str, str]]:
    """Return unique active-org databases, excluding the central/default DSN."""
    seen = {default_dsn.strip()}
    databases: list[tuple[str, str]] = []
    for row in rows:
        dsn = str(row["pg_dsn"]).strip()
        if not dsn or dsn in seen:
            continue
        seen.add(dsn)
        databases.append((str(row["org_id"]), dsn))
    return databases


async def migrate_all_databases(
    default_dsn: str,
    *,
    migrations: Sequence[Migration] | None = None,
    connect: Connect = _default_connect,
) -> dict[str, list[str]]:
    """Migrate central Postgres, then every distinct active org Postgres."""
    ordered = list(migrations) if migrations is not None else discover_migrations()
    results = {
        "central": await run_migrations(
            default_dsn,
            migrations=ordered,
            connect=connect,
            database_label="central database",
        )
    }

    try:
        central = await connect(default_dsn.strip(), command_timeout=30)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read active org databases: {type(exc).__name__}"
        ) from None
    try:
        rows = await central.fetch(
            "SELECT org_id, pg_dsn FROM org_env WHERE status = 'active' ORDER BY org_id"
        )
    finally:
        await central.close()

    for org_id, dsn in collect_org_databases(rows, default_dsn):
        results[org_id] = await run_migrations(
            dsn,
            migrations=ordered,
            connect=connect,
            database_label=f"org {org_id}",
        )
    return results


async def _run_cli(single_database: bool) -> None:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    if single_database:
        await run_migrations(dsn)
    else:
        await migrate_all_databases(dsn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply pending Eidetix PostgreSQL migrations")
    parser.add_argument(
        "--single-database",
        action="store_true",
        help="Migrate DATABASE_URL only; do not read central org_env",
    )
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    asyncio.run(_run_cli(args.single_database))


if __name__ == "__main__":
    main()
