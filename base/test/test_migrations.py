from __future__ import annotations

from pathlib import Path

import pytest

from base.db.migrations import (
    Migration,
    collect_org_databases,
    discover_migrations,
    run_migrations,
)


class _Transaction:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self):
        self.events.append("transaction:start")

    async def __aexit__(self, exc_type, exc, tb):
        self.events.append("transaction:rollback" if exc_type else "transaction:commit")
        return False


class _Connection:
    def __init__(self, applied: set[str] | None = None, fail_sql: str = "") -> None:
        self.applied = set(applied or ())
        self.fail_sql = fail_sql
        self.events: list[str] = []
        self.closed = False

    async def execute(self, sql: str, *args):
        if "pg_advisory_lock" in sql:
            self.events.append("lock")
        elif "pg_advisory_unlock" in sql:
            self.events.append("unlock")
        elif sql == self.fail_sql:
            self.events.append(f"execute:{sql}")
            raise RuntimeError("migration failed")
        elif sql.startswith("INSERT INTO schema_migrations"):
            self.applied.add(args[0])
            self.events.append(f"record:{args[0]}")
        elif not sql.startswith("CREATE TABLE IF NOT EXISTS schema_migrations"):
            self.events.append(f"execute:{sql}")
        return "OK"

    async def fetchval(self, sql: str, migration_id: str):
        assert "schema_migrations" in sql
        return migration_id in self.applied

    def transaction(self):
        return _Transaction(self.events)

    async def close(self):
        self.closed = True
        self.events.append("close")


def test_discover_migrations_uses_safe_ids_and_numeric_order(tmp_path: Path) -> None:
    (tmp_path / "010_second.sql").write_text("SELECT 10")
    (tmp_path / "002_first.sql").write_text("SELECT 2")
    (tmp_path / "README.md").write_text("ignored")

    migrations = discover_migrations(tmp_path)

    assert [migration.id for migration in migrations] == ["002_first", "010_second"]
    assert [migration.sql for migration in migrations] == ["SELECT 2", "SELECT 10"]


def test_discover_migrations_rejects_unsafe_sql_filename(tmp_path: Path) -> None:
    (tmp_path / "003_bad-name.sql").write_text("SELECT 3")

    with pytest.raises(ValueError, match="Unsafe migration filename"):
        discover_migrations(tmp_path)


@pytest.mark.asyncio
async def test_run_migrations_locks_skips_applied_and_records_pending() -> None:
    connection = _Connection(applied={"001_done"})

    async def connect(_dsn: str, **_kwargs):
        return connection

    applied = await run_migrations(
        "postgresql://secret@example/db",
        migrations=[
            Migration(1, "001_done", Path("001_done.sql"), "DESTRUCTIVE"),
            Migration(2, "002_pending", Path("002_pending.sql"), "CREATE THING"),
        ],
        connect=connect,
    )

    assert applied == ["002_pending"]
    assert "execute:DESTRUCTIVE" not in connection.events
    assert connection.events == [
        "lock",
        "transaction:start",
        "execute:CREATE THING",
        "record:002_pending",
        "transaction:commit",
        "unlock",
        "close",
    ]


@pytest.mark.asyncio
async def test_run_migrations_rolls_back_propagates_and_unlocks() -> None:
    connection = _Connection(fail_sql="BROKEN")

    async def connect(_dsn: str, **_kwargs):
        return connection

    with pytest.raises(RuntimeError, match="migration failed"):
        await run_migrations(
            "postgresql://secret@example/db",
            migrations=[Migration(18, "018_broken", Path("018_broken.sql"), "BROKEN")],
            connect=connect,
        )

    assert connection.events == [
        "lock",
        "transaction:start",
        "execute:BROKEN",
        "transaction:rollback",
        "unlock",
        "close",
    ]
    assert "018_broken" not in connection.applied


@pytest.mark.asyncio
async def test_connection_errors_do_not_expose_dsn_credentials() -> None:
    async def connect(dsn: str, **_kwargs):
        raise ValueError(f"could not parse {dsn}")

    with pytest.raises(RuntimeError) as error:
        await run_migrations(
            "postgresql://user:super-secret@example/db",
            migrations=[],
            connect=connect,
            database_label="central database",
        )

    assert "super-secret" not in str(error.value)
    assert str(error.value) == "Unable to connect to central database: ValueError"


def test_collect_org_databases_deduplicates_default_and_repeated_dsns() -> None:
    default = "postgresql://user:secret@central/db"
    rows = [
        {"org_id": "org:default", "pg_dsn": default},
        {"org_id": "org:a", "pg_dsn": "postgresql://a:secret@tenant/db"},
        {"org_id": "org:b", "pg_dsn": "postgresql://a:secret@tenant/db"},
        {"org_id": "org:c", "pg_dsn": " postgresql://c:secret@tenant/db "},
    ]

    assert collect_org_databases(rows, default) == [
        ("org:a", "postgresql://a:secret@tenant/db"),
        ("org:c", "postgresql://c:secret@tenant/db"),
    ]
