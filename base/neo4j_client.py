"""
neo4j_client.py — Thin wrapper around the neo4j Python driver.

Provides a Neo4jClient class that manages the driver lifecycle and
exposes a run() helper that returns a list of record dicts.

Per-org routing
---------------
Neo4jClient now accepts explicit connection params; they default to the
process-level env vars so all existing callers (processor, connectors,
one-off scripts) are unchanged.

OrgNeo4jRegistry provides a lazy per-org Neo4jClient cache keyed by org_id,
mirroring OrgPoolRegistry in base/db/client.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import Any, LiteralString

from neo4j import GraphDatabase, Driver, ManagedTransaction

_log = logging.getLogger(__name__)

_ORG_NEO4J_REGISTRY_LRU_CAP = 50


def _bolt_uri() -> str:
    return os.environ["NEO4J_URI"]


def _auth() -> tuple[str, str]:
    return os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]


class Neo4jClient:
    """Thread-safe Neo4j client. Instantiate once at startup.

    When called with explicit params the env vars are not read, so multiple
    org-specific instances can coexist in the same process.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        _uri = uri or _bolt_uri()
        _user = user or os.environ["NEO4J_USER"]
        _password = password or os.environ["NEO4J_PASSWORD"]
        self._driver: Driver = GraphDatabase.driver(_uri, auth=(_user, _password))
        self._database: str = database or os.environ.get("NEO4J_DATABASE", "neo4j")

    def verify(self) -> None:
        self._driver.verify_connectivity()

    def close(self) -> None:
        self._driver.close()

    def run(self, cypher: LiteralString, params: dict[str, Any] | None = None) -> list[dict]:
        """Execute a read Cypher query and return a list of record dicts."""
        params = params or {}

        def _work(tx: ManagedTransaction) -> list[dict]:
            result = tx.run(cypher, params)
            return [dict(record) for record in result]

        with self._driver.session(database=self._database) as session:
            return session.execute_read(_work)

    def run_write(self, cypher: LiteralString, params: dict[str, Any] | None = None) -> None:
        """Execute a write Cypher statement inside an explicit write transaction."""
        params = params or {}

        def _work(tx: ManagedTransaction) -> None:
            tx.run(cypher, params)

        with self._driver.session(database=self._database) as session:
            session.execute_write(_work)


# ---------------------------------------------------------------------------
# Per-org Neo4j client registry
# ---------------------------------------------------------------------------

class OrgNeo4jRegistry:
    """
    Lazy per-org Neo4jClient registry.

    The default org uses the process-global Neo4jClient (created at startup
    in webserver/main.py lifespan).  The registry short-circuits to it for
    the default org so no extra driver is opened.

    Non-default org clients are created on first request from the coords
    stored in org_env and cached with an LRU cap.

    Thread safety: asyncio.Lock guards all mutations.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._clients: OrderedDict[str, Neo4jClient] = OrderedDict()
        self._default_org_id: str = ""
        self._default_client: Neo4jClient | None = None

    def set_default(self, org_id: str, client: Neo4jClient) -> None:
        self._default_org_id = org_id
        self._default_client = client

    def get_default(self) -> Neo4jClient | None:
        return self._default_client

    async def get(
        self,
        org_id: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
    ) -> Neo4jClient:
        """Return the Neo4jClient for org_id, creating it from coords if needed."""
        if org_id == self._default_org_id and self._default_client is not None:
            return self._default_client

        async with self._lock:
            if org_id in self._clients:
                self._clients.move_to_end(org_id)
                return self._clients[org_id]

            client = Neo4jClient(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
            )
            self._clients[org_id] = client
            self._clients.move_to_end(org_id)

            while len(self._clients) > _ORG_NEO4J_REGISTRY_LRU_CAP:
                evicted_org, evicted_client = self._clients.popitem(last=False)
                _log.info("OrgNeo4jRegistry: evicting client for org=%s (LRU cap reached)", evicted_org)
                try:
                    evicted_client.close()
                except Exception as exc:
                    _log.warning("OrgNeo4jRegistry: error closing evicted client org=%s: %s", evicted_org, exc)

            return client

    async def remove(self, org_id: str) -> None:
        """Close and remove the client for an org."""
        async with self._lock:
            client = self._clients.pop(org_id, None)
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                _log.warning("OrgNeo4jRegistry: error closing client for org=%s: %s", org_id, exc)

    def close_all(self) -> None:
        """Close all non-default org clients (called on webserver shutdown)."""
        clients = list(self._clients.items())
        self._clients.clear()
        for org_id, client in clients:
            try:
                client.close()
            except Exception as exc:
                _log.warning("OrgNeo4jRegistry: error closing client org=%s on shutdown: %s", org_id, exc)


org_neo4j_registry = OrgNeo4jRegistry()
