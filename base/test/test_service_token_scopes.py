from __future__ import annotations

import sys
import types

class _FakeGraphDatabase:
    @staticmethod
    def driver(*_args, **_kwargs):
        return None


sys.modules.setdefault(
    "neo4j",
    types.SimpleNamespace(
        GraphDatabase=_FakeGraphDatabase,
        Driver=object,
        ManagedTransaction=object,
    ),
)
class _FakePasswordHasher:
    def __init__(self, *args, **kwargs):
        pass

    def hash(self, plaintext: str) -> str:
        return plaintext

    def verify(self, hashed: str, plaintext: str) -> bool:
        return hashed == plaintext

    def check_needs_rehash(self, _hashed: str) -> bool:
        return False


sys.modules.setdefault(
    "argon2",
    types.SimpleNamespace(PasswordHasher=_FakePasswordHasher),
)
sys.modules.setdefault(
    "argon2.exceptions",
    types.SimpleNamespace(
        VerifyMismatchError=Exception,
        VerificationError=Exception,
        InvalidHashError=Exception,
    ),
)

import pytest
from fastapi import HTTPException

from base.auth.routes import _normalize_scopes


def test_normalize_scopes_deduplicates_and_sorts() -> None:
    scopes = _normalize_scopes(["workflow:read", "mcp:read", "workflow:read"])
    assert scopes == ["mcp:read", "workflow:read"]


def test_normalize_scopes_rejects_unknown_scope() -> None:
    with pytest.raises(HTTPException) as exc:
        _normalize_scopes(["workflow:delete"])
    assert exc.value.status_code == 400
