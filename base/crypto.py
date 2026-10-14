"""
base/crypto.py — Symmetric encryption for credentials stored at rest.

Used to encrypt per-user OAuth refresh/access tokens (e.g. gmail_connections)
before they are written to Postgres. Never store the plaintext secret.

Key management
--------------
CREDENTIAL_ENCRYPTION_KEY must be a urlsafe-base64 32-byte key, i.e. the
format produced by `Fernet.generate_key()`. Supplied via environment
(populated from a K8s Secret; see deployment/k8s/helm/eidetix/values.yaml
`eidetix.credentialEncryptionKey`). Never committed, never logged.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken

_KEY_ENV_VAR = "CREDENTIAL_ENCRYPTION_KEY"


class CredentialEncryptionError(RuntimeError):
    """Raised when encryption/decryption cannot proceed."""


def _fernet() -> Fernet:
    key = os.environ.get(_KEY_ENV_VAR, "").strip()
    if not key:
        raise CredentialEncryptionError(
            f"{_KEY_ENV_VAR} is not configured on this server."
        )
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise CredentialEncryptionError(f"{_KEY_ENV_VAR} is malformed.") from exc


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a plaintext secret. Returns an opaque urlsafe-base64 token."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_secret(token: str) -> str:
    """Decrypt a token produced by encrypt_secret. Raises CredentialEncryptionError."""
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise CredentialEncryptionError("Stored credential could not be decrypted.") from exc
