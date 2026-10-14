"""
auth/password.py — argon2id password hashing and verification.

Uses argon2-cffi, which wraps the reference C implementation.
Argon2id is the recommended variant (resists both side-channel and
GPU-brute-force attacks).

The default parameters (time_cost=3, memory_cost=65536, parallelism=4)
match OWASP's minimum recommendation for argon2id as of 2025.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

_ph = PasswordHasher(
    time_cost=3,        # iterations
    memory_cost=65536,  # 64 MB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(plaintext: str) -> str:
    """Return an argon2id hash string suitable for storing in local_credentials."""
    return _ph.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    """
    Return True if plaintext matches hashed, False otherwise.
    Never raises — all argon2 exceptions are caught and converted to False.
    """
    try:
        return _ph.verify(hashed, plaintext)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    """
    Return True if the stored hash was produced with older parameters
    and should be re-hashed on next successful login.
    """
    return _ph.check_needs_rehash(hashed)
