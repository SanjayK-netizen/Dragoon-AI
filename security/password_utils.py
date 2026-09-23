from __future__ import annotations

import bcrypt


_MAX_BCRYPT_BYTES = 72
_DEFAULT_MIN_LENGTH = 12
_BCRYPT_ROUNDS = 12


def _validate_password(password: str, *, min_length: int) -> None:
    if not isinstance(password, str):
        raise TypeError("Password must be a string")
    if len(password) < min_length:
        raise ValueError(
            f"Password must be at least {min_length} characters long"
        )
    if len(password.encode("utf-8")) > _MAX_BCRYPT_BYTES:
        raise ValueError(
            "Password is too long for bcrypt; use at most 72 UTF-8 bytes"
        )


def hash_password(
    password: str,
    *,
    min_length: int = _DEFAULT_MIN_LENGTH,
) -> str:
    _validate_password(password, min_length=min_length)
    password_bytes = password.encode("utf-8")
    return bcrypt.hashpw(
        password_bytes,
        bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
    ).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    if not isinstance(password, str):
        raise TypeError("Password must be a string")
    if not isinstance(password_hash, str) or not password_hash:
        raise ValueError("Password hash must be a non-empty string")
    if len(password.encode("utf-8")) > _MAX_BCRYPT_BYTES:
        return False

    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("ascii"),
        )
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Stored password hash is invalid") from exc


def password_hash_needs_update(password_hash: str) -> bool:
    if not isinstance(password_hash, str) or not password_hash:
        raise ValueError("Password hash must be a non-empty string")
    parts = password_hash.split("$")
    if len(parts) != 4 or parts[1] not in {"2a", "2b", "2y"}:
        raise ValueError("Stored password hash is invalid")

    try:
        rounds = int(parts[2])
    except ValueError as exc:
        raise ValueError("Stored password hash is invalid") from exc

    return rounds != _BCRYPT_ROUNDS
