import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import dotenv_values


def mask_secret(value: Optional[str], keep_last: int = 4) -> str:
    if value is None or value == "":
        return "[empty]"
    text = str(value).strip()
    if len(text) <= keep_last:
        return "*" * max(len(text), 4)
    return "*" * (len(text) - keep_last) + text[-keep_last:]


def redact_dict(
    data: Any,
    sensitive_keys: Iterable[str] = ("password", "secret", "token", "api_key", "jwt"),
) -> Any:
    sensitive = {str(k).lower() for k in sensitive_keys}

    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            if str(key).lower() in sensitive:
                redacted[key] = mask_secret(str(value))
            else:
                redacted[key] = redact_dict(value, sensitive_keys)
        return redacted

    if isinstance(data, list):
        return [redact_dict(item, sensitive_keys) for item in data]

    return data


def get_secret(
    name: str,
    default: Optional[str] = None,
    *,
    required: bool = False,
    env_file: str | os.PathLike[str] = ".env",
) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        value = dotenv_values(env_file).get(name)

    if value is None and required:
        raise RuntimeError(f"Missing required secret: {name}")

    return value if value is not None else default


def read_text_file(
    path: str | os.PathLike[str],
    *,
    encoding: str = "utf-8",
    max_size: int = 1_048_576,
) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.stat().st_size > max_size:
        raise ValueError(f"File is too large to read safely: {file_path}")

    with file_path.open("r", encoding=encoding, errors="strict") as handle:
        return handle.read()


def write_text_file(
    path: str | os.PathLike[str],
    content: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            os.chmod(file_path, mode)
        except OSError:
            pass


def write_json_file(
    path: str | os.PathLike[str],
    payload: Dict[str, Any],
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> None:
    write_text_file(
        path,
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding=encoding,
        mode=mode,
    )


def log_safe(message: str, payload: dict[str, Any]) -> None:
    print(message, redact_dict(payload))