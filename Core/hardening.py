"""Dragoon — core/hardening.py (Phase 6)

Safety guardrails for user input and tool execution.

This module is intentionally small and conservative: it performs lightweight
validation and red-flag detection without trying to be a full security layer.
It is designed to sit in front of agent/tool execution and help prevent
unsafe or ambiguous actions from being run blindly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


DANGEROUS_TOOL_NAMES = {
    "delete_file",
    "remove_file",
    "rm",
    "shell",
    "powershell",
    "cmd",
    "format_disk",
    "shutdown",
    "reboot",
    "network_scan",
}

SUSPICIOUS_PATH_PATTERNS = (
    "..",
    "\\\\",
    "/etc",
    "/bin",
    "/usr",
    "/var",
    "system32",
    "program files",
)


def normalize_text(text: Any) -> str:
    """Return a cleaned string for safe logging or comparison."""
    if text is None:
        return ""
    value = str(text).strip()
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def contains_dangerous_shell_pattern(text: str) -> bool:
    """Detect shell-style or destructive command patterns."""
    normalized = normalize_text(text).lower()
    patterns = (
        "; rm ",
        "&& del ",
        "powershell",
        "cmd /c",
        "shutdown",
        "format c:",
        "curl ",
        "wget ",
        "http://",
        "https://",
    )
    return any(pattern in normalized for pattern in patterns)


def validate_safe_path(path: str, project_root: Optional[str] = None) -> str:
    """Reject traversal and unsafe absolute paths for local-file access.

    The file tool in the agent loop is intentionally constrained to the project
    directory; this helper enforces that at the hardening layer.
    """
    candidate = normalize_text(path)
    if not candidate:
        raise ValueError("file path is required")

    if candidate.startswith(("/", "\\")):
        raise ValueError("absolute paths are not allowed")

    if any(token in candidate.lower() for token in SUSPICIOUS_PATH_PATTERNS):
        raise ValueError("unsafe path detected")

    root = Path(project_root or os.getcwd()).resolve()
    requested = (root / candidate).resolve()

    try:
        requested.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the project root") from exc

    return str(requested)


def validate_tool_payload(tool_name: str, args: Any) -> Dict[str, Any]:
    """Validate a tool payload before it reaches execution."""
    if not isinstance(args, dict):
        raise ValueError(f"tool payload for {tool_name!r} must be a dict")

    cleaned: Dict[str, Any] = {}
    for key, value in args.items():
        key_name = normalize_text(key)
        if not key_name:
            continue
        cleaned[key_name] = value

    if tool_name in DANGEROUS_TOOL_NAMES:
        raise ValueError(f"tool {tool_name!r} is blocked by hardening rules")

    if tool_name == "open_local_file":
        if "path" not in cleaned:
            raise ValueError("open_local_file requires a path")
        validate_safe_path(str(cleaned["path"]))

    if tool_name == "set_reminder":
        text = normalize_text(cleaned.get("text", ""))
        due = normalize_text(cleaned.get("due", ""))
        if not text or not due:
            raise ValueError("reminder text and due time are required")

    return cleaned


def should_block_command(text: str) -> Tuple[bool, str]:
    """Return (blocked, reason) for hostile or unsafe prompting patterns."""
    value = normalize_text(text)
    if not value:
        return True, "empty input"

    if contains_dangerous_shell_pattern(value):
        return True, "contains shell-like destructive instruction"

    if "ignore previous instructions" in value.lower():
        return True, "contains prompt-injection pattern"

    if any(keyword in value.lower() for keyword in ("delete system", "format disk", "wipe drive")):
        return True, "contains destructive system instruction"

    return False, "ok"


def sanitize_for_logging(value: Any) -> str:
    """Convert arbitrary data to a compact, safe JSON string for logs."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)[:4000]
    except Exception:
        return normalize_text(str(value))[:4000]


__all__ = [
    "normalize_text",
    "contains_dangerous_shell_pattern",
    "validate_safe_path",
    "validate_tool_payload",
    "should_block_command",
    "sanitize_for_logging",
]
