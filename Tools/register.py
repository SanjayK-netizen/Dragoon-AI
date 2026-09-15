"""Low-risk tool registry for the Phase 4 agent loop."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

from Tools.reminder import set_reminder


def _safe_math(expr: str) -> float:
    if not expr or not all(ch.isdigit() or ch in " +-*/().%" for ch in expr):
        raise ValueError("unsafe or malformed math expression")
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"math evaluation failed: {exc}") from exc
    return float(value)


def get_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def calculate(value: Any) -> float:
    return _safe_math(str(value))


def open_local_file(path: str) -> str:
    root = Path.cwd().resolve()
    requested = (root / path).resolve()
    if root not in requested.parents and requested != root:
        raise ValueError("file path must remain inside the project directory")
    if not requested.is_file():
        raise FileNotFoundError(path)
    return requested.read_text(encoding="utf-8")


def build_registry() -> Dict[str, Callable[..., Any]]:
    return {
        "calculate": calculate,
        "get_time": get_time,
        "set_reminder": set_reminder,
        "open_local_file": open_local_file,
    }


TOOL_SCHEMAS = {
    "calculate": {"required": {"value"}, "optional": set()},
    "get_time": {"required": set(), "optional": set()},
    "set_reminder": {"required": {"text", "due"}, "optional": set()},
    "open_local_file": {"required": {"path"}, "optional": set()},
}


REGISTRY = build_registry()
