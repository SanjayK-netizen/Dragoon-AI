"""
Dragoon — core/agent_loop.py (Phase 4)

A minimal, deterministic tool loop that follows the build-plan state machine:
PARSE -> SELECT_TOOL -> EXECUTE -> VERIFY -> RESPOND

This is intentionally conservative and intentionally small. It does not try to
launch destructive tools; it starts with low-risk functions only.
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("dragoon")

VALID_STATES = ["PARSE", "SELECT_TOOL", "EXECUTE", "VERIFY", "RESPOND"]


def _safe_math(expr: str) -> float:
    if not expr or not all(ch.isdigit() or ch in " +-*/().%" for ch in expr):
        raise ValueError("unsafe or malformed math expression")
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"math evaluation failed: {exc}") from exc
    return float(value)


def _default_registry() -> Dict[str, Callable[..., Any]]:
    return {
        "calculate": lambda value: _safe_math(str(value)),
        "get_time": lambda value=None: "The current time is unavailable without the system clock wrapper.",
    }


def _parse_tool_call(text: str, tool_registry: Optional[Dict[str, Callable[..., Any]]] = None) -> Dict[str, Any]:
    registry = tool_registry or _default_registry()
    lowered = text.lower().strip()

    if "calculate" in lowered or any(op in lowered for op in ["+", "-", "*", "/", "%"]):
        expr = text
        for prefix in ["calculate ", "compute ", "what is "]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):]
                break
        return {"tool": "calculate", "args": {"value": expr.strip()}}

    if "time" in lowered and ("what time" in lowered or "current time" in lowered):
        return {"tool": "get_time", "args": {}}

    if text.strip().startswith("Use "):
        try:
            payload = json.loads(text[len("Use "):])
            if isinstance(payload, dict) and "tool" in payload:
                tool = payload["tool"]
                args = payload.get("args", {})
                if tool in registry:
                    return {"tool": tool, "args": args}
        except Exception:
            pass

    raise ValueError(f"No supported tool match for input: {text!r}")


def _verify_tool_result(tool: str, result: Any) -> bool:
    if tool == "calculate":
        return isinstance(result, (int, float))
    if tool == "get_time":
        return isinstance(result, str) and len(result) > 0
    return result is not None


def run_agent_loop(text: str, tool_registry: Optional[Dict[str, Callable[..., Any]]] = None) -> str:
    """Execute a minimal tool loop. Returns a human-readable result string."""
    registry = tool_registry or _default_registry()
    state = "PARSE"
    last_error = None

    try:
        for state in ["PARSE", "SELECT_TOOL", "EXECUTE", "VERIFY", "RESPOND"]:
            if state == "PARSE":
                parsed = _parse_tool_call(text, registry)
                tool_name = parsed["tool"]
                args = parsed["args"]
            elif state == "SELECT_TOOL":
                if tool_name not in registry:
                    raise ValueError(f"tool {tool_name!r} is not in the registry")
            elif state == "EXECUTE":
                try:
                    result = registry[tool_name](**args) if isinstance(args, dict) else registry[tool_name](args)
                except Exception as exc:
                    last_error = str(exc)
                    raise
            elif state == "VERIFY":
                if not _verify_tool_result(tool_name, result):
                    raise ValueError(f"verification failed for tool {tool_name!r}")
            elif state == "RESPOND":
                return str(result)

        return str(result)
    except Exception as exc:  # retry once on a transient tool failure
        if last_error is None:
            logger.exception("run_agent_loop failed before execution for text=%r", text)
            return f"I couldn't safely execute that command: {exc}"

        try:
            parsed = _parse_tool_call(text, registry)
            tool_name = parsed["tool"]
            args = parsed["args"]
            result = registry[tool_name](**args) if isinstance(args, dict) else registry[tool_name](args)
            if not _verify_tool_result(tool_name, result):
                raise ValueError(f"verification failed for tool {tool_name!r}")
            return str(result)
        except Exception as retry_exc:
            logger.exception("run_agent_loop failed on retry for text=%r", text)
            return f"I hit a tool error while executing that command: {retry_exc}"
