"""
Dragoon — core/agent_loop.py (Phase 4)

A minimal, deterministic tool loop that follows the build-plan state machine:
PARSE -> SELECT_TOOL -> EXECUTE -> VERIFY -> RESPOND

This is intentionally conservative and intentionally small. It does not try to
launch destructive tools; it starts with low-risk functions only.
"""

import json
import logging
import re
from enum import Enum
from typing import Any, Callable, Dict, Optional

from core.hardening import should_block_command, validate_tool_payload
from Tools.register import REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger("dragoon")

class AgentState(Enum):
    PARSE = "PARSE"
    SELECT_TOOL = "SELECT_TOOL"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    RESPOND = "RESPOND"


VALID_STATES = [state.value for state in AgentState]


def _default_registry() -> Dict[str, Callable[..., Any]]:
    return REGISTRY.copy()


def _coerce_arg(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        if stripped.lower() in {"none", "null"}:
            return None
        try:
            if re.fullmatch(r"[-+]?\d+", stripped):
                return int(stripped)
            if re.fullmatch(r"[-+]?\d*\.\d+", stripped):
                return float(stripped)
        except Exception:
            pass
    return value


def _parse_inline_args(raw_args: str) -> Dict[str, Any]:
    if not raw_args or not raw_args.strip():
        return {}

    args: Dict[str, Any] = {}
    for piece in re.split(r"\s+(?:and\s+)?", raw_args.strip()):
        if not piece:
            continue
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        args[key.strip()] = _coerce_arg(value.strip().strip("\"'"))
    return args


def _parse_tool_call(text: str, tool_registry: Optional[Dict[str, Callable[..., Any]]] = None) -> Dict[str, Any]:
    registry = tool_registry or _default_registry()
    blocked, reason = should_block_command(text)
    if blocked:
        raise ValueError(f"unsafe command blocked: {reason}")

    lowered = text.lower().strip()

    addition_match = re.match(r"^(?:add|sum)\s+(.+?)\s+(?:and|to|plus)\s+(.+)$", text.strip(), re.IGNORECASE)
    if addition_match and all(re.search(r"\d", part) for part in addition_match.groups()):
        return {
            "tool": "calculate",
            "args": {"value": f"{addition_match.group(1)} + {addition_match.group(2)}"},
        }

    if not text.strip().startswith("Use ") and (
        "calculate" in lowered or any(op in lowered for op in ["+", "-", "*", "/", "%"])
    ):
        expr = text
        for prefix in ["calculate ", "compute ", "what is "]:
            if expr.lower().startswith(prefix):
                expr = expr[len(prefix):]
                break
        expr = re.sub(r"\bplus\b", "+", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(minus|less)\b", "-", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(times|multiplied by)\b", "*", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\b(divided by|over)\b", "/", expr, flags=re.IGNORECASE)
        return {"tool": "calculate", "args": {"value": expr.strip()}}

    reminder_match = re.match(r"^(?:set|create) a reminder\s+(?:to\s+)?(.+)$", text.strip(), re.IGNORECASE)
    if reminder_match:
        reminder_text = reminder_match.group(1).strip()
        due_match = re.match(r"^(.+?)\s+at\s+(.+)$", reminder_text, re.IGNORECASE)
        if due_match:
            reminder_text, due = due_match.groups()
        else:
            for_prefix = re.match(r"^for\s+(.+)$", reminder_text, re.IGNORECASE)
            if for_prefix:
                reminder_text, due = "Reminder", for_prefix.group(1)
            else:
                due = "unspecified time"
        return {"tool": "set_reminder", "args": {"text": reminder_text.strip(), "due": due.strip()}}

    file_match = re.match(r"^(?:open|read) (?:the )?(?:local )?file\s+(.+)$", text.strip(), re.IGNORECASE)
    if file_match:
        return {"tool": "open_local_file", "args": {"path": file_match.group(1).strip()}}

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

        remainder = text[len("Use "):].strip()
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?:with\s+)?(.+)?$", remainder)
        if match:
            tool_name = match.group(1)
            raw_args = (match.group(2) or "").strip()
            if tool_name in registry:
                return {"tool": tool_name, "args": _parse_inline_args(raw_args)}

    raise ValueError(f"No supported tool match for input: {text!r}")


def _verify_tool_result(tool: str, result: Any) -> bool:
    if tool == "calculate":
        return isinstance(result, (int, float))
    if tool == "get_time" or tool == "open_local_file":
        return isinstance(result, str) and len(result) > 0
    if tool == "set_reminder":
        return isinstance(result, dict) and result.get("ok") is True and result.get("id") is not None
    return result is not None


def _validate_args(tool: str, args: Any) -> None:
    if not isinstance(args, dict):
        return
    schema = TOOL_SCHEMAS.get(tool)
    if schema is None:
        return
    keys = set(args)
    missing = schema["required"] - keys
    unknown = keys - schema["required"] - schema["optional"]
    if missing or unknown:
        raise ValueError(f"invalid arguments for {tool}: missing={sorted(missing)}, unknown={sorted(unknown)}")


def run_agent_loop(text: str, tool_registry: Optional[Dict[str, Callable[..., Any]]] = None) -> str:
    """Execute a minimal tool loop. Returns a human-readable result string."""
    registry = tool_registry or _default_registry()
    state = "PARSE"
    last_error = None

    try:
        for state in AgentState:
            if state is AgentState.PARSE:
                parsed = _parse_tool_call(text, registry)
                tool_name = parsed["tool"]
                args = parsed["args"]
            elif state is AgentState.SELECT_TOOL:
                if tool_name not in registry:
                    raise ValueError(f"tool {tool_name!r} is not in the registry")
                _validate_args(tool_name, args)
                args = validate_tool_payload(tool_name, args)
            elif state is AgentState.EXECUTE:
                try:
                    if isinstance(args, dict):
                        result = registry[tool_name](**args)
                    else:
                        result = registry[tool_name](args)
                except Exception as exc:
                    last_error = str(exc)
                    raise
            elif state is AgentState.VERIFY:
                if not _verify_tool_result(tool_name, result):
                    raise ValueError(f"verification failed for tool {tool_name!r}")
            elif state is AgentState.RESPOND:
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
            _validate_args(tool_name, args)
            args = validate_tool_payload(tool_name, args)
            if isinstance(args, dict):
                result = registry[tool_name](**args)
            else:
                result = registry[tool_name](args)
            if not _verify_tool_result(tool_name, result):
                raise ValueError(f"verification failed for tool {tool_name!r}")
            return str(result)
        except Exception as retry_exc:
            logger.exception("run_agent_loop failed on retry for text=%r", text)
            return f"I hit a tool error while executing that command: {retry_exc}"
