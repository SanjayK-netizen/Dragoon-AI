"""
Dragoon — core/memory.py (Phase 3)

Provides a lightweight SQLite-backed memory layer for context retrieval and
state updates. This is intentionally narrow: it stores command history and a
key/value context table, and exposes a bounded retrieval API used by the
question/conversation path.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


DATA_DIR = os.environ.get("DRAGOON_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data"))
DB_PATH = os.path.join(DATA_DIR, "dragoon.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                intent TEXT NOT NULL,
                outcome TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS context (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


_init_db()


def update_context(key: str, value: Any) -> None:
    """Store a key/value pair in the context table, overwriting prior values."""
    with _get_connection() as conn:
        serialized = json.dumps(value, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO context(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (str(key), serialized, _now_iso()),
        )
        conn.commit()


def _is_recent(timestamp: str, *, max_age_hours: int = 24) -> bool:
    """Return True when a timestamp is still within the configured recency window."""
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return parsed.astimezone(timezone.utc) >= cutoff


def get_context(n: int = 5) -> Dict[str, Any]:
    """Return a bounded view of recent commands plus fresh persisted context state."""
    with _get_connection() as conn:
        recent_rows = conn.execute(
            """
            SELECT raw_text, intent, outcome, ts
            FROM commands
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(0, int(n)),),
        ).fetchall()

        context_rows = conn.execute(
            "SELECT key, value, updated_at FROM context ORDER BY updated_at DESC"
        ).fetchall()

    commands = [
        {
            "raw_text": row["raw_text"],
            "intent": row["intent"],
            "outcome": row["outcome"],
            "ts": row["ts"],
        }
        for row in recent_rows
    ]

    context_map: Dict[str, Any] = {}
    for row in context_rows:
        if not _is_recent(row["updated_at"], max_age_hours=24):
            continue
        try:
            context_map[row["key"]] = json.loads(row["value"])
        except (TypeError, ValueError):
            context_map[row["key"]] = row["value"]

    return {
        "commands": commands,
        "context": context_map,
        "count": len(commands),
    }


def record_command(raw_text: str, intent: str, outcome: str) -> None:
    """Persist a command event for memory retrieval."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO commands(ts, raw_text, intent, outcome)
            VALUES (?, ?, ?, ?)
            """,
            (_now_iso(), raw_text, intent, outcome),
        )
        conn.commit()
