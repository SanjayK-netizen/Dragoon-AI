"""SQLite-backed, non-destructive reminder tool."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict


def _database_path() -> str:
	data_dir = os.environ.get(
		"DRAGOON_DATA_DIR",
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data"),
	)
	os.makedirs(data_dir, exist_ok=True)
	return os.path.join(data_dir, "dragoon.db")


def _init_table(conn: sqlite3.Connection) -> None:
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS reminders (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			text TEXT NOT NULL,
			due TEXT NOT NULL,
			created_at TEXT NOT NULL
		)
		"""
	)


def set_reminder(text: str, due: str) -> Dict[str, Any]:
	if not text or not text.strip() or not due or not due.strip():
		raise ValueError("reminder text and due time are required")

	with sqlite3.connect(_database_path()) as conn:
		_init_table(conn)
		cursor = conn.execute(
			"INSERT INTO reminders(text, due, created_at) VALUES (?, ?, ?)",
			(text.strip(), due.strip(), datetime.now(timezone.utc).isoformat()),
		)
		reminder_id = cursor.lastrowid
		conn.commit()

	return {"ok": True, "id": reminder_id, "text": text.strip(), "due": due.strip()}
