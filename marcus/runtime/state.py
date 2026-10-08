from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeStore:
    """Durable sessions and job lifecycle; no credentials or hidden reasoning."""

    def __init__(self, path: Path = Path("data/runtime.sqlite")) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS turns (id TEXT PRIMARY KEY, session_id TEXT, user_text TEXT, reply TEXT, created_at TEXT);
                CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, created_at);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, session_id TEXT, status TEXT, created_at TEXT,
                    updated_at TEXT, result TEXT, error TEXT);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def ensure_session(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", session_id):
            raise ValueError("Session id must be 1–64 letters, digits, hyphens, or underscores")
        timestamp = now()
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?)", (session_id, timestamp, timestamp))

    def history(self, session_id: str, *, limit: int = 6) -> list[dict[str, str]]:
        self.ensure_session(session_id)
        with self.connect() as connection:
            rows = connection.execute("SELECT user_text, reply FROM turns WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                                      (session_id, limit)).fetchall()
        messages = []
        for row in reversed(rows):
            messages.extend([{"role": "user", "content": row["user_text"]},
                             {"role": "assistant", "content": row["reply"]}])
        return messages

    def save_turn(self, session_id: str, turn_id: str, user_text: str, reply: str) -> None:
        self.ensure_session(session_id)
        timestamp = now()
        with self.connect() as connection:
            connection.execute("INSERT INTO turns VALUES (?,?,?,?,?)", (turn_id, session_id, user_text, reply, timestamp))
            connection.execute("UPDATE sessions SET updated_at=? WHERE id=?", (timestamp, session_id))

    def sessions(self) -> list[dict]:
        with self.connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 100")]

    def create_job(self, session_id: str) -> str:
        self.ensure_session(session_id)
        job_id = uuid.uuid4().hex[:12]
        timestamp = now()
        with self.connect() as connection:
            connection.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)", (job_id, session_id, "queued", timestamp, timestamp, None, None))
        return job_id

    def update_job(self, job_id: str, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        if status not in {"queued", "running", "completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Invalid job status")
        with self.connect() as connection:
            connection.execute("UPDATE jobs SET status=?, updated_at=?, result=?, error=? WHERE id=?",
                               (status, now(), json.dumps(result) if result is not None else None, error, job_id))

    def job(self, job_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def jobs(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT id, session_id, status, created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall()
        return [dict(row) for row in rows]

    def recover(self) -> int:
        # A crashed job may already have executed tools. Never replay it silently.
        with self.connect() as connection:
            cursor = connection.execute("UPDATE jobs SET status='interrupted', updated_at=?, error='Runtime stopped before completion; inspect trace before retrying' WHERE status IN ('queued','running')", (now(),))
        return cursor.rowcount
