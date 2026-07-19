"""Minimal persistent job queue used by the web UI and one local worker."""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import sqlite3
import uuid
from typing import Any, Iterable, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "jobs.sqlite3"
DEFAULT_LOG_DIR = ROOT / "data" / "job_logs"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


class JobStore:
    """SQLite-backed queue with explicit, small state transitions."""

    def __init__(self, db_path: Optional[str] = None, log_dir: Optional[str] = None):
        self.db_path = pathlib.Path(db_path or os.environ.get("MRL_JOBS_DB", DEFAULT_DB_PATH))
        self.log_dir = pathlib.Path(log_dir or os.environ.get("MRL_JOB_LOG_DIR", DEFAULT_LOG_DIR))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    command_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    return_code INTEGER,
                    pid INTEGER,
                    log_path TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)"
            )

    def enqueue(self, kind: str, command: Iterable[str]) -> dict[str, Any]:
        argv = [str(value) for value in command]
        if not argv:
            raise ValueError("Job command must not be empty")
        job_id = uuid.uuid4().hex
        log_path = self.log_dir / f"{job_id}.log"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(id, kind, command_json, status, created_at, log_path)
                VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (job_id, kind, json.dumps(argv, ensure_ascii=False), _utc_now(), str(log_path)),
            )
        return self.get(job_id)

    def claim_next(self) -> Optional[dict[str, Any]]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE id=? AND status='queued'",
                (_utc_now(), row["id"]),
            )
            connection.commit()
            return self.get(row["id"])
        finally:
            connection.close()

    def set_pid(self, job_id: str, pid: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET pid=? WHERE id=?", (pid, job_id))

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] == "queued":
                connection.execute(
                    "UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
                    (_utc_now(), job_id),
                )
            elif row["status"] == "running":
                connection.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
        return self.get(job_id)

    def is_cancelling(self, job_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["status"] == "cancelling")

    def finish(self, job_id: str, return_code: int, error: Optional[str] = None) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] == "cancelling":
                status = "cancelled"
            else:
                status = "completed" if return_code == 0 else "failed"
            connection.execute(
                """
                UPDATE jobs
                SET status=?, finished_at=?, return_code=?, pid=NULL, error=?
                WHERE id=?
                """,
                (status, _utc_now(), return_code, error, job_id),
            )
        return self.get(job_id)

    def mark_interrupted(self, reason: str = "worker restarted") -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status='interrupted', finished_at=?, pid=NULL, error=?
                WHERE status IN ('running', 'cancelling')
                """,
                (_utc_now(), reason),
            )
            return cursor.rowcount

    def get(self, job_id: str, include_command: bool = False) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._normalise(row, include_command=include_command)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [self._normalise(row) for row in rows]

    def read_log(self, job_id: str, max_chars: int = 20000) -> str:
        job = self.get(job_id)
        path = pathlib.Path(job["log_path"])
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - max_chars))
            return file.read()

    @staticmethod
    def _normalise(row: sqlite3.Row, include_command: bool = False) -> dict[str, Any]:
        result = {
            "id": row["id"],
            "kind": row["kind"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "return_code": row["return_code"],
            "pid": row["pid"],
            "log_path": row["log_path"],
            "error": row["error"],
        }
        if include_command:
            result["command"] = json.loads(row["command_json"])
        return result
