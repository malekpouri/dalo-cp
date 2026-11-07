"""Run registry in SQLite (WAL mode) under runs/registry.sqlite.

States: queued -> running -> completed | failed | interrupted.
A ``running`` entry whose heartbeat is older than ``STALE_AFTER_S`` belongs to
a process that died (for example, the machine was shut down) and is marked
``interrupted`` by :func:`Registry.mark_stale`.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from ..utils.paths import RUNS_DIR

STALE_AFTER_S = 600
HEARTBEAT_EVERY_S = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    experiment   TEXT,
    method       TEXT,
    dataset      TEXT,
    seed         INTEGER,
    config_hash  TEXT,
    git_commit   TEXT,
    status       TEXT,
    created_at   REAL,
    started_at   REAL,
    ended_at     REAL,
    heartbeat    REAL,
    progress     REAL,
    host         TEXT,
    pid          INTEGER,
    run_dir      TEXT,
    error        TEXT,
    extra        TEXT
);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS runs_experiment ON runs(experiment);
"""


class Registry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else RUNS_DIR / "registry.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        finally:
            conn.close()

    def register(self, run_id: str, *, experiment: str, method: str, dataset: str, seed: int,
                 config_hash: str, git_commit: str, run_dir: str) -> None:
        """Insert a queued run; leaves an existing entry untouched."""
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO runs (run_id, experiment, method, dataset, seed, config_hash,"
                " git_commit, status, created_at, run_dir) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, experiment, method, dataset, seed, config_hash, git_commit, "queued",
                 time.time(), run_dir),
            )

    def set_running(self, run_id: str, git_commit: str) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status='running', started_at=COALESCE(started_at, ?), heartbeat=?,"
                " host=?, pid=?, git_commit=?, error=NULL WHERE run_id=?",
                (now, now, socket.gethostname(), os.getpid(), git_commit, run_id),
            )

    def heartbeat(self, run_id: str, progress: float) -> None:
        with self._conn() as c:
            c.execute("UPDATE runs SET heartbeat=?, progress=? WHERE run_id=?", (time.time(), progress, run_id))

    def finish(self, run_id: str, status: str, error: str | None = None, extra: dict | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status=?, ended_at=?, heartbeat=?, error=?, extra=COALESCE(?, extra),"
                " progress=CASE WHEN ?='completed' THEN 1.0 ELSE progress END WHERE run_id=?",
                (status, time.time(), time.time(), error, json.dumps(extra) if extra else None, status, run_id),
            )

    def mark_stale(self, stale_after_s: float = STALE_AFTER_S) -> list[str]:
        cutoff = time.time() - stale_after_s
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id FROM runs WHERE status='running' AND COALESCE(heartbeat, 0) < ?", (cutoff,)
            ).fetchall()
            ids = [r["run_id"] for r in rows]
            c.executemany(
                "UPDATE runs SET status='interrupted', error='stale heartbeat (process ended)' WHERE run_id=?",
                [(i,) for i in ids],
            )
        return ids

    def get(self, run_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list(self, status: str | None = None, experiment: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if experiment:
            q += " AND experiment=?"
            args.append(experiment)
        q += " ORDER BY created_at"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM runs GROUP BY status")}
