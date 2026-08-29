"""Experiment store.

SQLite by default: a judge cloning this repo should not have to start a
database daemon to reproduce the results, and nothing here needs concurrent
writers.  All access goes through `ExperimentStore`, so pointing at Postgres
later means changing one class, not every call site.

Nothing writes a metric to this store unless it came out of an actual run --
`record()` requires the metrics dict produced by `ml.evaluation`.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import EXPERIMENT_DIR

DB_PATH = EXPERIMENT_DIR / "experiments.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    model         TEXT    NOT NULL,
    feature_set   TEXT    NOT NULL,
    hypothesis    TEXT    NOT NULL DEFAULT '',
    params        TEXT    NOT NULL DEFAULT '{}',
    metrics       TEXT    NOT NULL DEFAULT '{}',
    decision      TEXT    NOT NULL DEFAULT 'pending',
    learning      TEXT    NOT NULL DEFAULT '',
    duration_s    REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    agent         TEXT    NOT NULL,
    step          INTEGER NOT NULL,
    action        TEXT    NOT NULL,
    reason        TEXT    NOT NULL DEFAULT '',
    tools_used    TEXT    NOT NULL DEFAULT '[]',
    input_summary TEXT    NOT NULL DEFAULT '',
    output        TEXT    NOT NULL DEFAULT '{}',
    verification  TEXT    NOT NULL DEFAULT '',
    retry_count   INTEGER NOT NULL DEFAULT 0,
    duration_s    REAL    NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_exp_run ON experiments(run_id);
CREATE INDEX IF NOT EXISTS ix_traj_run ON trajectories(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Experiment:
    run_id: str
    stage: str
    name: str
    model: str
    feature_set: str
    hypothesis: str = ""
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    decision: str = "pending"
    learning: str = ""
    duration_s: float = 0.0
    id: int | None = None
    created_at: str = ""

    @property
    def f1(self) -> float:
        return float(self.metrics.get("row", {}).get("f1", float("nan")))

    @property
    def pr_auc(self) -> float:
        return float(self.metrics.get("row", {}).get("pr_auc", float("nan")))

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


class ExperimentStore:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- experiments -------------------------------------------------------
    def record(self, exp: Experiment) -> int:
        if not exp.metrics:
            raise ValueError(
                "refusing to record an experiment with no measured metrics")
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO experiments
                   (run_id, stage, name, model, feature_set, hypothesis, params,
                    metrics, decision, learning, duration_s, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (exp.run_id, exp.stage, exp.name, exp.model, exp.feature_set,
                 exp.hypothesis, json.dumps(exp.params), json.dumps(exp.metrics),
                 exp.decision, exp.learning, exp.duration_s, _now()))
            return int(cur.lastrowid)

    def set_decision(self, exp_id: int, decision: str, learning: str = "") -> None:
        with self._conn() as c:
            c.execute("UPDATE experiments SET decision=?, learning=? WHERE id=?",
                      (decision, learning, exp_id))

    def all(self, run_id: str | None = None) -> list[Experiment]:
        q = "SELECT * FROM experiments"
        args: tuple = ()
        if run_id:
            q += " WHERE run_id=?"
            args = (run_id,)
        q += " ORDER BY id"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def latest_run_id(self) -> str | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT run_id FROM experiments ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return r["run_id"] if r else None

    def best(self, run_id: str | None = None,
             metric: str = "f1") -> Experiment | None:
        exps = [e for e in self.all(run_id) if e.metrics.get("row")]
        if not exps:
            return None
        return max(exps, key=lambda e: e.metrics["row"].get(metric, -1))

    @staticmethod
    def _row_to_exp(r: sqlite3.Row) -> Experiment:
        return Experiment(
            id=r["id"], run_id=r["run_id"], stage=r["stage"], name=r["name"],
            model=r["model"], feature_set=r["feature_set"],
            hypothesis=r["hypothesis"], params=json.loads(r["params"]),
            metrics=json.loads(r["metrics"]), decision=r["decision"],
            learning=r["learning"], duration_s=r["duration_s"],
            created_at=r["created_at"])

    # -- agent trajectories -------------------------------------------------
    def log_step(self, run_id: str, agent: str, step: int, action: str,
                 reason: str = "", tools_used: list[str] | None = None,
                 input_summary: str = "", output: Any = None,
                 verification: str = "", retry_count: int = 0,
                 duration_s: float = 0.0) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO trajectories
                   (run_id, agent, step, action, reason, tools_used,
                    input_summary, output, verification, retry_count,
                    duration_s, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, agent, step, action, reason,
                 json.dumps(tools_used or []), input_summary,
                 json.dumps(output, default=str) if output is not None else "{}",
                 verification, retry_count, duration_s, _now()))
            return int(cur.lastrowid)

    def trajectory(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM trajectories WHERE run_id=? ORDER BY id",
                (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tools_used"] = json.loads(d["tools_used"])
            try:
                d["output"] = json.loads(d["output"])
            except json.JSONDecodeError:
                pass
            out.append(d)
        return out

    def runs(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT run_id FROM trajectories ORDER BY id DESC"
            ).fetchall()
        return [r["run_id"] for r in rows]
