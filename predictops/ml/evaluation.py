"""Metrics.

Two views of the same predictions:

* **row level** -- every 10-minute sample is a case (F1, PR-AUC, ...).  This is
  the primary metric because the class balance is realistic and it is the
  number a model is actually trained against.
* **event level** -- every real failure is a case.  This is what a maintenance
  planner cares about: did we catch the event at all, how much warning did we
  get, and how many false alarms did the crew have to walk out to.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import HORIZON_HOURS


@dataclass
class RowMetrics:
    f1: float
    precision: float
    recall: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    threshold: float
    n: int
    n_positive: int

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class EventMetrics:
    n_events: int
    detected: int
    detection_rate: float
    mean_early_warning_h: float
    median_early_warning_h: float
    false_alarms_per_machine_day: float
    detection_by_type: dict

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def row_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                threshold: float) -> RowMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = (y_prob >= threshold).astype(int)

    neg = y_true == 0
    fpr = float((y_hat[neg] == 1).mean()) if neg.any() else 0.0
    both = len(np.unique(y_true)) > 1
    return RowMetrics(
        f1=float(f1_score(y_true, y_hat, zero_division=0)),
        precision=float(precision_score(y_true, y_hat, zero_division=0)),
        recall=float(recall_score(y_true, y_hat, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        pr_auc=float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        false_positive_rate=fpr,
        threshold=float(threshold),
        n=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   grid: np.ndarray | None = None) -> float:
    """Choose the operating point that maximises F1 -- on validation data only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0:
        return 0.5
    if grid is None:
        qs = np.unique(np.quantile(y_prob, np.linspace(0.50, 0.9995, 220)))
        grid = np.unique(np.concatenate([qs, np.linspace(0.05, 0.95, 37)]))
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def event_metrics(machines: np.ndarray, timestamps: np.ndarray,
                  y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                  failures: pd.DataFrame,
                  horizon_hours: int = HORIZON_HOURS) -> EventMetrics:
    """Score the failures that fall inside the evaluated period.

    An event counts as detected if the model raised the alarm at any point in
    the horizon before it; early warning is measured from the *first* such
    alarm, which is what a planner actually gets to act on.
    """
    alerts = np.asarray(y_prob, dtype=float) >= threshold
    ts = pd.to_datetime(pd.Series(timestamps))
    frame = pd.DataFrame({"machine_id": machines, "timestamp": ts,
                          "alert": alerts, "label": np.asarray(y_true)})

    if len(frame) == 0:
        return EventMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, {})

    lo, hi = frame["timestamp"].min(), frame["timestamp"].max()
    f = failures.copy()
    f["failure_time"] = pd.to_datetime(f["failure_time"])
    window = pd.Timedelta(hours=horizon_hours)
    # only events whose whole warning window lies in the evaluated period
    f = f[(f["failure_time"] - window >= lo) & (f["failure_time"] <= hi)]

    detected, leads = 0, []
    by_type: dict[str, list[int]] = {}
    for _, ev in f.iterrows():
        sub = frame[(frame.machine_id == ev.machine_id)
                    & (frame.timestamp < ev.failure_time)
                    & (frame.timestamp >= ev.failure_time - window)]
        hit = sub[sub.alert]
        got = int(len(hit) > 0)
        detected += got
        if got:
            lead = (ev.failure_time - hit["timestamp"].min()).total_seconds() / 3600.0
            leads.append(lead)
        by_type.setdefault(ev.failure_type, []).append(got)

    # false alarms: alerts on rows with no failure ahead, per machine-day
    fp = int((frame["alert"] & (frame["label"] == 0)).sum())
    span_days = max((hi - lo).total_seconds() / 86400.0, 1e-9)
    n_machines = max(frame["machine_id"].nunique(), 1)
    fa_rate = fp / (span_days * n_machines)

    n_events = int(len(f))
    return EventMetrics(
        n_events=n_events,
        detected=detected,
        detection_rate=detected / n_events if n_events else 0.0,
        mean_early_warning_h=float(np.mean(leads)) if leads else 0.0,
        median_early_warning_h=float(np.median(leads)) if leads else 0.0,
        false_alarms_per_machine_day=float(fa_rate),
        detection_by_type={k: {"n": len(v), "detected": int(sum(v)),
                               "rate": round(float(np.mean(v)), 4)}
                           for k, v in sorted(by_type.items())},
    )


def full_report(machines: np.ndarray, timestamps: np.ndarray,
                y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                failures: pd.DataFrame) -> dict:
    return {
        "row": row_metrics(y_true, y_prob, threshold).to_dict(),
        "event": event_metrics(machines, timestamps, y_true, y_prob,
                               threshold, failures).to_dict(),
    }
