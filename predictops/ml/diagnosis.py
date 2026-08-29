"""Failure type, time-to-failure and calibrated confidence.

The risk model answers *whether*.  These three answer *what*, *when* and
*how sure* -- and each is a fitted model, not a heuristic the agent made up:

* `FailureTypeClassifier`  multi-class over the engineered features, trained
  only on rows inside a real warning window.
* `TimeToFailureRegressor` hours until the failure, same rows.
* `ReliabilityCurve`       empirical precision by score bin, measured on
  validation.  This is what "confidence 0.91" means here: *of validation cases
  scored this high, 91% were genuinely followed by a failure.*  It is a
  measured frequency, not the raw sigmoid output relabelled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HORIZON_HOURS
from ..data.schemas import FAILURE_MODES


@dataclass
class FailureTypeClassifier:
    booster: object = None
    classes: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    train_support: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (class index, full probability matrix)."""
        import xgboost as xgb
        p = self.booster.predict(xgb.DMatrix(x, feature_names=self.columns))
        if p.ndim == 1:                      # binary edge case
            p = np.vstack([1 - p, p]).T
        return p.argmax(axis=1), p

    def label(self, x: np.ndarray) -> list[dict]:
        idx, p = self.predict(x)
        out = []
        for i, row in enumerate(p):
            order = np.argsort(-row)
            out.append({
                "failure_type": self.classes[int(idx[i])],
                "confidence": float(row[int(idx[i])]),
                "alternatives": [
                    {"failure_type": self.classes[int(j)],
                     "probability": round(float(row[int(j)]), 4)}
                    for j in order[1:3]],
            })
        return out


@dataclass
class TimeToFailureRegressor:
    booster: object = None
    columns: list[str] = field(default_factory=list)
    residual_std: float = 1.5

    def predict(self, x: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        p = self.booster.predict(xgb.DMatrix(x, feature_names=self.columns))
        return np.clip(p, 0.2, HORIZON_HOURS)

    def window(self, x: np.ndarray) -> list[dict]:
        """A window, not a point estimate -- the residual spread is measured."""
        p = self.predict(x)
        half = max(self.residual_std, 0.5)
        return [{"eta_hours": round(float(v), 2),
                 "window_low_h": round(float(max(v - half, 0.1)), 2),
                 "window_high_h": round(float(min(v + half, HORIZON_HOURS)), 2)}
                for v in p]


@dataclass
class ReliabilityCurve:
    """Measured tail precision, from validation.

    `confidence(p)` answers: *of validation cases scored at least this high,
    what fraction were genuinely followed by a failure inside the horizon?*

    Tail precision, not per-bin precision. An earlier version binned scores by
    quantile, which was actively misleading here: with ~1.4% positives, the top
    quantile bin spans the top 8% of all rows, so a 0.996 score reported ~13%
    confidence -- diluted by the thousands of ordinary rows sharing its bin.
    Scoring the tail instead gives the number an operator actually wants: the
    hit rate if they alerted at this level.
    """

    edges: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    support: list[int] = field(default_factory=list)
    min_support: int = 20

    @classmethod
    def fit(cls, y: np.ndarray, prob: np.ndarray,
            min_support: int = 20) -> "ReliabilityCurve":
        y = np.asarray(y).astype(int)
        prob = np.asarray(prob, dtype=float)
        # Fine resolution at the top, where the decisions are made.
        grid = np.unique(np.concatenate([
            np.linspace(0.0, 0.9, 19), np.array([0.92, 0.94, 0.96, 0.97, 0.98,
                                                 0.985, 0.99, 0.995, 0.999])]))
        prec, sup, kept = [], [], []
        for edge in grid:
            m = prob >= edge
            n = int(m.sum())
            if n < min_support:
                break          # beyond here the tail is too thin to measure
            kept.append(float(edge))
            sup.append(n)
            prec.append(float(y[m].mean()))
        if not kept:
            kept, prec, sup = [0.0], [float(y.mean()) if len(y) else 0.0], [len(y)]
        return cls(edges=kept, precision=prec, support=sup,
                   min_support=min_support)

    def confidence(self, prob: float) -> float:
        """Precision of the highest measured tail this score reaches."""
        if not self.edges:
            return float(prob)
        i = int(np.clip(np.searchsorted(self.edges, prob, side="right") - 1,
                        0, len(self.precision) - 1))
        return float(self.precision[i])

    def to_dict(self) -> dict:
        return {"edges": self.edges, "precision": self.precision,
                "support": self.support}

    @classmethod
    def from_dict(cls, d: dict) -> "ReliabilityCurve":
        return cls(d["edges"], d["precision"], d["support"])


# --------------------------------------------------------------------------
def _warning_window_rows(df: pd.DataFrame, split: str):
    """Rows inside a real warning window -- the only place a type/ETA exists."""
    d = df[(df["split"] == split) & (df["label"] == 1)
           & (df["is_downtime"] == 0) & (df["sensor_dropout"] == 0)]
    return d[d["horizon_failure_type"].astype(str) != ""]


def train_failure_type_classifier(df: pd.DataFrame, columns: list[str],
                                  seed: int = 42) -> FailureTypeClassifier:
    import xgboost as xgb

    tr = _warning_window_rows(df, "train")
    va = _warning_window_rows(df, "val")
    classes = sorted(tr["horizon_failure_type"].astype(str).unique())
    idx = {c: i for i, c in enumerate(classes)}

    x_tr = tr[columns].to_numpy(dtype=np.float32)
    y_tr = tr["horizon_failure_type"].astype(str).map(idx).to_numpy()
    x_va = va[columns].to_numpy(dtype=np.float32)
    y_va = va["horizon_failure_type"].astype(str).map(idx).fillna(0).to_numpy()

    dtr = xgb.DMatrix(x_tr, label=y_tr, feature_names=columns)
    evals = [(dtr, "train")]
    if len(va):
        evals.append((xgb.DMatrix(x_va, label=y_va, feature_names=columns), "val"))
    params = {"objective": "multi:softprob", "num_class": len(classes),
              "eval_metric": "mlogloss", "eta": 0.08, "max_depth": 5,
              "subsample": 0.85, "colsample_bytree": 0.7, "reg_lambda": 2.0,
              "seed": seed, "nthread": 0}
    booster = xgb.train(params, dtr, num_boost_round=300, evals=evals,
                        early_stopping_rounds=30, verbose_eval=False)
    return FailureTypeClassifier(
        booster=booster, classes=classes, columns=list(columns),
        train_support=tr["horizon_failure_type"].value_counts().to_dict())


def train_time_to_failure_regressor(df: pd.DataFrame, columns: list[str],
                                    seed: int = 42) -> TimeToFailureRegressor:
    import xgboost as xgb

    tr = _warning_window_rows(df, "train")
    va = _warning_window_rows(df, "val")
    x_tr = tr[columns].to_numpy(dtype=np.float32)
    y_tr = tr["time_to_failure_h"].to_numpy(dtype=np.float32)
    dtr = xgb.DMatrix(x_tr, label=y_tr, feature_names=columns)
    evals = [(dtr, "train")]
    if len(va):
        evals.append((xgb.DMatrix(va[columns].to_numpy(dtype=np.float32),
                                  label=va["time_to_failure_h"].to_numpy(
                                      dtype=np.float32),
                                  feature_names=columns), "val"))
    params = {"objective": "reg:squarederror", "eval_metric": "rmse",
              "eta": 0.08, "max_depth": 5, "subsample": 0.85,
              "colsample_bytree": 0.7, "reg_lambda": 2.0, "seed": seed,
              "nthread": 0}
    booster = xgb.train(params, dtr, num_boost_round=300, evals=evals,
                        early_stopping_rounds=30, verbose_eval=False)

    reg = TimeToFailureRegressor(booster=booster, columns=list(columns))
    if len(va):
        pred = reg.predict(va[columns].to_numpy(dtype=np.float32))
        resid = va["time_to_failure_h"].to_numpy(dtype=float) - pred
        reg.residual_std = float(np.nanstd(resid))
    return reg


# The evidence layer reports ambient-corrected and normalised channels, while
# failure signatures are written in plain sensor names.  Without this mapping a
# correct diagnosis scores as unsupported purely because the two vocabularies
# disagree -- `temp_excess` is the temperature signal, stated better.
SIGNATURE_ALIASES = {
    "temp_excess": "temperature",
    "rpm_instability_1h": "rpm_instability",
    "vib_per_load": "vibration",
    "cur_per_load": "current",
}


def canonical_channel(channel: str) -> str:
    return SIGNATURE_ALIASES.get(channel, channel)


def observed_directions(evidence: list) -> dict:
    """Map evidence items onto signature vocabulary, keeping real movements.

    A 'flat' reading (e.g. load held steady) is context, not a direction, so it
    must not overwrite a real movement on the same canonical channel.
    """
    out: dict = {}
    for e in evidence:
        ch = canonical_channel(e.get("channel", ""))
        direction = e.get("direction")
        if direction in ("up", "down") or ch not in out:
            out[ch] = direction
    return out


def expected_signature(failure_type: str) -> dict:
    """The channel moves a domain expert associates with a failure mode.

    Used by the investigator to score hypotheses, and by the verifier to check
    that the evidence actually matches the diagnosis it is offered for.
    """
    m = FAILURE_MODES.get(failure_type)
    if m is None:
        return {}
    sig = {}
    if m.vib_mult:
        sig["vibration"] = "up" if m.vib_mult > 0 else "down"
    if m.temp_add:
        sig["temperature"] = "up" if m.temp_add > 0 else "down"
    if m.current_mult:
        sig["current"] = "up" if m.current_mult > 0 else "down"
    if m.pressure_mult:
        sig["pressure"] = "up" if m.pressure_mult > 0 else "down"
    if m.voltage_mult:
        sig["voltage"] = "up" if m.voltage_mult > 0 else "down"
    if m.rpm_jitter_mult >= 1.5:
        sig["rpm_instability"] = "up"
    return sig
