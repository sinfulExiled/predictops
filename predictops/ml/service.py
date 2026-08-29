"""Model service — the boundary between the agents and the ML.

The models are **instruments, not agents**. An agent asks a question; the
service answers it with a fitted model. Nothing below this line knows what an
agent is, and nothing above it knows whether the answer came from a tree or a
transformer.

That boundary is what let the research agent swap XGBoost in for the TFT
without a single change to the investigation, remediation, simulation or
verification agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOOKBACK_STEPS
from .bundle import BUNDLE_DIR, ModelBundle, window_for
from .training import static_index


@dataclass
class Prediction:
    machine_id: str
    timestamp: str
    failure_probability: float
    alert: bool
    investigate: bool
    risk_band: str
    alert_threshold: float
    investigate_threshold: float
    confidence: float
    failure_type: str | None
    failure_type_confidence: float
    failure_type_alternatives: list = field(default_factory=list)
    eta: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ModelService:
    """One place that owns the fitted models and how they are called.

    `investigate_threshold` is deliberately lower than `alert_threshold`: it is
    the trigger for *looking*, not for *alarming*. The agent layer decides what
    to do with a case in between, which is where the hypothesis advocates and
    the adjudicator earn their place.
    """

    def __init__(self, bundle: ModelBundle,
                 investigate_ratio: float = 0.45):
        self.bundle = bundle
        self.alert_threshold = float(bundle.threshold)
        self.investigate_threshold = float(bundle.threshold * investigate_ratio)

    @classmethod
    def load(cls, path: Path = BUNDLE_DIR, **kw) -> "ModelService":
        return cls(ModelBundle.load(path), **kw)

    # -- the questions an agent may ask -----------------------------------
    def predict(self, df: pd.DataFrame, machine_id: str,
                timestamp) -> Prediction:
        b = self.bundle
        x, raw = window_for(df, machine_id, timestamp, b.channels, b.scaler,
                            b.lookback)
        if b.is_sequence():
            prob = float(b.score_windows(
                x[None, ...], np.array([static_index(machine_id)]))[0])
        else:
            row = df[(df.machine_id == machine_id)
                     & (df.timestamp == pd.Timestamp(timestamp))]
            prob = float(b.score_rows(
                row[b.tabular_columns].to_numpy(dtype=np.float32))[0])

        tab = df[(df.machine_id == machine_id)
                 & (df.timestamp == pd.Timestamp(timestamp))]
        ftype, fconf, alts = None, 0.0, []
        eta = {"eta_hours": None, "window_low_h": None, "window_high_h": None}
        if len(tab):
            if b.type_classifier is not None:
                got = b.type_classifier.label(
                    tab[b.type_classifier.columns].to_numpy(dtype=np.float32))[0]
                ftype, fconf, alts = (got["failure_type"],
                                      round(got["confidence"], 4),
                                      got["alternatives"])
            if b.ttf_regressor is not None:
                eta = b.ttf_regressor.window(
                    tab[b.ttf_regressor.columns].to_numpy(dtype=np.float32))[0]

        band = ("high" if prob >= self.alert_threshold
                else "watch" if prob >= self.investigate_threshold
                else "normal")
        return Prediction(
            machine_id=machine_id, timestamp=str(timestamp),
            failure_probability=round(prob, 4),
            alert=bool(prob >= self.alert_threshold),
            investigate=bool(prob >= self.investigate_threshold),
            risk_band=band,
            alert_threshold=round(self.alert_threshold, 4),
            investigate_threshold=round(self.investigate_threshold, 4),
            confidence=round(float(b.confidence(prob)), 4),
            failure_type=ftype, failure_type_confidence=fconf,
            failure_type_alternatives=alts, eta=eta,
            model={"kind": b.kind, "feature_set": b.feature_set,
                   "lookback_steps": b.lookback})

    def window(self, df: pd.DataFrame, machine_id: str, timestamp):
        b = self.bundle
        return window_for(df, machine_id, timestamp, b.channels, b.scaler,
                          b.lookback)

    def raw_window(self, df: pd.DataFrame, machine_id: str, timestamp):
        b = self.bundle
        return window_for(df, machine_id, timestamp, b.channels, None,
                          b.lookback)[1]

    def attribution(self, df: pd.DataFrame, machine_id: str,
                    timestamp) -> list[dict]:
        """Which channels the model itself weighted."""
        b = self.bundle
        if b.is_sequence() and b.torch_model is not None:
            x, _ = self.window(df, machine_id, timestamp)
            ex = b.explain_windows(x[None, ...],
                                   np.array([static_index(machine_id)]))
            imp = ex["channel_importance"][0].numpy()
            order = np.argsort(-np.abs(imp))[:5]
            return [{"channel": b.channels[int(i)],
                     "weight": round(float(imp[int(i)]), 5)} for i in order]
        if b.tree_model is not None:
            return [{"channel": k, "weight": round(v, 5)} for k, v in
                    list(b.tree_model.feature_importance().items())[:5]]
        return []

    def score_frame(self, enriched: pd.DataFrame, machine_id: str,
                    timestamp) -> float:
        """Score an arbitrary (e.g. counterfactual) telemetry frame."""
        b = self.bundle
        if b.is_sequence():
            x, _ = window_for(enriched, machine_id, timestamp, b.channels,
                              b.scaler, b.lookback)
            return float(b.score_windows(
                x[None, ...], np.array([static_index(machine_id)]))[0])
        row = enriched[enriched["timestamp"] == pd.Timestamp(timestamp)]
        return float(b.score_rows(
            row[b.tabular_columns].to_numpy(dtype=np.float32))[0])

    def describe(self) -> dict:
        b = self.bundle
        return {"kind": b.kind, "feature_set": b.feature_set,
                "alert_threshold": round(self.alert_threshold, 4),
                "investigate_threshold": round(self.investigate_threshold, 4),
                "lookback_steps": b.lookback,
                "selection_rationale": b.selection_rationale}
