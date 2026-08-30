"""The deployable bundle: everything needed to serve one prediction.

Assembled once, after the research agent has chosen a winner, and then loaded
by the prediction/investigation/simulation agents and by the API.  Keeping it
in one object is what stops the serving path from quietly using a different
scaler, threshold or feature order than the evaluation did.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import LOOKBACK_STEPS, MODEL_DIR
from ..data.preprocessing import Scaler
from .diagnosis import (
    FailureTypeClassifier,
    ReliabilityCurve,
    TimeToFailureRegressor,
)

BUNDLE_DIR = MODEL_DIR / "bundle"


@dataclass
class ModelBundle:
    kind: str                       # "tft" | "lstm" | "xgboost" | "ensemble"
    feature_set: str
    channels: list[str]             # sequence channel order
    tabular_columns: list[str]      # tabular feature order
    threshold: float
    scaler: Scaler
    lookback: int = LOOKBACK_STEPS
    torch_model: object = None
    tree_model: object = None
    ensemble: dict | None = None
    reliability: ReliabilityCurve = field(default_factory=ReliabilityCurve)
    type_classifier: FailureTypeClassifier | None = None
    ttf_regressor: TimeToFailureRegressor | None = None
    metrics: dict = field(default_factory=dict)
    selection_rationale: str = ""

    # -- scoring -----------------------------------------------------------
    def is_sequence(self) -> bool:
        return self.kind in ("lstm", "tft", "ensemble")

    def score_windows(self, x: np.ndarray, static: np.ndarray) -> np.ndarray:
        """x: (B, T, C) already scaled.  Returns failure probability."""
        import torch

        from .training import predict_tensors
        if not self.is_sequence():
            raise ValueError(f"{self.kind} is a tabular model; use score_rows")
        return predict_tensors(self.torch_model,
                               torch.from_numpy(x.astype(np.float32)),
                               torch.from_numpy(static.astype(np.int64)))

    def score_rows(self, x: np.ndarray) -> np.ndarray:
        return self.tree_model.predict_proba(x.astype(np.float32))

    def explain_windows(self, x: np.ndarray, static: np.ndarray) -> dict:
        import torch
        return self.torch_model.explain(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(static.astype(np.int64)))

    def confidence(self, prob: float) -> float:
        return self.reliability.confidence(prob)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path = BUNDLE_DIR) -> Path:
        import pickle

        import torch
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind, "feature_set": self.feature_set,
            "channels": self.channels, "tabular_columns": self.tabular_columns,
            "threshold": self.threshold, "lookback": self.lookback,
            "scaler": self.scaler.to_dict(),
            "reliability": self.reliability.to_dict(),
            "ensemble": self.ensemble,
            "metrics": self.metrics,
            "selection_rationale": self.selection_rationale,
        }
        (path / "bundle.json").write_text(json.dumps(meta, indent=2))
        if self.torch_model is not None:
            torch.save(self.torch_model.state_dict(), path / "sequence.pt")
        for name, obj in (("tree", self.tree_model),
                          ("type_classifier", self.type_classifier),
                          ("ttf_regressor", self.ttf_regressor)):
            if obj is not None:
                with open(path / f"{name}.pkl", "wb") as fh:
                    pickle.dump(obj, fh)
        return path

    @classmethod
    def load(cls, path: Path = BUNDLE_DIR) -> "ModelBundle":
        import pickle

        import torch

        from .training import MACHINE_TYPES, build_model
        meta = json.loads((path / "bundle.json").read_text())
        bundle = cls(
            kind=meta["kind"], feature_set=meta["feature_set"],
            channels=meta["channels"], tabular_columns=meta["tabular_columns"],
            threshold=meta["threshold"], lookback=meta["lookback"],
            scaler=Scaler.from_dict(meta["scaler"]),
            reliability=ReliabilityCurve.from_dict(meta["reliability"]),
            ensemble=meta.get("ensemble"), metrics=meta.get("metrics", {}),
            selection_rationale=meta.get("selection_rationale", ""))
        seq = path / "sequence.pt"
        if seq.exists():
            if bundle.kind == "ensemble":
                from .torch_models import EnsembleModel
                meta_ens = bundle.ensemble or {}
                kinds = meta_ens.get("member_kinds") or ["lstm", "tft"]
                w = float(meta_ens.get("weight_first", 0.5))
                members = [build_model(k, len(bundle.channels)) for k in kinds]
                model = EnsembleModel(members, [w, 1.0 - w])
            else:
                model = build_model(bundle.kind, len(bundle.channels))
            model.load_state_dict(torch.load(seq, map_location="cpu"))
            model.eval()
            bundle.torch_model = model
        for name, attr in (("tree", "tree_model"),
                           ("type_classifier", "type_classifier"),
                           ("ttf_regressor", "ttf_regressor")):
            p = path / f"{name}.pkl"
            if p.exists():
                with open(p, "rb") as fh:
                    setattr(bundle, attr, pickle.load(fh))
        return bundle


def window_for(df: pd.DataFrame, machine_id: str, timestamp, channels: list[str],
               scaler: Scaler | None, lookback: int = LOOKBACK_STEPS
               ) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract one scaled lookback window ending at `timestamp`.

    Returns the model input and the raw slice, so evidence can be quoted in
    engineering units rather than in standard deviations.
    """
    g = df[df["machine_id"] == machine_id].sort_values("timestamp")
    ts = pd.Timestamp(timestamp)
    hit = np.flatnonzero((g["timestamp"] == ts).to_numpy())
    if len(hit) == 0:
        raise ValueError(f"{machine_id} has no sample at {ts}")
    i = int(hit[0])
    if i < lookback - 1:
        raise ValueError(f"{machine_id} at {ts} has only {i + 1} prior samples; "
                         f"{lookback} required")
    raw = g.iloc[i - lookback + 1:i + 1]
    x = raw[channels].to_numpy(dtype=np.float64)
    if scaler is not None:
        x = scaler.transform(x)
    return x.astype(np.float32), raw
