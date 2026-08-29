"""Run one model candidate end to end and record what actually happened.

Protocol, identical for every candidate:

  fit on train  ->  tune the decision threshold on val  ->  freeze it
                ->  score the canonical test rows once

The threshold is never re-tuned on test.  The canonical row set comes from
`evaluation.harness`, so the threshold rule, the trees and the sequence models
are all graded on the same exam paper.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import LOOKBACK_STEPS, MODEL_DIR, TrainingConfig
from ..data.preprocessing import Scaler
from ..evaluation.harness import EvaluationSet
from ..ml.baseline import ThresholdBaseline
from ..ml.dataset import PreparedData
from ..ml.evaluation import full_report, pick_threshold, row_metrics
from ..ml.features import feature_columns, sequence_channels
from ..ml.training import (
    TrainedModel,
    predict_windows,
    train_sequence_model,
)
from ..ml.trees import train_tree
from .registry import Experiment, ExperimentStore


@dataclass
class Fitted:
    """Everything the pipeline needs to reuse a trained candidate."""

    name: str
    kind: str
    feature_set: str
    predict_test: np.ndarray
    threshold: float
    metrics: dict
    scaler: Scaler | None = None
    channels: list[str] | None = None
    torch_model: object = None
    tree_model: object = None
    extra: dict | None = None


class ExperimentRunner:
    def __init__(self, data: PreparedData, store: ExperimentStore, run_id: str,
                 lookback: int = LOOKBACK_STEPS, device: str = "cpu"):
        self.data = data
        self.store = store
        self.run_id = run_id
        self.lookback = lookback
        self.device = device
        self._cache: dict[str, tuple] = {}
        self.fitted: dict[str, Fitted] = {}

    # -- shared plumbing ---------------------------------------------------
    def sequence_bits(self, feature_set: str):
        """Scaler + train/val/test window indices + canonical eval sets."""
        if feature_set in self._cache:
            return self._cache[feature_set]
        cols = [c for c in sequence_channels(feature_set)
                if c in self.data.df.columns]
        scaler = Scaler.fit(self.data.split("train"), cols)
        tr = self.data.windows("train", feature_set, scaler, self.lookback)
        va = self.data.windows("val", feature_set, scaler, self.lookback)
        te = self.data.windows("test", feature_set, scaler, self.lookback)
        ev_va = EvaluationSet.from_windows(va, "val")
        ev_te = EvaluationSet.from_windows(te, "test")
        bits = (scaler, cols, tr, va, te, ev_va, ev_te)
        self._cache[feature_set] = bits
        return bits

    def _report(self, ev: EvaluationSet, score: np.ndarray,
                threshold: float) -> dict:
        return full_report(ev.machine_id, ev.timestamp, ev.y, score,
                           threshold, self.data.failures)

    def _record(self, stage: str, name: str, model: str, feature_set: str,
                hypothesis: str, params: dict, metrics: dict,
                duration: float) -> int:
        return self.store.record(Experiment(
            run_id=self.run_id, stage=stage, name=name, model=model,
            feature_set=feature_set, hypothesis=hypothesis, params=params,
            metrics=metrics, duration_s=round(duration, 2)))

    # -- candidates --------------------------------------------------------
    def run_baseline(self, stage: str = "Baseline") -> tuple[Fitted, int]:
        t0 = time.time()
        _, _, _, _, _, ev_va, ev_te = self.sequence_bits("engineered")

        val = self.data.split("val")
        val = val[(val.is_downtime == 0) & (val.sensor_dropout == 0)]
        model = ThresholdBaseline().fit(val)

        def score_split(split_name):
            d = self.data.split(split_name)
            d = d[(d.is_downtime == 0) & (d.sensor_dropout == 0)]
            d = d.sort_values(["machine_id", "timestamp"])
            return (d["machine_id"].to_numpy(), d["timestamp"].to_numpy(),
                    model.score(d))

        mv, tv, sv = score_split("val")
        val_aligned = ev_va.align(mv, tv, sv)

        m, t, s = score_split("test")
        aligned = ev_te.align(m, t, s)
        metrics = self._report(ev_te, aligned, model.k)
        metrics["val"] = row_metrics(ev_va.y, val_aligned, model.k).to_dict()
        dt = time.time() - t0

        exp_id = self._record(
            stage, "threshold rule on vibration / temperature / current",
            "threshold_baseline", "raw",
            "A per-machine z-score alarm is what the plant already runs; "
            "establish where that leaves us.",
            {"k": round(model.k, 3), "channels": list(model.channels)},
            metrics, dt)
        f = Fitted("baseline", "threshold_baseline", "raw", aligned, model.k,
                   metrics)
        self.fitted["baseline"] = f
        return f, exp_id

    def run_tree(self, kind: str, feature_set: str, stage: str,
                 hypothesis: str) -> tuple[Fitted, int]:
        t0 = time.time()
        _, _, _, _, _, ev_va, ev_te = self.sequence_bits("engineered")
        cols = feature_columns(self.data.df, feature_set)

        x_tr, y_tr, *_ = self.data.tabular("train", feature_set)
        x_va, y_va, m_va, t_va, _ = self.data.tabular("val", feature_set)
        x_te, y_te, m_te, t_te, _ = self.data.tabular("test", feature_set)

        model = train_tree(kind, x_tr, y_tr, x_va, y_va, cols)

        p_va = ev_va.align(m_va, t_va, model.predict_proba(x_va))
        threshold = pick_threshold(ev_va.y, p_va)
        p_te = ev_te.align(m_te, t_te, model.predict_proba(x_te))
        metrics = self._report(ev_te, p_te, threshold)
        metrics["val"] = row_metrics(ev_va.y, p_va, threshold).to_dict()
        dt = time.time() - t0

        name = f"{kind} on {feature_set} features"
        exp_id = self._record(stage, name, kind, feature_set, hypothesis,
                              {"n_features": len(cols), "threshold": threshold},
                              metrics, dt)
        f = Fitted(f"{kind}_{feature_set}", kind, feature_set, p_te, threshold,
                   metrics, tree_model=model,
                   extra={"feature_importance": dict(
                       list(model.feature_importance().items())[:25])})
        self.fitted[f.name] = f
        return f, exp_id

    def run_sequence(self, kind: str, feature_set: str, stage: str,
                     hypothesis: str, cfg: TrainingConfig | None = None,
                     neg_per_pos: float = 12.0,
                     progress: bool = True) -> tuple[Fitted, int]:
        t0 = time.time()
        cfg = cfg or TrainingConfig()
        scaler, cols, tr, va, te, ev_va, ev_te = self.sequence_bits(feature_set)
        _, _, _, _, _, _, ev_te_canon = self.sequence_bits("engineered")

        trained: TrainedModel = train_sequence_model(
            kind, tr, va, cfg, neg_per_pos=neg_per_pos, progress=progress)

        p_va = predict_windows(trained.model, va, device=self.device)
        threshold = pick_threshold(va.labels, p_va)
        p_te = predict_windows(trained.model, te, device=self.device)

        ev_te_local = EvaluationSet.from_windows(te, "test")
        aligned = ev_te_canon.align(ev_te_local.machine_id,
                                    ev_te_local.timestamp, p_te)
        metrics = self._report(ev_te_canon, aligned, threshold)
        metrics["val"] = row_metrics(va.labels, p_va, threshold).to_dict()
        dt = time.time() - t0

        path = MODEL_DIR / f"{kind}_{feature_set}.pt"
        trained.threshold = threshold
        trained.scaler = scaler
        trained.save(path)

        name = f"{kind.upper()} on {feature_set} channels"
        exp_id = self._record(
            stage, name, kind, feature_set, hypothesis,
            {"lookback": self.lookback, "n_channels": len(cols),
             "neg_per_pos": neg_per_pos, "epochs_run": trained.epochs_run,
             "val_pr_auc": round(trained.val_pr_auc, 5),
             "threshold": round(threshold, 5),
             "checkpoint": str(path.relative_to(MODEL_DIR.parent.parent))},
            metrics, dt)

        f = Fitted(f"{kind}_{feature_set}", kind, feature_set, aligned,
                   threshold, metrics, scaler=scaler, channels=cols,
                   torch_model=trained.model,
                   extra={"history": trained.history,
                          "val_pr_auc": trained.val_pr_auc,
                          "checkpoint": str(path)})
        self.fitted[f.name] = f
        return f, exp_id

    def run_ensemble(self, members: list[str], feature_set: str, stage: str,
                     hypothesis: str) -> tuple[Fitted, int]:
        """Blend two sequence models, with the weight fitted on validation."""
        t0 = time.time()
        scaler, cols, tr, va, te, ev_va, ev_te = self.sequence_bits(feature_set)
        _, _, _, _, _, _, ev_te_canon = self.sequence_bits("engineered")

        models = [self.fitted[m].torch_model for m in members]
        val_probs = [predict_windows(m, va, device=self.device) for m in models]
        test_probs = [predict_windows(m, te, device=self.device) for m in models]

        from sklearn.metrics import average_precision_score
        best_w, best_s = 0.5, -1.0
        for w in np.linspace(0.0, 1.0, 21):
            p = w * val_probs[0] + (1 - w) * val_probs[1]
            s = float(average_precision_score(va.labels, p))
            if s > best_s:
                best_s, best_w = s, float(w)

        p_va = best_w * val_probs[0] + (1 - best_w) * val_probs[1]
        threshold = pick_threshold(va.labels, p_va)
        p_te = best_w * test_probs[0] + (1 - best_w) * test_probs[1]

        ev_local = EvaluationSet.from_windows(te, "test")
        aligned = ev_te_canon.align(ev_local.machine_id, ev_local.timestamp, p_te)
        metrics = self._report(ev_te_canon, aligned, threshold)
        metrics["val"] = row_metrics(va.labels, p_va, threshold).to_dict()
        dt = time.time() - t0

        name = f"ensemble {members[0]} + {members[1]}"
        exp_id = self._record(
            stage, name, "ensemble", feature_set, hypothesis,
            {"members": members, "weight_first": round(best_w, 3),
             "val_pr_auc": round(best_s, 5), "threshold": round(threshold, 5)},
            metrics, dt)

        f = Fitted("ensemble", "ensemble", feature_set, aligned, threshold,
                   metrics, scaler=scaler, channels=cols,
                   extra={"members": members, "weight_first": best_w,
                          "val_pr_auc": best_s})
        self.fitted["ensemble"] = f
        return f, exp_id
