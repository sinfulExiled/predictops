"""Run the fixed scenario suite through both systems.

Fairness rules, applied deliberately:

* Both systems see the **same cases** and the **same telemetry**.
* Both get a decision threshold tuned on the **same validation split**, and
  frozen before the suite is touched.
* The baseline is given a cause-attribution rule too -- the channel that
  tripped, mapped to the failure mode that channel most often signals.  Without
  that, "cause accuracy" would be a category the baseline is structurally
  unable to score in, and the comparison would be theatre.

The baseline genuinely cannot produce a remediation plan, a simulation or a
verification verdict.  Those are reported as *capabilities the baseline does
not have* rather than as zero scores, because scoring a system at zero for a
question it was never asked is not a measurement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..agents.orchestrator import PredictOpsEngine
from ..ml.baseline import ThresholdBaseline
from .scenarios import Scenario

# Which failure mode a tripped channel most often indicates.  This is the
# best a threshold rule can do, and it is what the baseline is credited with.
CHANNEL_TO_MODE = {
    "vibration": "bearing_degradation",
    "temperature": "motor_overheating",
    "current": "electrical_fault",
    "pressure": "pressure_loss",
}


@dataclass
class CaseResult:
    scenario_id: str
    category: str
    difficulty: str
    machine_id: str
    timestamp: str
    expected_alert: bool
    expected_type: str | None
    alert: bool
    probability: float
    predicted_type: str | None
    correct_alert: bool
    correct_type: bool | None
    early_warning_h: float | None = None
    verification: str = ""
    safe_to_act: bool | None = None
    plan: list = field(default_factory=list)
    simulated_reduction: float | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------
def run_baseline_suite(engine: PredictOpsEngine,
                       suite: list[Scenario]) -> list[CaseResult]:
    data = engine.data
    val = data.split("val")
    val = val[(val.is_downtime == 0) & (val.sensor_dropout == 0)]
    model = ThresholdBaseline().fit(val)

    df = data.df
    results = []
    for sc in suite:
        t0 = time.time()
        g = df[(df.machine_id == sc.machine_id)
               & (df.timestamp <= pd.Timestamp(sc.timestamp))]
        score = float(model.score(g)[-1])
        alert = bool(score >= model.k)

        # cause attribution: whichever channel is furthest above its own history
        ptype = None
        if alert:
            tail = g.tail(engine.bundle.lookback if engine.bundle else 36)
            best_z, best_ch = -np.inf, None
            for ch in CHANNEL_TO_MODE:
                if ch not in g.columns:
                    continue
                hist = g[ch].to_numpy(dtype=float)[:-1]
                if len(hist) < 12 or not np.isfinite(hist).any():
                    continue
                mu, sd = np.nanmean(hist), np.nanstd(hist)
                z = (float(tail[ch].iloc[-1]) - mu) / (sd + 1e-6)
                if z > best_z:
                    best_z, best_ch = z, ch
            ptype = CHANNEL_TO_MODE.get(best_ch)

        results.append(CaseResult(
            scenario_id=sc.id, category=sc.category, difficulty=sc.difficulty,
            machine_id=sc.machine_id, timestamp=sc.timestamp,
            expected_alert=sc.expect_alert, expected_type=sc.expect_failure_type,
            alert=alert, probability=round(score, 4), predicted_type=ptype,
            correct_alert=(alert == sc.expect_alert),
            correct_type=(None if not sc.expect_alert
                          else bool(ptype == sc.expect_failure_type)),
            early_warning_h=(sc.hours_to_failure
                             if alert and sc.expect_alert else None),
            verification="not available (threshold rule)",
            duration_s=time.time() - t0))
    return results


def run_agent_suite(engine: PredictOpsEngine, suite: list[Scenario],
                    full_workflow: bool = True,
                    verbose: bool = True) -> list[CaseResult]:
    results = []
    for i, sc in enumerate(suite, 1):
        t0 = time.time()
        if verbose:
            print(f"  [{i:>2}/{len(suite)}] {sc.id} {sc.category:<32} "
                  f"{sc.machine_id}", flush=True)
        if full_workflow:
            rep = engine.run_incident(sc.machine_id, sc.timestamp, save=False)
            pred, ver = rep.prediction, rep.verification
            sim = rep.simulation
            arms = [a for a in sim.get("arms", []) if a.get("simulated")]
            best = (min(arms, key=lambda a: a["failure_probability_simulated"])
                    if arms else None)
            reduction = (round(sim["no_action"]["failure_probability_simulated"]
                               - best["failure_probability_simulated"], 4)
                         if best else None)
            results.append(CaseResult(
                scenario_id=sc.id, category=sc.category,
                difficulty=sc.difficulty, machine_id=sc.machine_id,
                timestamp=sc.timestamp, expected_alert=sc.expect_alert,
                expected_type=sc.expect_failure_type,
                alert=bool(pred["alert"]),
                probability=pred["failure_probability"],
                predicted_type=(rep.investigation.get("likely_failure_type")
                                if pred["alert"] else None),
                correct_alert=(bool(pred["alert"]) == sc.expect_alert),
                # A system that stayed silent named no cause. Reading the
                # investigation's internal ranking here instead would credit
                # the system for a diagnosis it never surfaced to anyone.
                correct_type=(None if not sc.expect_alert else bool(
                    pred["alert"]
                    and rep.investigation.get("likely_failure_type")
                    == sc.expect_failure_type)),
                early_warning_h=(sc.hours_to_failure
                                 if pred["alert"] and sc.expect_alert else None),
                verification=ver["verdict"],
                safe_to_act=ver["safe_to_act"],
                plan=[p["intervention_id"] for p in rep.remediation["plan"]],
                simulated_reduction=reduction,
                duration_s=time.time() - t0))
        else:
            from ..agents.predictor import PredictionAgent
            out = PredictionAgent().execute(
                engine.ctx, service=engine.service, machine_id=sc.machine_id,
                timestamp=sc.timestamp).output
            results.append(CaseResult(
                scenario_id=sc.id, category=sc.category,
                difficulty=sc.difficulty, machine_id=sc.machine_id,
                timestamp=sc.timestamp, expected_alert=sc.expect_alert,
                expected_type=sc.expect_failure_type,
                alert=bool(out["alert"]),
                probability=out["failure_probability"],
                predicted_type=out.get("failure_type") if out["alert"] else None,
                correct_alert=(bool(out["alert"]) == sc.expect_alert),
                correct_type=(None if not sc.expect_alert
                              else bool(out.get("failure_type")
                                        == sc.expect_failure_type)),
                duration_s=time.time() - t0))
    return results


# --------------------------------------------------------------------------
def score_suite(results: list[CaseResult]) -> dict:
    df = pd.DataFrame([r.to_dict() for r in results])
    pos = df[df.expected_alert]
    neg = df[~df.expected_alert]

    tp = int((pos.alert).sum())
    fn = int((~pos.alert).sum())
    fp = int((neg.alert).sum())
    tn = int((~neg.alert).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    typed = pos[pos.correct_type.notna()]
    # Two honest readings of "did it name the right cause":
    #   coverage  -- over every real failure (silence counts as a miss)
    #   precision -- over the failures it actually raised
    alerted_pos = pos[pos.alert]
    hard = df[df.difficulty == "hard"]

    by_cat = {}
    for cat, g in df.groupby("category"):
        by_cat[cat] = {
            "n": int(len(g)),
            "correct": int(g.correct_alert.sum()),
            "accuracy": round(float(g.correct_alert.mean()), 3),
        }

    out = {
        "n_cases": int(len(df)),
        "alert_accuracy": round(float(df.correct_alert.mean()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "false_alarm_rate_on_nuisance_cases": round(
            float(neg.alert.mean()) if len(neg) else 0.0, 4),
        "cause_accuracy": (round(float(typed.correct_type.mean()), 4)
                           if len(typed) else None),
        "cause_accuracy_when_alerted": (
            round(float(alerted_pos.correct_type.mean()), 4)
            if len(alerted_pos) else None),
        "n_alerted_positives": int(len(alerted_pos)),
        "hard_case_accuracy": (round(float(hard.correct_alert.mean()), 4)
                               if len(hard) else None),
        "mean_early_warning_h": (
            round(float(df.early_warning_h.dropna().mean()), 2)
            if df.early_warning_h.notna().any() else 0.0),
        "median_seconds_per_case": round(float(df.duration_s.median()), 3),
        "by_category": by_cat,
    }
    if "verification" in df and df["verification"].notna().any():
        out["verification_verdicts"] = df["verification"].value_counts().to_dict()
    if "safe_to_act" in df and df["safe_to_act"].notna().any():
        out["cases_cleared_to_act"] = int(df["safe_to_act"].fillna(False).sum())
    if "simulated_reduction" in df and df["simulated_reduction"].notna().any():
        out["mean_simulated_risk_reduction"] = round(
            float(df["simulated_reduction"].dropna().mean()), 4)
    return out
