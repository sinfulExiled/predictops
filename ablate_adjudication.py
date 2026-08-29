#!/usr/bin/env python
"""Does the hypothesis contest actually earn its place?

The architectural claim is: because a flagged case must survive a benign
counter-argument, the model can afford a *lower* trigger and buy recall
without paying for it in false callouts.

That is a testable claim, so this tests it. For a sweep of alert thresholds it
compares, on the same scenarios:

    model_only   -- alert iff probability >= threshold
    adjudicated  -- the model flags, then the two advocates argue and the
                    adjudicator decides

If adjudication never changes a verdict, or changes them for the worse, this
script says so and the architecture does not get to claim the win.

    python ablate_adjudication.py
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from predictops.agents.base import AgentContext
from predictops.agents.context import ContextAgent
from predictops.agents.hypothesis import (
    Adjudicator,
    ConfoundAdvocate,
    DegradationAdvocate,
)
from predictops.agents.investigator import InvestigationAgent, SignatureLibrary
from predictops.agents.predictor import PredictionAgent
from predictops.config import REPORT_DIR
from predictops.evaluation.scenarios import load_suite
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import MockProvider
from predictops.ml.bundle import BUNDLE_DIR, ModelBundle
from predictops.ml.dataset import prepare
from predictops.ml.service import ModelService


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.35,0.25,0.15",
                    help="alert thresholds as fractions of the tuned one")
    args = ap.parse_args()

    data = prepare()
    suite = load_suite()
    bundle = ModelBundle.load(BUNDLE_DIR)
    tuned = float(bundle.threshold)
    ratios = [float(r) for r in args.ratios.split(",")]

    store = ExperimentStore()
    library = SignatureLibrary.build(data.df, data.failures, bundle.channels,
                                     bundle.lookback)

    # --- score every scenario once; the contest does not depend on the
    #     threshold, only on whether the case was flagged for investigation ---
    print(f"scoring {len(suite)} scenarios and running the contest on each ...")
    t0 = time.time()
    service = ModelService(bundle, investigate_ratio=min(ratios) * 0.95)
    ctx = AgentContext(run_id=f"ablation-{datetime.now():%Y%m%d-%H%M%S}",
                       data=data, store=store, provider=MockProvider(),
                       verbose=False)

    rows = []
    for i, sc in enumerate(suite, 1):
        pred = PredictionAgent().execute(
            ctx, service=service, machine_id=sc.machine_id,
            timestamp=sc.timestamp).output
        con = ContextAgent().execute(
            ctx, machine_id=sc.machine_id, timestamp=sc.timestamp).output
        inv = InvestigationAgent().execute(
            ctx, service=service, machine_id=sc.machine_id,
            timestamp=sc.timestamp, library=library).output
        builder = ctx.state["evidence_builder"]
        deg = DegradationAdvocate().execute(
            ctx, builder=builder, prediction=pred, context=con,
            neighbours=inv.get("similar_past_failures", [])).output
        conf = ConfoundAdvocate().execute(
            ctx, builder=builder, context=con, prediction=pred).output
        rows.append({
            "id": sc.id, "category": sc.category, "difficulty": sc.difficulty,
            "expect": bool(sc.expect_alert),
            "p": float(pred["failure_probability"]),
            "deg": float(deg["score"]), "conf": float(conf["score"]),
            "benign": (conf.get("alternative_explanations") or [{}])[0]
                      .get("explanation"),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(suite)}", flush=True)
    print(f"  done in {time.time() - t0:.0f}s\n")

    # --- sweep -----------------------------------------------------------
    results = []
    for ratio in ratios:
        thr = tuned * ratio
        m = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        a = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        flips = []
        for r in rows:
            model_alert = r["p"] >= thr
            adj = Adjudicator().execute(
                ctx,
                degradation={"score": r["deg"]},
                confound={"score": r["conf"],
                          "alternative_explanations":
                              [{"explanation": r["benign"]}] if r["benign"] else []},
                prediction={"alert": model_alert, "investigate": model_alert,
                            "failure_probability": r["p"]}).output
            adj_alert = bool(adj["alert"])

            for box, flag in ((m, model_alert), (a, adj_alert)):
                if r["expect"]:
                    box["tp" if flag else "fn"] += 1
                else:
                    box["fp" if flag else "tn"] += 1
            if adj_alert != model_alert:
                flips.append({"id": r["id"], "category": r["category"],
                              "expect": r["expect"],
                              "model": model_alert, "adjudicated": adj_alert,
                              "decision": adj["decision"],
                              "deg": r["deg"], "conf": r["conf"],
                              "helped": (adj_alert == r["expect"])})

        mp, mr, mf = prf(m["tp"], m["fp"], m["fn"])
        ap_, ar, af = prf(a["tp"], a["fp"], a["fn"])
        results.append({
            "ratio": ratio, "threshold": round(thr, 4),
            "model_only": {"precision": round(mp, 4), "recall": round(mr, 4),
                           "f1": round(mf, 4), **m},
            "adjudicated": {"precision": round(ap_, 4), "recall": round(ar, 4),
                            "f1": round(af, 4), **a},
            "verdicts_changed": len(flips),
            "changes_that_helped": sum(1 for f in flips if f["helped"]),
            "changes_that_hurt": sum(1 for f in flips if not f["helped"]),
            "flips": flips,
        })

    # --- report ------------------------------------------------------------
    print("=" * 92)
    print(f"{'threshold':>10}  {'MODEL ONLY':^28}  {'ADJUDICATED':^28}  "
          f"{'changed':>8}")
    print(f"{'':>10}  {'prec':>8}{'recall':>10}{'F1':>10}  "
          f"{'prec':>8}{'recall':>10}{'F1':>10}  {'+/-':>8}")
    print("-" * 92)
    for r in results:
        m, a = r["model_only"], r["adjudicated"]
        print(f"{r['threshold']:>10.4f}  {m['precision']:>8.3f}"
              f"{m['recall']:>10.3f}{m['f1']:>10.3f}  "
              f"{a['precision']:>8.3f}{a['recall']:>10.3f}{a['f1']:>10.3f}  "
              f"{r['changes_that_helped']:>4}/{r['changes_that_hurt']:<3}")
    print("=" * 92)

    total_changed = sum(r["verdicts_changed"] for r in results)
    helped = sum(r["changes_that_helped"] for r in results)
    hurt = sum(r["changes_that_hurt"] for r in results)
    print(f"\nAcross the sweep the adjudicator changed {total_changed} verdict(s): "
          f"{helped} correct, {hurt} incorrect.")

    best_m = max(results, key=lambda r: r["model_only"]["f1"])
    best_a = max(results, key=lambda r: r["adjudicated"]["f1"])
    print(f"Best model-only F1  : {best_m['model_only']['f1']:.4f} "
          f"at threshold {best_m['threshold']:.4f}")
    print(f"Best adjudicated F1 : {best_a['adjudicated']['f1']:.4f} "
          f"at threshold {best_a['threshold']:.4f}")
    delta = best_a["adjudicated"]["f1"] - best_m["model_only"]["f1"]
    print(f"\nVERDICT: adjudication is worth {delta:+.4f} F1 at the best "
          f"operating point of each.")
    if total_changed == 0:
        print("It never changed a verdict on this suite -- on this evidence "
              "the contest is not earning its place.")

    for r in results:
        for f in r["flips"]:
            print(f"  [thr {r['threshold']:.3f}] {f['id']} {f['category']:<28} "
                  f"model={'alert' if f['model'] else 'quiet'} -> "
                  f"{'alert' if f['adjudicated'] else 'quiet'} "
                  f"({f['decision']}, deg {f['deg']:.2f} vs conf {f['conf']:.2f}) "
                  f"{'CORRECT' if f['helped'] else 'WRONG'}")

    out = REPORT_DIR / "ablation_adjudication.json"
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "tuned_threshold": tuned,
        "cases": rows,
        "sweep": results,
        "summary": {"verdicts_changed": total_changed, "helped": helped,
                    "hurt": hurt,
                    "best_model_only_f1": best_m["model_only"]["f1"],
                    "best_adjudicated_f1": best_a["adjudicated"]["f1"],
                    "delta_f1": round(delta, 4)},
    }, indent=2, default=str))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
