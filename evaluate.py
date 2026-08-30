#!/usr/bin/env python
"""Phase 12 -- baseline vs agent on the fixed scenario suite.

    python evaluate.py                 # full agent workflow (slower)
    python evaluate.py --model-only    # prediction only, no agent workflow
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from predictops.agents.orchestrator import PredictOpsEngine
from predictops.config import REPORT_DIR
from predictops.evaluation.scenario_runner import (
    run_agent_suite,
    run_baseline_suite,
    score_suite,
)
from predictops.evaluation.scenarios import (
    build_suite,
    load_suite,
    save_suite,
    summarise,
)
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import get_provider
from predictops.ml.dataset import prepare


def _pct(v):
    return "--" if v is None else f"{v * 100:.1f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-only", action="store_true",
                   help="skip investigation/remediation/simulation/verification")
    p.add_argument("--provider", default=None)
    p.add_argument("--rebuild-suite", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    data = prepare()
    def _fresh_suite():
        s = build_suite(data.df, data.failures)
        save_suite(s)
        return s

    if args.rebuild_suite:
        suite = _fresh_suite()
    else:
        try:
            suite = load_suite()
        except FileNotFoundError:
            suite = _fresh_suite()
        else:
            # A suite built against a different dataset names machines that no
            # longer exist, which used to surface as an opaque IndexError deep
            # inside the baseline. Detect it and rebuild.
            known = set(data.df["machine_id"].unique())
            missing = [c.machine_id for c in suite if c.machine_id not in known]
            if missing:
                print(f"scenario suite references {len(missing)} machine(s) "
                      f"not in this dataset (e.g. {missing[0]}) — rebuilding\n")
                suite = _fresh_suite()

    print(f"scenario suite: {json.dumps(summarise(suite))}\n")

    engine = PredictOpsEngine(
        data=data, store=ExperimentStore(),
        provider=get_provider(args.provider),
        run_id=f"eval-{datetime.now():%Y%m%d-%H%M%S}", verbose=False)
    engine.load_bundle()
    print(f"model under test: {engine.bundle.kind} on "
          f"{engine.bundle.feature_set} features "
          f"(threshold {engine.bundle.threshold:.4f})\n")

    print("running SIMPLE BASELINE (threshold rule) ...")
    base = run_baseline_suite(engine, suite)
    base_score = score_suite(base)

    print("running AGENT SOLUTION ...")
    agent = run_agent_suite(engine, suite, full_workflow=not args.model_only)
    agent_score = score_suite(agent)

    rows = [
        ("Alert accuracy (primary)", "alert_accuracy", True),
        ("F1", "f1", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("Cause accuracy (all real failures)", "cause_accuracy", True),
        ("Cause accuracy (of those alerted)", "cause_accuracy_when_alerted",
         True),
        ("Hard-case accuracy", "hard_case_accuracy", True),
        ("False alarms on nuisance cases", "false_alarm_rate_on_nuisance_cases",
         True),
    ]
    print("\n" + "=" * 84)
    print(f"{'METRIC':<38}{'SIMPLE BASELINE':>16}{'AGENT SOLUTION':>16}{'CHANGE':>13}")
    print("-" * 84)
    for label, key, as_pct in rows:
        b, a = base_score.get(key), agent_score.get(key)
        if b is None and a is None:
            continue
        fb, fa = (_pct(b), _pct(a)) if as_pct else (b, a)
        delta = ("--" if (b is None or a is None)
                 else f"{(a - b) * 100:+.1f} pp")
        print(f"{label:<38}{fb:>16}{fa:>16}{delta:>13}")
    print(f"{'Median seconds per case':<38}"
          f"{base_score['median_seconds_per_case']:>16.3f}"
          f"{agent_score['median_seconds_per_case']:>16.3f}"
          f"{'':>13}")
    print("=" * 84)

    print("\nCapabilities the baseline does not have "
          "(reported, not scored as zero):")
    print(f"  verification verdicts      : "
          f"{agent_score.get('verification_verdicts', 'n/a')}")
    print(f"  cases cleared to act       : "
          f"{agent_score.get('cases_cleared_to_act', 'n/a')} / {len(suite)}")
    print(f"  mean simulated risk drop   : "
          f"{agent_score.get('mean_simulated_risk_reduction', 'n/a')}")

    print("\nPer-category alert accuracy:")
    print(f"  {'category':<34}{'baseline':>10}{'agent':>10}")
    for cat in sorted(agent_score["by_category"]):
        b = base_score["by_category"].get(cat, {}).get("accuracy")
        a = agent_score["by_category"][cat]["accuracy"]
        n = agent_score["by_category"][cat]["n"]
        print(f"  {cat + f' (n={n})':<34}{_pct(b):>10}{_pct(a):>10}")

    out = REPORT_DIR / "evaluation.json"
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "suite": summarise(suite),
        "model_under_test": {"kind": engine.bundle.kind,
                             "feature_set": engine.bundle.feature_set,
                             "threshold": engine.bundle.threshold},
        "baseline": base_score,
        "agent": agent_score,
        "llm_usage": engine.ctx.usage.to_dict(),
        "wall_clock_s": round(time.time() - t0, 1),
        "cases": {"baseline": [r.to_dict() for r in base],
                  "agent": [r.to_dict() for r in agent]},
    }, indent=2, default=str))
    print(f"\nwrote -> {out}   ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
