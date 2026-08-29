#!/usr/bin/env python
"""Phase 5 -- run the model bake-off through the research agents.

    python run_experiments.py            # full run  (~15 min on 4 CPU cores)
    python run_experiments.py --quick    # fewer epochs, for a smoke test

Produces: every experiment recorded in the registry, the deployable model
bundle, and the improvement changelog.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from predictops.agents.orchestrator import PredictOpsEngine
from predictops.config import REPORT_DIR
from predictops.experiments.changelog import write_changelog
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import get_provider
from predictops.ml.bundle import BUNDLE_DIR
from predictops.ml.dataset import prepare


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fewer epochs")
    p.add_argument("--provider", default=None, help="mock | anthropic | openai")
    p.add_argument("--run-id", default=None)
    args = p.parse_args()

    run_id = args.run_id or f"research-{datetime.now():%Y%m%d-%H%M%S}"
    t0 = time.time()

    data = prepare()
    store = ExperimentStore()
    provider = get_provider(args.provider)
    engine = PredictOpsEngine(data=data, store=store, provider=provider,
                              run_id=run_id)
    print(f"run_id={run_id}  llm_provider={provider.name}/{provider.model}")
    print()

    print("[1/3] data scientist agent")
    print("[2/3] model research agent")
    research = engine.research(quick=args.quick)
    ds_out = research["data_scientist"]
    mr_out = research["model_research"]

    print()
    print(f"      leakage audit: {ds_out['leakage_audit']['verdict']}")
    print(f"      plan: {ds_out['recommended_plan']}")

    print()
    print("=" * 84)
    header = (f"{'STAGE':<13}{'MODEL':<34}{'VAL PR-AUC':>11}"
              f"{'TEST F1':>9}  DECISION")
    print(header)
    print("-" * 84)
    for e in mr_out["experiments"]:
        print(f"{e['experiment']:<13}{e['name'][:33]:<34}"
              f"{e['val_pr_auc']:>11.4f}{e['test_f1']:>9.4f}  {e['decision']}")
    print("=" * 84)

    sel = mr_out["selection"]
    print()
    print(f"SELECTED: {sel['name']}")
    print(f"  {sel['rationale']}")
    print(f"  (selection metric: {sel['selected_on']})")

    print()
    print(f"[3/3] deployable bundle -> {BUNDLE_DIR}")
    print(f"      kind={engine.bundle.kind} "
          f"feature_set={engine.bundle.feature_set} "
          f"threshold={engine.bundle.threshold:.4f}")

    changelog_path = write_changelog(store, run_id)
    print(f"      changelog -> {changelog_path}")

    print()
    print(f"LLM usage: {engine.ctx.usage.to_dict()}")
    print(f"Total wall clock: {time.time() - t0:.1f}s")

    out = REPORT_DIR / f"experiments_{run_id}.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "provider": {"name": provider.name, "model": provider.model},
        "data_scientist": ds_out,
        "model_research": mr_out,
        "bundle": {"kind": engine.bundle.kind,
                   "feature_set": engine.bundle.feature_set,
                   "threshold": engine.bundle.threshold,
                   "channels": engine.bundle.channels},
        "llm_usage": engine.ctx.usage.to_dict(),
        "wall_clock_s": round(time.time() - t0, 1),
    }, indent=2, default=str))
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
