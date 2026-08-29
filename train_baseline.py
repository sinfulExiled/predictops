#!/usr/bin/env python
"""Fit and score the threshold baseline (Phase 2)."""
from __future__ import annotations

import json

from predictops.config import REPORT_DIR
from predictops.ml.baseline import run_baseline
from predictops.ml.dataset import prepare
from predictops.ml.evaluation import full_report


def main() -> None:
    data = prepare()
    out = run_baseline(data, split="test")
    report = full_report(out["machine_id"], out["timestamp"], out["y"],
                         out["score"], out["threshold"], data.failures)
    report["model"] = "threshold_baseline"
    report["tuned_k"] = out["threshold"]
    path = REPORT_DIR / "baseline_test.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote -> {path}")


if __name__ == "__main__":
    main()
