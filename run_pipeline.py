#!/usr/bin/env python
"""End-to-end incident workflow for one machine.

    python run_pipeline.py                          # riskiest machine in test
    python run_pipeline.py --machine PUMP-020
    python run_pipeline.py --scenario S01
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from predictops.agents.orchestrator import PredictOpsEngine
from predictops.evaluation.scenarios import load_suite
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import get_provider
from predictops.ml.dataset import prepare
from predictops.reporting import render_incident


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--machine", default=None)
    p.add_argument("--timestamp", default=None)
    p.add_argument("--scenario", default=None, help="scenario id, e.g. S01")
    p.add_argument("--provider", default=None)
    p.add_argument("--json", action="store_true", help="print raw JSON")
    args = p.parse_args()

    data = prepare()
    engine = PredictOpsEngine(
        data=data, store=ExperimentStore(), provider=get_provider(args.provider),
        run_id=f"incident-{datetime.now():%Y%m%d-%H%M%S}", verbose=True)
    engine.load_bundle()

    machine, timestamp = args.machine, args.timestamp
    if args.scenario:
        sc = next(s for s in load_suite() if s.id == args.scenario)
        machine, timestamp = sc.machine_id, sc.timestamp
        print(f"scenario {sc.id}: {sc.category} ({sc.difficulty}) -- {sc.notes}\n")
    if machine is None:
        fleet = engine.fleet_scores()
        top = fleet.dropna(subset=["failure_probability"]).iloc[0]
        machine = top["machine_id"]
        timestamp = fleet.attrs["timestamp"]
        print(f"no machine given; picked the highest-risk machine in the test "
              f"period: {machine} at {timestamp}\n")
    if timestamp is None:
        g = data.df[data.df.machine_id == machine]
        timestamp = g["timestamp"].max()

    report = engine.run_incident(machine, pd.Timestamp(timestamp))

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(render_incident(report.to_dict()))

    path = report.save()
    print(f"\nreport -> {path}")
    print(f"trajectory: {len(report.trajectory)} agent steps recorded "
          f"under run_id={report.run_id}")


if __name__ == "__main__":
    main()
