#!/usr/bin/env python
"""Optional third baseline: telemetry summary -> one LLM prompt -> diagnosis.

The hackathon brief lists "one direct prompt with basic instructions" as a
reasonable baseline, and this project's central architectural claim is that a
language model should not be the predictor. That claim deserves to be measured
rather than asserted, so this script measures it on exactly the same scenario
suite the threshold baseline and the agent workflow are scored on.

    export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY
    python llm_baseline.py --provider anthropic

It deliberately **refuses to run on MockProvider**: a mocked language model
would produce numbers that look like an LLM baseline and are not one, and this
project does not publish figures that did not come out of a real execution.

Cost: 45 cases x roughly 1.5k input / 150 output tokens. Under $1 on Claude
Opus 5 at the time of writing; the exact spend is metered and printed.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import pandas as pd

from predictops.config import HORIZON_HOURS, REPORT_DIR, STEPS_PER_HOUR
from predictops.data.schemas import FAILURE_MODES
from predictops.evaluation.scenario_runner import CaseResult, score_suite
from predictops.evaluation.scenarios import load_suite
from predictops.llm.provider import UsageMeter, get_provider
from predictops.ml.dataset import prepare

SYSTEM = (
    "You are a reliability engineer monitoring industrial machines. You will "
    "be given a summary of one machine's recent telemetry. Decide whether the "
    "machine will suffer a failure within the next "
    f"{HORIZON_HOURS} hours, and if so which failure mode. Be decisive but do "
    "not raise an alarm for normal operating variation such as a production "
    "load change or a hot day."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "will_fail_within_horizon": {"type": "boolean"},
        "failure_type": {
            "type": "string",
            "enum": sorted(FAILURE_MODES) + ["none"],
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["will_fail_within_horizon", "failure_type", "confidence",
                 "reasoning"],
    "additionalProperties": False,
}

CHANNELS = ["temperature", "vibration", "pressure", "rpm", "current",
            "voltage", "load", "ambient_temp"]


def summarise_window(window: pd.DataFrame) -> str:
    """The telemetry, as text. Hourly means keep the prompt readable."""
    lines = [f"Machine: {window['machine_id'].iloc[0]}",
             f"Window: {window['timestamp'].iloc[0]} to "
             f"{window['timestamp'].iloc[-1]} (6 hours, 10-minute samples)",
             f"Operating hours since last service: "
             f"{window['operating_hours'].iloc[-1]:.0f}",
             "",
             "Hourly means:"]
    hourly = window.set_index("timestamp").resample("1h")[CHANNELS].mean()
    header = "  time   " + "".join(f"{c[:9]:>11}" for c in CHANNELS)
    lines.append(header)
    for ts, row in hourly.iterrows():
        cells = "".join(f"{row[c]:>11.2f}" for c in CHANNELS)
        lines.append(f"  {str(ts)[11:16]}  {cells}")

    first = window.head(STEPS_PER_HOUR)[CHANNELS].mean()
    last = window.tail(STEPS_PER_HOUR)[CHANNELS].mean()
    lines.append("")
    lines.append("Change over the window (first hour -> last hour):")
    for c in CHANNELS:
        if abs(first[c]) > 1e-9:
            lines.append(f"  {c:<14}{first[c]:>9.2f} -> {last[c]:>9.2f}  "
                         f"({(last[c] - first[c]) / abs(first[c]) * 100:+.1f}%)")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None,
                   help="anthropic | openai (mock is refused)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    provider = get_provider(args.provider)
    if provider.name == "mock":
        raise SystemExit(
            "Refusing to run: no LLM provider is available, and a mocked "
            "language model is not an LLM baseline.\n"
            "Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) and pass "
            "--provider anthropic.")

    data = prepare()
    suite = load_suite()
    if args.limit:
        suite = suite[:args.limit]
    print(f"LLM baseline: {provider.name}/{provider.model} on "
          f"{len(suite)} scenarios\n")

    usage = UsageMeter()
    results: list[CaseResult] = []
    t0 = time.time()

    for i, sc in enumerate(suite, 1):
        c0 = time.time()
        g = data.df[(data.df.machine_id == sc.machine_id)
                    & (data.df.timestamp <= pd.Timestamp(sc.timestamp))]
        window = g.sort_values("timestamp").tail(6 * STEPS_PER_HOUR)

        res = provider.structured(
            SYSTEM, summarise_window(window), SCHEMA,
            fallback={"will_fail_within_horizon": False, "failure_type": "none",
                      "confidence": 0.0, "reasoning": "provider error"})
        usage.add("llm_baseline", res)
        if res.used_fallback and res.error:
            print(f"  [{i:>2}/{len(suite)}] {sc.id} provider error: {res.error}")

        alert = bool(res.data.get("will_fail_within_horizon", False))
        ftype = res.data.get("failure_type", "none")
        ftype = None if ftype in ("none", "") else ftype

        results.append(CaseResult(
            scenario_id=sc.id, category=sc.category, difficulty=sc.difficulty,
            machine_id=sc.machine_id, timestamp=sc.timestamp,
            expected_alert=sc.expect_alert, expected_type=sc.expect_failure_type,
            alert=alert, probability=float(res.data.get("confidence", 0.0)),
            predicted_type=ftype if alert else None,
            correct_alert=(alert == sc.expect_alert),
            correct_type=(None if not sc.expect_alert
                          else bool(ftype == sc.expect_failure_type)),
            early_warning_h=(sc.hours_to_failure
                             if alert and sc.expect_alert else None),
            verification="not available (single prompt)",
            duration_s=time.time() - c0))
        print(f"  [{i:>2}/{len(suite)}] {sc.id} {sc.category:<30} "
              f"{'ALERT' if alert else 'quiet':<6} "
              f"{'ok' if results[-1].correct_alert else 'WRONG'}", flush=True)

    score = score_suite(results)
    print("\n" + "=" * 70)
    for k in ("n_cases", "alert_accuracy", "f1", "precision", "recall",
              "cause_accuracy", "hard_case_accuracy",
              "false_alarm_rate_on_nuisance_cases"):
        print(f"  {k:<38}{score.get(k)}")
    print("=" * 70)
    print(f"\nLLM usage: {json.dumps(usage.to_dict())}")
    print(f"Wall clock: {time.time() - t0:.1f}s")

    out = REPORT_DIR / "llm_baseline.json"
    out.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "provider": {"name": provider.name, "model": provider.model},
        "score": score,
        "usage": usage.to_dict(),
        "cases": [r.to_dict() for r in results],
    }, indent=2, default=str))
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
