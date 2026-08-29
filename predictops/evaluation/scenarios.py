"""The fixed scenario suite.

Built once from the test split, by rule, from a fixed seed -- so the same
cases come out on every machine, and neither the baseline nor the agent can be
tuned against a case it has not seen.

The suite deliberately over-samples the situations that separate a real
predictor from a threshold: nuisance load surges, hot weather, dropouts,
subtle and atypical degradations, and failures that develop too fast to catch.
A suite of easy cases would make both systems look good and tell us nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, HORIZON_HOURS, LOOKBACK_STEPS, STEPS_PER_HOUR

SUITE_PATH = DATA_DIR / "scenarios.json"


@dataclass
class Scenario:
    id: str
    category: str
    difficulty: str                 # easy | moderate | hard
    machine_id: str
    timestamp: str
    expect_alert: bool
    expect_failure_type: str | None
    hours_to_failure: float | None
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _usable(df: pd.DataFrame, machine_id: str, ts: pd.Timestamp,
            lookback: int = LOOKBACK_STEPS) -> bool:
    """A case is only fair if a model can actually score it."""
    g = df[(df.machine_id == machine_id) & (df.timestamp <= ts)]
    if len(g) < lookback:
        return False
    tail = g.tail(lookback)
    return bool(tail["is_downtime"].sum() == 0
                and int(tail["sensor_dropout"].iloc[-1]) == 0
                and tail["timestamp"].iloc[-1] == ts)


def build_suite(df: pd.DataFrame, failures: pd.DataFrame,
                split: str = "test", seed: int = 7,
                target: int = 32) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    d = df[df["split"] == split]
    lo, hi = d["timestamp"].min(), d["timestamp"].max()

    f = failures.copy()
    for c in ("failure_time", "degradation_start", "repair_time"):
        f[c] = pd.to_datetime(f[c])
    window = pd.Timedelta(hours=HORIZON_HOURS)
    in_split = f[(f["failure_time"] - window >= lo) & (f["failure_time"] <= hi)]

    out: list[Scenario] = []
    used: set[tuple[str, str]] = set()

    def add(cat, diff, mid, ts, alert, ftype, ttf, notes) -> bool:
        key = (mid, str(ts))
        if key in used or not _usable(df, mid, ts):
            return False
        used.add(key)
        out.append(Scenario(
            id=f"S{len(out) + 1:02d}", category=cat, difficulty=diff,
            machine_id=mid, timestamp=str(ts), expect_alert=alert,
            expect_failure_type=ftype,
            hours_to_failure=round(float(ttf), 2) if ttf is not None else None,
            notes=notes))
        return True

    def pick_before(ev, hours_before: float):
        """The sample `hours_before` hours ahead of a failure."""
        ts = ev.failure_time - pd.Timedelta(hours=hours_before)
        g = df[(df.machine_id == ev.machine_id) & (df.timestamp <= ts)]
        return g["timestamp"].iloc[-1] if len(g) else None

    # --- positives, grouped by what makes them hard ---------------------
    groups = [
        ("clear_degradation", "easy",
         in_split[(~in_split.is_sudden) & (~in_split.is_atypical)
                  & (in_split.severity_scale >= 0.8)], 3.0,
         "textbook degradation, late stage"),
        ("subtle_degradation", "hard",
         in_split[(in_split.severity_scale < 0.65) & (~in_split.is_sudden)], 3.0,
         "damped symptom amplitude -- the signal is there but small"),
        ("sudden_failure", "hard",
         in_split[in_split.is_sudden], 1.5,
         "degradation develops in under 2.5 h; little warning available"),
        ("atypical_pattern", "hard",
         in_split[in_split.is_atypical], 3.0,
         "the mode's signature channel is suppressed -- textbook rules miss it"),
        ("early_warning", "moderate",
         in_split[~in_split.is_sudden], 5.5,
         "near the far edge of the horizon; catching it is worth the most"),
        ("late_warning", "easy",
         in_split[~in_split.is_sudden], 1.0,
         "close to failure; symptoms should be unmistakable"),
    ]
    for cat, diff, subset, hours, note in groups:
        if subset.empty:
            continue
        take = subset.sample(min(len(subset), 4), random_state=seed)
        for _, ev in take.iterrows():
            ts = pick_before(ev, hours)
            if ts is None:
                continue
            add(cat, diff, ev.machine_id, ts, True, ev.failure_type, hours, note)

    # --- one case per failure type, so no mode is unrepresented ----------
    for ftype, grp in in_split.groupby("failure_type"):
        if any(s.expect_failure_type == ftype for s in out):
            continue
        ev = grp.iloc[0]
        ts = pick_before(ev, 3.0)
        if ts is not None:
            add(f"mode_{ftype}", "moderate", ev.machine_id, ts, True, ftype,
                3.0, f"coverage case for {ftype}")

    # --- negatives: the nuisances a threshold rule fires on ---------------
    neg = d[(d.label == 0) & (d.is_downtime == 0) & (d.sensor_dropout == 0)]

    def sample_neg(cat, diff, mask, note, n=4):
        pool = neg[mask]
        if pool.empty:
            return
        idx = rng.choice(len(pool), size=min(n * 3, len(pool)), replace=False)
        added = 0
        for i in idx:
            if added >= n:
                break
            r = pool.iloc[int(i)]
            if add(cat, diff, r.machine_id, r.timestamp, False, None, None, note):
                added += 1

    hot = neg["heatwave"] == 1
    sample_neg("hot_weather_no_failure", "hard", hot,
               "ambient heat wave raises temperature with no fault present")

    surge = (neg["load_surge"] == 1) & (neg["vib_delta_1h"] > 0)
    sample_neg("load_surge_no_failure", "hard", surge,
               "vibration and current rise because the duty rose, not the machine")

    # A dropout *inside* the lookback while the scored row itself is clean.
    # (A row whose own sample is missing is not scoreable and is already
    # excluded, so filtering on `sensor_dropout` here would match nothing.)
    gaps_in_window = (
        d.sort_values(["machine_id", "timestamp"])
         .groupby("machine_id")["sensor_dropout"]
         .transform(lambda x: x.rolling(LOOKBACK_STEPS, min_periods=1).sum()))
    d = d.assign(_dropouts_in_window=gaps_in_window)
    neg = neg.join(d["_dropouts_in_window"])
    sample_neg("missing_telemetry", "hard",
               neg["_dropouts_in_window"].fillna(0) > 0,
               "sensor dropout inside the lookback window, scored row clean",
               n=3)

    spike = neg["vibration"] > neg.groupby("machine_id")["vibration"].transform(
        "quantile", 0.995)
    sample_neg("sensor_spike_no_failure", "hard", spike,
               "isolated transducer glitch, not a trend", n=3)

    multi = ((neg["vib_z_6h"] > 1.5) & (neg["temp_excess_z_6h"] > 1.5)) \
        if "vib_z_6h" in neg.columns else None
    if multi is not None:
        sample_neg("multiple_anomalies_no_failure", "hard", multi,
                   "two channels are simultaneously elevated, still no failure",
                   n=3)

    quiet = (neg["load_surge"] == 0) & (neg["heatwave"] == 0)
    sample_neg("healthy_machine", "easy", quiet,
               "ordinary running -- must not alert", n=5)

    # top up with more healthy cases if the suite came out short
    if len(out) < target:
        sample_neg("healthy_machine", "easy", quiet,
                   "ordinary running -- must not alert", n=target - len(out))
    return out


def save_suite(suite: list[Scenario], path: Path = SUITE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([s.to_dict() for s in suite], indent=2))
    return path


def load_suite(path: Path = SUITE_PATH) -> list[Scenario]:
    return [Scenario(**d) for d in json.loads(path.read_text())]


def summarise(suite: list[Scenario]) -> dict:
    df = pd.DataFrame([s.to_dict() for s in suite])
    return {
        "n_cases": len(df),
        "n_positive": int(df["expect_alert"].sum()),
        "n_negative": int((~df["expect_alert"]).sum()),
        "by_difficulty": df["difficulty"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
    }
