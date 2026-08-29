"""A reproducible counterfactual environment.

Given the telemetry history of one machine, this rolls the machine forward a
few hours under a chosen intervention and returns the synthetic telemetry.
The risk model then scores that synthetic future.  So the headline number --
"risk 0.87 -> 0.39" -- is produced by the *trained model*, reading telemetry it
has never seen, not by a rule that decides how much an action ought to help.

Honest limits, stated plainly because they bound what the number means:

* Degradation is continued by **trend persistence**: the recent per-channel
  slope is extrapolated with damping.  A real fault can accelerate
  non-linearly, so a long rollout understates risk.
* The intervention effect model is a first-order approximation drawn from the
  same physical relationships as the data generator.  On real telemetry those
  coefficients would have to be fitted from historical work orders.
* Therefore the simulation is a **ranking tool** -- it compares candidate
  actions against each other and against doing nothing under identical
  assumptions.  Its absolute risk numbers are labelled `simulated` everywhere
  and must not be read as a forecast.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import STEPS_PER_HOUR
from .interventions import Intervention

# Channels the rollout evolves directly; everything else is derived.
ROLLOUT_CHANNELS = ["temperature", "vibration", "pressure", "rpm", "current",
                    "voltage", "load", "humidity", "ambient_temp"]

# Trend damping per step: a fault's recent slope is continued but not
# extrapolated indefinitely.
TREND_DAMPING = 0.97


@dataclass
class RolloutResult:
    frame: pd.DataFrame          # history + synthetic future
    synthetic_from: pd.Timestamp
    steps: int
    intervention_id: str
    is_simulated: bool = True


def _recent_slope(values: np.ndarray, hours: float = 2.0) -> float:
    n = int(hours * STEPS_PER_HOUR)
    s = values[-n:]
    s = s[np.isfinite(s)]
    if len(s) < 4:
        return 0.0
    return float(np.polyfit(np.arange(len(s)), s, 1)[0])


def rollout(history: pd.DataFrame, hours: float,
            intervention: Intervention | None = None,
            resolution_minutes: int = 60 // STEPS_PER_HOUR) -> RolloutResult:
    """Continue the machine `hours` into the future.

    `intervention=None` is the do-nothing arm.  Both arms use the same trend
    continuation, so the difference between them isolates the action.
    """
    history = history.sort_values("timestamp").reset_index(drop=True)
    steps = max(int(round(hours * STEPS_PER_HOUR)), 1)
    last = history.iloc[-1]
    step_delta = pd.Timedelta(minutes=resolution_minutes)

    slopes = {c: _recent_slope(history[c].to_numpy(dtype=float))
              for c in ROLLOUT_CHANNELS if c in history.columns}
    current = {c: float(last[c]) for c in slopes}

    effects = dict(intervention.effects) if intervention else {}
    settle_steps = max(int(round(
        (intervention.settle_hours if intervention else 0.0) * STEPS_PER_HOUR)), 1)

    rows = []
    for k in range(1, steps + 1):
        # 1. continue the underlying trend (damped)
        for c in current:
            current[c] += slopes[c] * (TREND_DAMPING ** k)

        row = dict(last)
        row["timestamp"] = last["timestamp"] + step_delta * k
        for c, v in current.items():
            row[c] = v

        # 2. apply the intervention, ramped in over its settle time
        if effects:
            ramp = min(k / settle_steps, 1.0)
            for ch, eff in effects.items():
                if ch not in row:
                    continue
                base = float(row[ch])
                if "mul" in eff:
                    target = base * float(eff["mul"])
                elif "add" in eff:
                    target = base + float(eff["add"])
                else:
                    continue
                row[ch] = base + (target - base) * ramp

            # 3. propagate a load change through the channels it drives.
            # Same first-order relationships the plant model uses: heat and
            # current scale with duty, vibration less so.
            if "load" in effects:
                f = float(effects["load"].get("mul", 1.0))
                ramped = 1.0 + (f - 1.0) * min(k / settle_steps, 1.0)
                amb = float(row.get("ambient_temp", 20.0))
                excess = float(row["temperature"]) - amb
                row["temperature"] = amb + excess * (0.34 + 0.66 * ramped)
                row["current"] = float(row["current"]) * (0.28 + 0.74 * ramped)
                row["vibration"] = float(row["vibration"]) * (0.72 + 0.46 * ramped) / 1.18
                if float(row.get("pressure", 0.0)) > 0:
                    row["pressure"] = float(row["pressure"]) * (0.55 + 0.5 * ramped)

        row["power"] = (np.sqrt(3.0) * float(row["voltage"])
                        * float(row["current"]) * 0.86 / 1000.0)
        row["operating_hours"] = float(last["operating_hours"]) + k / STEPS_PER_HOUR
        row["is_downtime"] = 0
        row["sensor_dropout"] = 0
        row["label"] = 0                    # unknown; never used for scoring
        rows.append(row)

    future = pd.DataFrame(rows)
    frame = pd.concat([history, future], ignore_index=True)
    return RolloutResult(
        frame=frame, synthetic_from=future["timestamp"].iloc[0], steps=steps,
        intervention_id=intervention.id if intervention else "no_action")


def score_rollout(result: RolloutResult, bundle, machine_id: str) -> dict:
    """Re-derive features on the counterfactual and score it with the model."""
    from ..ml.bundle import window_for
    from ..ml.features import add_causal_features
    from ..ml.training import static_index

    enriched = add_causal_features(result.frame.copy())
    end = enriched["timestamp"].max()

    if bundle.is_sequence():
        x, raw = window_for(enriched, machine_id, end, bundle.channels,
                            bundle.scaler, bundle.lookback)
        prob = float(bundle.score_windows(x[None, ...],
                                          np.array([static_index(machine_id)]))[0])
    else:
        row = enriched[enriched["timestamp"] == end]
        x = row[bundle.tabular_columns].to_numpy(dtype=np.float32)
        prob = float(bundle.score_rows(x)[0])
        raw = enriched.tail(bundle.lookback)

    tail = raw.tail(STEPS_PER_HOUR)
    return {
        "failure_probability_simulated": round(prob, 4),
        "at_timestamp": str(end),
        "is_simulated": True,
        "channels": {c: round(float(tail[c].mean()), 3)
                     for c in ("temperature", "vibration", "current",
                               "pressure", "load")
                     if c in tail.columns},
    }
