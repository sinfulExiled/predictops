"""Shared evidence toolkit.

Every agent that makes a factual claim builds it here, so a claim means the
same thing wherever it appears and the verifier can re-derive it by name.

An evidence item is a fact plus its own recipe:

    {"id": "E1", "claim": "...", "channel": "vibration",
     "metric": "pct_change", "value": 178.34, "unit": "%",
     "recompute": {"fn": "pct_change", "channel": "vibration", "hours": 3.0}}

The two hypothesis advocates disagree about what the facts *mean*. They are
not allowed to disagree about the facts themselves, because both draw from
this module and the verifier re-runs the recipes against raw telemetry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import STEPS_PER_HOUR

DEFAULT_HOURS = 3.0

# Evidence is swept over several horizons, longest first. A fixed 3 h window
# missed slow degradations entirely: on a machine the model scored at 1.00,
# temperature had climbed steadily all window but only ~7% in the last three
# hours, so no channel cleared the materiality bar, the signature match came
# out at 0/3, and the verifier failed a correct and imminent diagnosis.
EVIDENCE_HORIZONS = (6.0, 3.0, 1.0)

# A movement smaller than this is noise, not evidence.
MATERIAL_PCT = 8.0
MATERIAL_ABS = {"temperature": 3.0, "temp_excess": 3.0, "load": 0.05}

# Channels quoted as evidence, with the unit they are reported in.
EVIDENCE_CHANNELS = {
    "vibration": "mm/s",
    "temperature": "deg C",
    "temp_excess": "deg C above ambient",
    "current": "A",
    "pressure": "bar",
    "voltage": "V",
    "load": "fraction",
    "rpm_instability_1h": "ratio",
}


# --------------------------------------------------------------------------
# recomputable primitives -- the verifier calls these same functions by name
# --------------------------------------------------------------------------
def pct_change(window: pd.DataFrame, channel: str,
               hours: float = DEFAULT_HOURS) -> float:
    """Change from the first hour of the comparison span to the last hour."""
    n = int(hours * STEPS_PER_HOUR)
    s = window[channel].to_numpy(dtype=float)[-n:]
    if len(s) < 2 * STEPS_PER_HOUR:
        return 0.0
    head = np.nanmean(s[:STEPS_PER_HOUR])
    tail = np.nanmean(s[-STEPS_PER_HOUR:])
    if not np.isfinite(head) or abs(head) < 1e-9:
        return 0.0
    return float((tail - head) / abs(head) * 100.0)


def abs_change(window: pd.DataFrame, channel: str,
               hours: float = DEFAULT_HOURS) -> float:
    n = int(hours * STEPS_PER_HOUR)
    s = window[channel].to_numpy(dtype=float)[-n:]
    if len(s) < 2 * STEPS_PER_HOUR:
        return 0.0
    return float(np.nanmean(s[-STEPS_PER_HOUR:]) - np.nanmean(s[:STEPS_PER_HOUR]))


def peak_ratio(window: pd.DataFrame, channel: str,
               hours: float = DEFAULT_HOURS) -> float:
    """Peak over median. A transducer glitch spikes; a trend does not."""
    n = int(hours * STEPS_PER_HOUR)
    s = window[channel].to_numpy(dtype=float)[-n:]
    s = s[np.isfinite(s)]
    if len(s) < 6:
        return 0.0
    med = float(np.median(s))
    if abs(med) < 1e-9:
        return 0.0
    return float(np.max(np.abs(s)) / abs(med))


def monotonicity(window: pd.DataFrame, channel: str,
                 hours: float = DEFAULT_HOURS) -> float:
    """Fraction of hourly steps moving the same way as the overall change.

    A developing fault climbs steadily; a load episode goes up and comes back.
    """
    n = int(hours * STEPS_PER_HOUR)
    s = window[channel].to_numpy(dtype=float)[-n:]
    s = s[np.isfinite(s)]
    if len(s) < 2 * STEPS_PER_HOUR:
        return 0.0
    hourly = np.array([s[i:i + STEPS_PER_HOUR].mean()
                       for i in range(0, len(s) - STEPS_PER_HOUR + 1,
                                      STEPS_PER_HOUR)])
    if len(hourly) < 2:
        return 0.0
    diffs = np.diff(hourly)
    overall = hourly[-1] - hourly[0]
    if abs(overall) < 1e-12 or len(diffs) == 0:
        return 0.0
    return float((np.sign(diffs) == np.sign(overall)).mean())


RECOMPUTE_FNS = {
    "pct_change": pct_change,
    "abs_change": abs_change,
    "peak_ratio": peak_ratio,
    "monotonicity": monotonicity,
}


# --------------------------------------------------------------------------
class EvidenceBuilder:
    """Accumulates evidence items with stable ids."""

    def __init__(self, window: pd.DataFrame, prefix: str = "E"):
        self.window = window
        self.prefix = prefix
        self.items: list[dict] = []

    def _next_id(self) -> str:
        return f"{self.prefix}{len(self.items) + 1}"

    def add(self, claim: str, channel: str, fn: str, value: float, unit: str,
            direction: str, hours: float = DEFAULT_HOURS) -> dict:
        w = self.window
        item = {
            "id": self._next_id(), "claim": claim, "channel": channel,
            "metric": fn, "value": round(float(value), 4), "unit": unit,
            "direction": direction,
            "source": (f"telemetry[{w['machine_id'].iloc[0]}, "
                       f"{w['timestamp'].iloc[-int(hours * STEPS_PER_HOUR)]}"
                       f" .. {w['timestamp'].iloc[-1]}]"),
            "recompute": {"fn": fn, "channel": channel, "hours": hours},
        }
        self.items.append(item)
        return item

    def measure(self, channel: str, fn: str,
                hours: float = DEFAULT_HOURS) -> float:
        """Compute without recording -- for advocates that need a number to
        reason with but have no claim to make about it."""
        return RECOMPUTE_FNS[fn](self.window, channel, hours)

    def channel_movements(self, horizons=EVIDENCE_HORIZONS) -> list[dict]:
        """Every channel that moved materially, over whichever horizon shows it.

        A slow fault and a fast one leave the same trace at different time
        scales, so each channel is checked at several horizons and reported at
        the one where its movement is largest relative to the materiality bar.
        One item per channel -- the same rise is not evidence three times.
        """
        if isinstance(horizons, (int, float)):
            horizons = (float(horizons),)

        for ch, unit in EVIDENCE_CHANNELS.items():
            if ch not in self.window.columns:
                continue
            use_abs = ch in MATERIAL_ABS
            bar = MATERIAL_ABS[ch] if use_abs else MATERIAL_PCT

            best = None
            for hours in horizons:
                pct = pct_change(self.window, ch, hours)
                delta = abs_change(self.window, ch, hours)
                magnitude = delta if use_abs else pct
                strength = abs(magnitude) / bar
                if strength >= 1.0 and (best is None or strength > best[0]):
                    best = (strength, hours, pct, delta, magnitude)
            if best is None:
                continue

            _, hours, pct, delta, magnitude = best
            direction = "rose" if magnitude > 0 else "fell"
            span = f"{hours:.0f} hour" + ("s" if hours != 1 else "")
            if use_abs:
                claim = (f"{ch.replace('_', ' ').capitalize()} {direction} "
                         f"{abs(delta):.1f} {unit} over the last {span}")
                self.add(claim, ch, "abs_change", delta, unit,
                         "up" if delta > 0 else "down", hours)
            else:
                claim = (f"{ch.capitalize()} {direction} {abs(pct):.0f}% over "
                         f"the last {span}")
                self.add(claim, ch, "pct_change", pct, "%",
                         "up" if pct > 0 else "down", hours)
        return self.items
