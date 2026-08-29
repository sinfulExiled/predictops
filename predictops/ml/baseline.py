"""The baseline a plant already has: fixed alarm thresholds.

This is deliberately a *strong* version of the naive approach, not a straw man.
It gets:

* per-machine normalisation (each machine's own healthy history sets its
  baseline, which is better than a single plant-wide limit),
* three channels OR-ed together, not one,
* the same validation split every model uses, to tune its trip level.

What it cannot do is condition one channel on another -- it has no way to say
"vibration is high *but load is also high, so this is fine*".  That is the gap
the sequence models are asked to close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ["vibration", "temperature", "current"]


@dataclass
class ThresholdBaseline:
    """Alert when any channel exceeds k standard deviations of its own history.

    The per-machine mean/std are estimated causally with an expanding window,
    so the rule never sees the future either.
    """

    k: float = 3.0
    channels: tuple[str, ...] = tuple(CHANNELS)
    min_history: int = 36

    @staticmethod
    def _causal_z(df: pd.DataFrame, channels) -> np.ndarray:
        """Expanding-window z-score per machine -- uses past samples only."""
        df = df.sort_values(["machine_id", "timestamp"])
        g = df.groupby("machine_id", sort=False)
        zs = []
        for c in channels:
            mean = g[c].transform(lambda s: s.shift(1).expanding(min_periods=12).mean())
            std = g[c].transform(lambda s: s.shift(1).expanding(min_periods=12).std())
            zs.append(((df[c] - mean) / (std + 1e-6)).to_numpy())
        stack = np.vstack(zs)
        # early samples have no history yet -> score them as "quiet"
        all_nan = np.isnan(stack).all(axis=0)
        stack = np.where(np.isnan(stack), -np.inf, stack)
        out = stack.max(axis=0)
        out[all_nan] = np.nan
        return out

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Return a monotone score so ROC/PR-AUC are comparable with the models."""
        z = self._causal_z(df, self.channels)
        return np.nan_to_num(z, nan=-5.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.k).astype(int)

    def fit(self, val_df: pd.DataFrame,
            grid: np.ndarray | None = None) -> "ThresholdBaseline":
        """Pick the trip level that maximises validation F1."""
        from sklearn.metrics import f1_score

        grid = np.arange(0.5, 6.01, 0.1) if grid is None else grid
        s = self.score(val_df)
        y = val_df.sort_values(["machine_id", "timestamp"])["label"].to_numpy()
        best_k, best_f1 = self.k, -1.0
        for k in grid:
            f = f1_score(y, (s >= k).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_k = f, float(k)
        self.k = best_k
        return self


def run_baseline(prepared, split: str = "test") -> dict:
    """Fit on validation, score on `split`.  Returns arrays plus the threshold."""
    val = prepared.split("val")
    val = val[(val.is_downtime == 0) & (val.sensor_dropout == 0)]
    model = ThresholdBaseline().fit(val)

    d = prepared.split(split)
    d = d[(d.is_downtime == 0) & (d.sensor_dropout == 0)]
    d = d.sort_values(["machine_id", "timestamp"])
    return {
        "score": model.score(d),
        "y": d["label"].to_numpy().astype(int),
        "machine_id": d["machine_id"].to_numpy(),
        "timestamp": d["timestamp"].to_numpy(),
        "threshold": model.k,
        "model": model,
    }
