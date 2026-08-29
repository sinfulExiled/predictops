"""Splits, imputation, scaling and window construction.

Leakage controls, in one place so they can be audited:

1.  Splits are **chronological**, not random.  A model is always trained on
    the past and scored on the future.
2.  A **purge gap** of one prediction horizon is dropped at each split
    boundary, so no training row's label window overlaps the next split.
3.  Imputation is forward-fill within a machine (a real historian returns the
    last known value); the fallback constant is the *training* median.
4.  The scaler is fitted on the training split only.
5.  Windows never span a downtime block, and never cross machines.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    LOOKBACK_STEPS,
    PURGE_STEPS,
    STEPS_PER_HOUR,
    TRAIN_FRACTION,
    VAL_FRACTION,
)

MAX_FFILL_STEPS = 3  # ~30 min of holding the last known value


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def make_time_splits(df: pd.DataFrame,
                     train_fraction: float = TRAIN_FRACTION,
                     val_fraction: float = VAL_FRACTION,
                     purge_steps: int = PURGE_STEPS) -> pd.DataFrame:
    """Label each row train/val/test/purge by timestamp."""
    df = df.copy()
    stamps = np.sort(df["timestamp"].unique())
    n = len(stamps)
    i_train = int(n * train_fraction)
    i_val = int(n * (train_fraction + val_fraction))

    split = np.full(n, "test", dtype=object)
    split[:i_train] = "train"
    split[i_train:i_val] = "val"
    # purge across both boundaries: the horizon *before* a boundary leaks
    split[max(i_train - purge_steps, 0):i_train] = "purge"
    split[max(i_val - purge_steps, 0):i_val] = "purge"

    lookup = pd.Series(split, index=pd.Index(stamps, name="timestamp"))
    df["split"] = df["timestamp"].map(lookup).astype(str)
    return df


# --------------------------------------------------------------------------
# imputation and scaling
# --------------------------------------------------------------------------
def impute(df: pd.DataFrame, columns: list[str],
           fallback: dict[str, float] | None = None
           ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Causal fill: hold the last reading briefly, then fall back to a constant."""
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("machine_id", sort=False)
    for c in columns:
        if c in df.columns:
            df[c] = g[c].transform(lambda s: s.ffill(limit=MAX_FFILL_STEPS))
    if fallback is None:
        train = df[df.get("split", "train") == "train"] if "split" in df else df
        fallback = {c: float(train[c].median(skipna=True))
                    if c in df.columns and np.isfinite(train[c].median(skipna=True))
                    else 0.0 for c in columns}
    for c in columns:
        if c in df.columns:
            df[c] = df[c].fillna(fallback.get(c, 0.0))
    return df, fallback


@dataclass
class Scaler:
    """Standardiser fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "Scaler":
        x = df[columns].to_numpy(dtype=np.float64)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std[~np.isfinite(std) | (std < 1e-6)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean=mean, std=std, columns=list(columns))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "columns": self.columns}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]), d["columns"])


# --------------------------------------------------------------------------
# usable rows and windows
# --------------------------------------------------------------------------
def usable_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows a prediction may be made on: machine running, real data present."""
    # pandas returns read-only views; copy before combining in place
    ok = (df["is_downtime"] == 0).to_numpy(copy=True)
    if "sensor_dropout" in df.columns:
        ok &= (df["sensor_dropout"] == 0).to_numpy()
    return ok


@dataclass
class WindowIndex:
    """Lazily-sliced sequence windows.

    Holds one contiguous 2-D array per machine and a list of valid window end
    positions.  Materialising every window as a dense 3-D tensor would cost
    hundreds of megabytes; slicing on demand costs nothing.
    """

    machine_ids: list[str]
    matrices: dict[str, np.ndarray]          # machine -> (T, F) float32
    ends: np.ndarray                          # (N,) end row within machine
    which: np.ndarray                         # (N,) index into machine_ids
    labels: np.ndarray                        # (N,) int8
    timestamps: np.ndarray                    # (N,) datetime64
    lookback: int
    columns: list[str]

    def __len__(self) -> int:
        return len(self.ends)

    def window(self, i: int) -> np.ndarray:
        m = self.matrices[self.machine_ids[self.which[i]]]
        e = int(self.ends[i])
        return m[e - self.lookback + 1:e + 1]

    def stack(self, idx: np.ndarray | None = None) -> np.ndarray:
        """Materialise a subset (used for small eval batches only)."""
        idx = np.arange(len(self)) if idx is None else idx
        return np.stack([self.window(int(i)) for i in idx])

    def machine_of(self, i: int) -> str:
        return self.machine_ids[self.which[i]]


def build_window_index(df: pd.DataFrame, columns: list[str],
                       scaler: Scaler | None = None,
                       lookback: int = LOOKBACK_STEPS,
                       split: str | None = None) -> WindowIndex:
    """Build valid sequence windows for one split.

    A window is valid when the whole lookback lies inside one machine, contains
    no downtime sample, and its final row is a usable prediction point.
    """
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    machine_ids: list[str] = []
    matrices: dict[str, np.ndarray] = {}
    ends, which, labels, stamps = [], [], [], []

    for mi, (mid, grp) in enumerate(df.groupby("machine_id", sort=True)):
        x = grp[columns].to_numpy(dtype=np.float32)
        if scaler is not None:
            x = scaler.transform(x.astype(np.float64))
        machine_ids.append(mid)
        matrices[mid] = np.ascontiguousarray(x, dtype=np.float32)

        down = (grp["is_downtime"] == 1).to_numpy()
        # cumulative downtime count -> a window is clean when the count is flat
        cum_down = np.concatenate([[0], np.cumsum(down)])
        n = len(grp)
        idx = np.arange(n)
        long_enough = idx >= lookback - 1
        clean = np.zeros(n, dtype=bool)
        valid_idx = idx[long_enough]
        clean[valid_idx] = (cum_down[valid_idx + 1]
                            - cum_down[valid_idx - lookback + 1]) == 0

        ok = long_enough & clean & usable_mask(grp)
        if split is not None and "split" in grp.columns:
            ok &= (grp["split"] == split).to_numpy()

        sel = idx[ok]
        ends.append(sel)
        which.append(np.full(len(sel), mi, dtype=np.int32))
        labels.append(grp["label"].to_numpy()[sel])
        stamps.append(grp["timestamp"].to_numpy()[sel])

    return WindowIndex(
        machine_ids=machine_ids, matrices=matrices,
        ends=np.concatenate(ends) if ends else np.array([], dtype=int),
        which=np.concatenate(which) if which else np.array([], dtype=np.int32),
        labels=np.concatenate(labels) if labels else np.array([], dtype=np.int8),
        timestamps=(np.concatenate(stamps) if stamps
                    else np.array([], dtype="datetime64[s]")),
        lookback=lookback, columns=list(columns),
    )


def balanced_subsample(labels: np.ndarray, neg_per_pos: float, seed: int
                       ) -> np.ndarray:
    """Keep every positive and a random share of negatives.

    Training only -- evaluation always runs on the untouched distribution.
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    k = min(len(neg), int(len(pos) * neg_per_pos))
    keep_neg = rng.choice(neg, size=k, replace=False) if k > 0 else neg[:0]
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return idx


def horizon_event_table(failures: pd.DataFrame) -> pd.DataFrame:
    """Failure events with parsed timestamps, for event-level scoring."""
    f = failures.copy()
    for c in ("degradation_start", "failure_time", "repair_time"):
        f[c] = pd.to_datetime(f[c])
    return f
