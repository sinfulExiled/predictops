"""The canonical evaluation set.

Sequence models can only score a row that has a full, clean lookback behind it;
tabular models and the threshold rule can score every row.  Scoring them on
different row sets would quietly hand one side an easier exam.

So the evaluation set is fixed once -- the valid sequence windows of a split --
and *every* model, baseline included, is scored on exactly those
(machine_id, timestamp) pairs.  Any model that cannot produce a score for one
of them is a bug, not a smaller test set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.preprocessing import WindowIndex


@dataclass
class EvaluationSet:
    machine_id: np.ndarray
    timestamp: np.ndarray
    y: np.ndarray
    split: str

    def __len__(self) -> int:
        return len(self.y)

    @property
    def key(self) -> pd.DataFrame:
        return pd.DataFrame({"machine_id": self.machine_id,
                             "timestamp": pd.to_datetime(self.timestamp)})

    @classmethod
    def from_windows(cls, wi: WindowIndex, split: str) -> "EvaluationSet":
        machines = np.array([wi.machine_of(int(i)) for i in range(len(wi))])
        return cls(machine_id=machines,
                   timestamp=pd.to_datetime(wi.timestamps).to_numpy(),
                   y=wi.labels.astype(int), split=split)

    def align(self, machine_id, timestamp, score) -> np.ndarray:
        """Reindex an arbitrary model's output onto the canonical rows.

        Missing rows are an error, not something to silently drop -- a model
        that skipped rows would otherwise look better than it is.
        """
        src = pd.DataFrame({"machine_id": np.asarray(machine_id),
                            "timestamp": pd.to_datetime(pd.Series(timestamp)),
                            "score": np.asarray(score, dtype=float)})
        src = src.drop_duplicates(subset=["machine_id", "timestamp"])
        merged = self.key.merge(src, on=["machine_id", "timestamp"], how="left")
        missing = int(merged["score"].isna().sum())
        if missing:
            raise ValueError(
                f"model produced no score for {missing}/{len(self)} canonical "
                f"evaluation rows; comparison would be unfair")
        return merged["score"].to_numpy()


def build_evaluation_set(prepared, feature_set: str, scaler, split: str,
                         lookback: int) -> tuple[EvaluationSet, WindowIndex]:
    wi = prepared.windows(split, feature_set, scaler, lookback=lookback)
    return EvaluationSet.from_windows(wi, split), wi
