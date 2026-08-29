"""Assemble the modelling dataset once and cache it.

Every model in the project consumes `PreparedData`, so the features, splits,
imputation and scaling are identical across the baseline, the tree models and
the sequence models.  That is what makes the comparison fair.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, LOOKBACK_STEPS
from ..data.generator import load_dataset
from ..data.preprocessing import (
    Scaler,
    build_window_index,
    impute,
    make_time_splits,
)
from .features import add_causal_features, feature_columns, sequence_channels

CACHE = DATA_DIR / "prepared.parquet"
CACHE_META = DATA_DIR / "prepared_meta.json"

# Files whose contents determine the feature values. The cache is keyed on
# these as well as on the data, because keying on the data alone let a stale
# cache survive a change to the feature code -- which silently reported a
# baseline F1 of 0.2588 while a clean checkout of the same commit produced
# 0.2581.
_FEATURE_SOURCES = ("ml/features.py", "data/preprocessing.py", "config.py")


def _code_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    h = hashlib.sha256()
    for rel in _FEATURE_SOURCES:
        f = root / rel
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class PreparedData:
    df: pd.DataFrame
    failures: pd.DataFrame
    machines: pd.DataFrame
    # The context agent builds its dossier from this; carrying it here keeps
    # every agent reading the same loaded copy.
    maintenance: pd.DataFrame = field(default_factory=pd.DataFrame)

    def split(self, name: str) -> pd.DataFrame:
        return self.df[self.df["split"] == name]

    def tabular(self, name: str, feature_set: str, scaler: Scaler | None = None):
        """(X, y, machine_ids, timestamps) for one split, usable rows only."""
        d = self.split(name)
        d = d[(d["is_downtime"] == 0) & (d["sensor_dropout"] == 0)]
        cols = feature_columns(self.df, feature_set)
        x = d[cols].to_numpy(dtype=np.float64)
        if scaler is not None:
            x = scaler.transform(x)
        return (x.astype(np.float32), d["label"].to_numpy().astype(np.int8),
                d["machine_id"].to_numpy(), d["timestamp"].to_numpy(), cols)

    def windows(self, name: str, feature_set: str, scaler: Scaler | None = None,
                lookback: int = LOOKBACK_STEPS):
        cols = sequence_channels(feature_set)
        cols = [c for c in cols if c in self.df.columns]
        return build_window_index(self.df, cols, scaler=scaler,
                                  lookback=lookback, split=name)

    def fit_scaler(self, columns: list[str]) -> Scaler:
        return Scaler.fit(self.split("train"), columns)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    skip = {"label", "time_to_failure_h", "degradation_active", "severity",
            "is_downtime", "load_surge", "heatwave", "sensor_dropout", "site"}
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def prepare(force: bool = False, data_dir: Path = DATA_DIR) -> PreparedData:
    """Load -> features -> splits -> impute.  Cached on disk."""
    raw = load_dataset(data_dir)
    if CACHE.exists() and CACHE_META.exists() and not force:
        meta = json.loads(CACHE_META.read_text())
        fresh = (meta.get("source_checksum")
                 == raw.manifest["checksums"]["telemetry"]
                 and meta.get("code_fingerprint") == _code_fingerprint())
        if fresh:
            df = pd.read_parquet(CACHE)
            return PreparedData(df, raw.failures, raw.machines,
                                raw.maintenance)

    df = add_causal_features(raw.telemetry)
    df = make_time_splits(df)
    df, fallback = impute(df, _numeric_columns(df))
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df.to_parquet(CACHE, index=False)
    CACHE_META.write_text(json.dumps({
        "source_checksum": raw.manifest["checksums"]["telemetry"],
        "code_fingerprint": _code_fingerprint(),
        "n_rows": int(len(df)),
        "n_features_engineered": len(feature_columns(df, "engineered")),
        "impute_fallback": {k: round(v, 6) for k, v in fallback.items()},
        "split_counts": df["split"].value_counts().to_dict(),
    }, indent=2))
    return PreparedData(df, raw.failures, raw.machines, raw.maintenance)
