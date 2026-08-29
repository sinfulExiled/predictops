"""Causal feature engineering.

Every feature here is computed from data at or before its own timestamp.  No
`center=True` rolling, no forward fill, no group statistics that span the
future.  `tests/test_features.py` enforces this by recomputing features from a
truncated history and checking the values are unchanged.

Two named feature sets exist so the model-research agent can ablate them:

    raw         -- the sensor channels as delivered by the plant
    engineered  -- raw + causal rolling statistics + load/ambient normalisation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import STEPS_PER_HOUR

# Channels the rolling statistics are computed over.
ROLLING_CHANNELS = ["temperature", "vibration", "current", "pressure",
                    "rpm", "load"]

# Rolling window lengths, in samples (1 h, 3 h, 6 h at 10-minute resolution).
WINDOWS = {"1h": 1 * STEPS_PER_HOUR, "3h": 3 * STEPS_PER_HOUR,
           "6h": 6 * STEPS_PER_HOUR}

RAW_FEATURES = [
    "temperature", "vibration", "pressure", "rpm", "current", "voltage",
    "power", "load", "humidity", "ambient_temp", "operating_hours",
]

# Channels fed to the sequence models, one value per timestep.
SEQUENCE_CHANNELS_RAW = RAW_FEATURES
SEQUENCE_CHANNELS_ENGINEERED = RAW_FEATURES + [
    "temp_excess", "vib_per_load", "cur_per_load", "power_per_load",
    "vib_z_6h", "temp_excess_z_6h", "cur_z_6h", "rpm_instability_1h",
    "vib_delta_1h", "temp_excess_delta_1h", "cur_delta_1h", "prs_delta_1h",
    "is_missing",
]


def _causal_roll(g: pd.Series, window: int, stat: str) -> pd.Series:
    """Backward-looking rolling stat; needs at least half a window of history."""
    r = g.rolling(window=window, min_periods=max(window // 2, 2))
    return getattr(r, stat)()


def derived_columns() -> list[str]:
    """Every column `add_causal_features` creates.

    Used to make the function idempotent: the simulation re-derives features on
    a frame that already carries them, and silently concatenating a second copy
    produces duplicate column names, which turns `df[col]` into a DataFrame and
    breaks arithmetic alignment much later with an unrelated-looking error.
    """
    cols = ["temp_excess", "vib_per_load", "cur_per_load", "power_per_load"]
    for ch in ROLLING_CHANNELS:
        for name in WINDOWS:
            cols += [f"{ch}_mean_{name}", f"{ch}_std_{name}"]
        cols.append(f"{ch}_max_6h")
    for ch in ("temp_excess",):
        cols += [f"{ch}_mean_6h", f"{ch}_std_6h"]
    cols += ["vib_z_6h", "temp_excess_z_6h", "cur_z_6h"]
    for alias in ("vib", "temp_excess", "cur", "prs"):
        cols += [f"{alias}_delta_1h", f"{alias}_delta_3h"]
    cols += ["rpm_instability_1h", "load_delta_1h", "vib_rise_without_load",
             "is_missing"]
    return cols


def add_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the engineered feature columns.  Input must be sorted by
    (machine_id, timestamp).  Idempotent: existing derived columns are
    recomputed, not duplicated."""
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    stale = [c for c in derived_columns() if c in df.columns]
    if stale:
        df = df.drop(columns=stale)
    g = df.groupby("machine_id", sort=False)

    load = df["load"].clip(lower=0.15)

    # --- domain normalisations ------------------------------------------
    # Heat above ambient separates a hot day from a hot machine.
    df["temp_excess"] = df["temperature"] - df["ambient_temp"]
    # Per-unit-load figures separate a busy machine from a sick one.
    df["vib_per_load"] = df["vibration"] / load
    df["cur_per_load"] = df["current"] / load
    df["power_per_load"] = df["power"] / load

    # --- rolling statistics ---------------------------------------------
    out: dict[str, pd.Series] = {}
    for ch in ROLLING_CHANNELS:
        s = df[ch]
        for name, w in WINDOWS.items():
            out[f"{ch}_mean_{name}"] = g[ch].transform(
                lambda x, w=w: _causal_roll(x, w, "mean"))
            out[f"{ch}_std_{name}"] = g[ch].transform(
                lambda x, w=w: _causal_roll(x, w, "std"))
        out[f"{ch}_max_6h"] = g[ch].transform(
            lambda x: _causal_roll(x, WINDOWS["6h"], "max"))
    df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)

    # --- normalised anomaly scores (value vs its own recent history) -----
    for ch, alias in [("vibration", "vib"), ("temp_excess", "temp_excess"),
                      ("current", "cur")]:
        if f"{ch}_mean_6h" not in df.columns:
            gg = df.groupby("machine_id", sort=False)[ch]
            df[f"{ch}_mean_6h"] = gg.transform(
                lambda x: _causal_roll(x, WINDOWS["6h"], "mean"))
            df[f"{ch}_std_6h"] = gg.transform(
                lambda x: _causal_roll(x, WINDOWS["6h"], "std"))
        df[f"{alias}_z_6h"] = ((df[ch] - df[f"{ch}_mean_6h"])
                               / (df[f"{ch}_std_6h"] + 1e-6))

    # --- rates of change --------------------------------------------------
    g = df.groupby("machine_id", sort=False)
    for ch, alias in [("vibration", "vib"), ("temp_excess", "temp_excess"),
                      ("current", "cur"), ("pressure", "prs")]:
        df[f"{alias}_delta_1h"] = g[ch].transform(
            lambda x: x - x.shift(STEPS_PER_HOUR))
        df[f"{alias}_delta_3h"] = g[ch].transform(
            lambda x: x - x.shift(3 * STEPS_PER_HOUR))

    # Speed instability: the tell for a motor losing control of its load.
    df["rpm_instability_1h"] = df["rpm_std_1h"] / (df["rpm_mean_1h"].abs() + 1e-6)

    # Is the *load* rising too?  If so a vibration rise is probably innocent.
    df["load_delta_1h"] = g["load"].transform(
        lambda x: x - x.shift(STEPS_PER_HOUR))
    df["vib_rise_without_load"] = (
        df["vib_delta_1h"].clip(lower=0) * (1.0 - df["load_delta_1h"].clip(lower=0) * 4)
    )

    df["is_missing"] = df[["vibration", "temperature", "current"]].isna().any(
        axis=1).astype(np.float32)
    return df


def engineered_feature_columns(df: pd.DataFrame) -> list[str]:
    """Tabular feature list for the engineered set."""
    exclude = {
        "machine_id", "timestamp", "machine_type", "site", "label",
        "time_to_failure_h", "horizon_failure_type", "degradation_active",
        "severity", "is_downtime", "load_surge", "heatwave",
        "latent_failure_type", "sensor_dropout", "split",
    }
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "raw":
        return [c for c in RAW_FEATURES if c in df.columns]
    if feature_set == "engineered":
        return engineered_feature_columns(df)
    raise ValueError(f"unknown feature set: {feature_set}")


def sequence_channels(feature_set: str) -> list[str]:
    if feature_set == "raw":
        return list(SEQUENCE_CHANNELS_RAW)
    if feature_set == "engineered":
        return list(SEQUENCE_CHANNELS_ENGINEERED)
    raise ValueError(f"unknown feature set: {feature_set}")
