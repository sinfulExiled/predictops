"""Phase 2 -- features must be causal, splits must be leak-free, and the
baseline must be a real (if weak) predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictops.config import GeneratorConfig, HORIZON_HOURS, PURGE_STEPS
from predictops.data.generator import PlantGenerator
from predictops.data.preprocessing import (
    Scaler,
    balanced_subsample,
    build_window_index,
    impute,
    make_time_splits,
)
from predictops.ml.features import (
    add_causal_features,
    feature_columns,
    sequence_channels,
)

SMALL = GeneratorConfig(n_machines=6, days=8, seed=11)


@pytest.fixture(scope="module")
def prepared():
    ds = PlantGenerator(SMALL).generate()
    df = add_causal_features(ds.telemetry)
    df = make_time_splits(df)
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df, _ = impute(df, cols)
    return df, ds


# --- causality -------------------------------------------------------------
def test_features_do_not_depend_on_the_future(prepared):
    """Recompute features from a truncated history: values must be unchanged.

    This is the test that would catch a `center=True` rolling window or a
    backfill -- the two easiest ways to leak in a time-series pipeline.
    """
    df, ds = prepared
    full = add_causal_features(ds.telemetry)

    mid = ds.telemetry["timestamp"].quantile(0.6)
    truncated = add_causal_features(
        ds.telemetry[ds.telemetry.timestamp <= mid].copy())

    cols = [c for c in feature_columns(full, "engineered")
            if c in truncated.columns]
    a = full[full.timestamp <= mid].sort_values(["machine_id", "timestamp"])
    b = truncated.sort_values(["machine_id", "timestamp"])
    assert len(a) == len(b)

    diff = (a[cols].to_numpy(dtype=float) - b[cols].to_numpy(dtype=float))
    worst = np.nanmax(np.abs(diff)) if diff.size else 0.0
    assert worst < 1e-9, f"future leaked into features (max delta {worst})"


def test_no_leaky_ground_truth_in_feature_list(prepared):
    df, _ = prepared
    cols = set(feature_columns(df, "engineered"))
    for banned in ("label", "severity", "degradation_active",
                   "time_to_failure_h", "load_surge", "heatwave"):
        assert banned not in cols, f"{banned} must never be a model feature"


def test_sequence_channels_exist(prepared):
    df, _ = prepared
    for c in sequence_channels("engineered"):
        assert c in df.columns, c


# --- splits ----------------------------------------------------------------
def test_splits_are_chronological_and_purged(prepared):
    df, _ = prepared
    order = {"train": 0, "purge": 1, "val": 2, "test": 3}
    bounds = df.groupby("split")["timestamp"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["val", "min"]
    assert bounds.loc["val", "max"] < bounds.loc["test", "min"]
    assert "purge" in bounds.index
    assert set(df["split"]) <= set(order)


def test_purge_gap_covers_the_horizon(prepared):
    df, _ = prepared
    stamps = np.sort(df["timestamp"].unique())
    purged = df.loc[df.split == "purge", "timestamp"].nunique()
    # two boundaries, each dropping one horizon
    assert purged >= PURGE_STEPS, f"{purged} purged stamps < {PURGE_STEPS}"


def test_no_machine_appears_in_only_one_split(prepared):
    """Chronological splits keep every machine everywhere -- that is intended,
    and means the test measures forecasting, not machine memorisation."""
    df, _ = prepared
    for s in ("train", "val", "test"):
        assert df[df.split == s]["machine_id"].nunique() == SMALL.n_machines


# --- scaling ---------------------------------------------------------------
def test_scaler_is_fitted_on_train_only(prepared):
    df, _ = prepared
    cols = feature_columns(df, "engineered")[:8]
    s_train = Scaler.fit(df[df.split == "train"], cols)
    s_all = Scaler.fit(df, cols)
    assert not np.allclose(s_train.mean, s_all.mean), \
        "train-only and all-data scalers are identical; check the split filter"


def test_scaler_roundtrip(prepared):
    df, _ = prepared
    cols = feature_columns(df, "engineered")[:5]
    s = Scaler.fit(df[df.split == "train"], cols)
    s2 = Scaler.from_dict(s.to_dict())
    x = df[cols].to_numpy(dtype=float)[:100]
    assert np.allclose(s.transform(x), s2.transform(x))


# --- windows ---------------------------------------------------------------
def test_windows_are_clean_and_correctly_shaped(prepared):
    df, _ = prepared
    cols = sequence_channels("engineered")
    scaler = Scaler.fit(df[df.split == "train"], cols)
    wi = build_window_index(df, cols, scaler=scaler, lookback=12, split="train")
    assert len(wi) > 0
    w = wi.window(0)
    assert w.shape == (12, len(cols))
    assert np.isfinite(w).all()


def test_windows_never_span_downtime(prepared):
    df, _ = prepared
    cols = sequence_channels("engineered")
    wi = build_window_index(df, cols, lookback=12, split=None)
    down = {(m, i) for m, g in df.groupby("machine_id")
            for i in np.flatnonzero((g["is_downtime"] == 1).to_numpy())}
    for j in range(0, len(wi), max(len(wi) // 300, 1)):
        m = wi.machine_of(j)
        e = int(wi.ends[j])
        for k in range(e - wi.lookback + 1, e + 1):
            assert (m, k) not in down, "window contains a downtime sample"


def test_window_labels_match_the_end_row(prepared):
    df, _ = prepared
    cols = sequence_channels("engineered")
    wi = build_window_index(df, cols, lookback=12, split="train")
    d = df.sort_values(["machine_id", "timestamp"])
    for j in range(0, len(wi), max(len(wi) // 200, 1)):
        m = wi.machine_of(j)
        g = d[d.machine_id == m]
        assert g["label"].to_numpy()[int(wi.ends[j])] == wi.labels[j]


def test_balanced_subsample_keeps_every_positive():
    y = np.array([0] * 900 + [1] * 100)
    idx = balanced_subsample(y, neg_per_pos=3.0, seed=0)
    assert (y[idx] == 1).sum() == 100
    assert (y[idx] == 0).sum() == 300


# --- imputation ------------------------------------------------------------
def test_imputation_removes_nans_without_using_the_future(prepared):
    df, _ = prepared
    cols = ["vibration", "temperature", "current"]
    assert df[cols].isna().sum().sum() == 0

    raw = PlantGenerator(SMALL).generate().telemetry
    filled, fallback = impute(raw.copy(), cols)
    mid = filled.machine_id.iloc[0]
    one = filled[filled.machine_id == mid]

    # Same machine, same fallback constants, but only the first 200 samples
    # are visible.  Forward fill must produce identical values -- if it were
    # a backfill or an interpolation, withholding the future would change them.
    head = impute(raw[raw.machine_id == mid].head(200).copy(), cols,
                  fallback=fallback)[0]
    assert np.allclose(one[cols].to_numpy(dtype=float)[:200],
                       head[cols].to_numpy(dtype=float)[:200], equal_nan=True)


# --- baseline --------------------------------------------------------------
def test_baseline_beats_chance_but_not_by_much(prepared):
    from predictops.ml.baseline import ThresholdBaseline
    from sklearn.metrics import roc_auc_score

    df, _ = prepared
    d = df[(df.is_downtime == 0) & (df.sensor_dropout == 0)]
    d = d.sort_values(["machine_id", "timestamp"])
    s = ThresholdBaseline().score(d)
    auc = roc_auc_score(d["label"].to_numpy(), s)
    assert auc > 0.6, f"baseline is not a real predictor (AUC {auc:.3f})"
    assert auc < 0.95, f"baseline is suspiciously strong (AUC {auc:.3f})"
