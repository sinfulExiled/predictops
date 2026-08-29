"""Phase 1 -- the synthetic plant must be deterministic, labelled correctly
and free of temporal leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictops.config import GeneratorConfig, HORIZON_HOURS, SENSOR_COLUMNS
from predictops.data.generator import PlantGenerator
from predictops.data.schemas import FAILURE_MODES

SMALL = GeneratorConfig(n_machines=8, days=6, seed=42)


@pytest.fixture(scope="module")
def ds():
    return PlantGenerator(SMALL).generate()


# --- reproducibility -------------------------------------------------------
def test_same_seed_is_bit_identical():
    a = PlantGenerator(SMALL).generate().telemetry
    b = PlantGenerator(SMALL).generate().telemetry
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data():
    other = GeneratorConfig(n_machines=8, days=6, seed=43)
    a = PlantGenerator(SMALL).generate().telemetry
    b = PlantGenerator(other).generate().telemetry
    assert not np.allclose(
        a["vibration"].fillna(0).to_numpy(),
        b["vibration"].fillna(0).to_numpy(),
    )


# --- shape and schema ------------------------------------------------------
def test_schema_and_shape(ds):
    steps = SMALL.days * 24 * (60 // SMALL.resolution_minutes)
    assert len(ds.telemetry) == steps * SMALL.n_machines
    for col in SENSOR_COLUMNS + ["machine_id", "timestamp", "label"]:
        assert col in ds.telemetry.columns, col
    assert ds.telemetry["machine_id"].nunique() == SMALL.n_machines
    assert ds.machines["machine_id"].nunique() == SMALL.n_machines


def test_timestamps_are_on_a_regular_grid(ds):
    one = ds.telemetry[ds.telemetry.machine_id == ds.telemetry.machine_id.iloc[0]]
    gaps = one["timestamp"].diff().dropna().unique()
    assert len(gaps) == 1
    assert gaps[0] == pd.Timedelta(minutes=SMALL.resolution_minutes)


# --- labels ----------------------------------------------------------------
def test_label_is_binary_and_never_null(ds):
    assert ds.telemetry["label"].isna().sum() == 0
    assert set(ds.telemetry["label"].unique()) <= {0, 1}


def test_positive_label_implies_failure_inside_horizon(ds):
    """Every positive row must have a real failure in (t, t + horizon]."""
    fails = ds.failures.copy()
    fails["failure_time"] = pd.to_datetime(fails["failure_time"])
    by_machine = {m: g["failure_time"].to_numpy()
                  for m, g in fails.groupby("machine_id")}
    horizon = np.timedelta64(HORIZON_HOURS * 3600, "s")

    pos = ds.telemetry[ds.telemetry.label == 1]
    assert len(pos) > 0
    for mid, grp in pos.groupby("machine_id"):
        ft = by_machine[mid]
        ts = grp["timestamp"].to_numpy().astype("datetime64[s]")
        ok = np.zeros(len(ts), dtype=bool)
        for f in ft.astype("datetime64[s]"):
            ok |= (ts < f) & (ts >= f - horizon)
        assert ok.all(), f"{mid}: {(~ok).sum()} positives with no failure ahead"


def test_negative_label_has_no_failure_inside_horizon(ds):
    fails = ds.failures.copy()
    fails["failure_time"] = pd.to_datetime(fails["failure_time"])
    horizon = np.timedelta64(HORIZON_HOURS * 3600, "s")
    neg = ds.telemetry[(ds.telemetry.label == 0) & (ds.telemetry.is_downtime == 0)]
    for mid, grp in neg.groupby("machine_id"):
        ft = fails.loc[fails.machine_id == mid, "failure_time"]
        if ft.empty:
            continue
        ts = grp["timestamp"].to_numpy().astype("datetime64[s]")
        bad = np.zeros(len(ts), dtype=bool)
        for f in ft.to_numpy().astype("datetime64[s]"):
            bad |= (ts < f) & (ts >= f - horizon)
        assert not bad.any(), f"{mid}: {bad.sum()} negatives inside a failure window"


def test_downtime_rows_are_not_positive(ds):
    down = ds.telemetry[ds.telemetry.is_downtime == 1]
    assert len(down) > 0
    assert down["label"].sum() == 0


def test_class_imbalance_is_realistic(ds):
    rate = ds.telemetry["label"].mean()
    assert 0.005 < rate < 0.12, f"positive rate {rate:.4f} outside plausible range"


# --- failure realism -------------------------------------------------------
def test_multiple_failure_modes_are_present(ds):
    kinds = set(ds.failures["failure_type"])
    assert len(kinds) >= 4
    assert kinds <= set(FAILURE_MODES)


def test_degradation_shifts_the_expected_channels(ds):
    """Bearing events must raise vibration relative to the same machine's
    healthy baseline -- otherwise the label is not learnable."""
    tel = ds.telemetry
    bearing = ds.failures[
        (ds.failures.failure_type == "bearing_degradation")
        & (~ds.failures.is_atypical)
    ]
    if bearing.empty:
        pytest.skip("no typical bearing events in this seed")
    ratios = []
    for _, ev in bearing.iterrows():
        m = tel[(tel.machine_id == ev.machine_id) & (tel.is_downtime == 0)]
        start = pd.Timestamp(ev.degradation_start)
        fail = pd.Timestamp(ev.failure_time)
        late = m[(m.timestamp >= start + (fail - start) * 0.7)
                 & (m.timestamp < fail)]["vibration"].mean()
        healthy = m[m.degradation_active == 0]["vibration"].mean()
        if np.isfinite(late) and np.isfinite(healthy) and healthy > 0:
            ratios.append(late / healthy)
    assert np.median(ratios) > 1.3, f"bearing signal too weak: {np.median(ratios):.2f}x"


def test_task_is_not_trivially_threshold_separable(ds):
    """Guardrail: if one raw channel solved it, the ML story is worthless."""
    from sklearn.metrics import f1_score

    d = ds.telemetry[ds.telemetry.is_downtime == 0].dropna(subset=["vibration"])
    y = d["label"].to_numpy()
    z = d.groupby("machine_id")["vibration"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)).to_numpy()
    best = max(f1_score(y, (z > t).astype(int), zero_division=0)
               for t in np.arange(0.5, 4.01, 0.25))
    assert best < 0.75, f"single threshold reaches F1={best:.2f}; task too easy"


# --- nuisance processes ----------------------------------------------------
def test_missing_data_exists_but_is_bounded(ds):
    rate = ds.telemetry["vibration"].isna().mean()
    assert 0.001 < rate < 0.10, f"missing rate {rate:.4f}"


def test_confounders_are_present(ds):
    assert ds.telemetry["load_surge"].sum() > 0
    assert ds.telemetry["heatwave"].sum() > 0


def test_operating_hours_reset_at_maintenance(ds):
    mnt = ds.maintenance.copy()
    mnt["timestamp"] = pd.to_datetime(mnt["timestamp"])
    if mnt.empty:
        pytest.skip("no maintenance events")
    row = mnt.iloc[0]
    m = ds.telemetry[ds.telemetry.machine_id == row.machine_id]
    at = m[m.timestamp == row.timestamp]["operating_hours"]
    if at.empty:
        pytest.skip("maintenance outside sampled window")
    assert at.iloc[0] < 1.0


def test_sensors_are_physically_plausible(ds):
    """Ranges hold on every reported reading (dropouts are NaN, not junk)."""
    up = ds.telemetry[ds.telemetry.is_downtime == 0]
    assert up["vibration"].min() >= 0
    assert up["current"].min() >= 0
    assert up["temperature"].dropna().between(-30, 200).all()
    assert up["voltage"].dropna().between(300, 480).all()
    assert up["load"].dropna().between(0.0, 1.3).all()
    assert up["rpm"].dropna().between(0, 6000).all()
