"""Central configuration for PredictOps.

Everything that affects reproducibility lives here or in an explicit CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
EXPERIMENT_DIR = ROOT / "artifacts" / "experiments"
REPORT_DIR = ROOT / "artifacts" / "reports"
TRAJECTORY_DIR = ROOT / "artifacts" / "trajectories"

for _d in (DATA_DIR, MODEL_DIR, EXPERIMENT_DIR, REPORT_DIR, TRAJECTORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Prediction task ------------------------------------------------------
# "Will this machine fail within the next HORIZON_HOURS, given only telemetry
# available at or before the current timestamp?"
HORIZON_HOURS = 6
RESOLUTION_MINUTES = 10
STEPS_PER_HOUR = 60 // RESOLUTION_MINUTES          # 6
HORIZON_STEPS = HORIZON_HOURS * STEPS_PER_HOUR     # 36
LOOKBACK_STEPS = 36                                # 6 h of history per window

# Sensor channels present in raw telemetry.
SENSOR_COLUMNS = [
    "temperature",     # deg C
    "vibration",       # mm/s RMS
    "pressure",        # bar
    "rpm",             # rev/min
    "current",         # A
    "voltage",         # V
    "power",           # kW
    "load",            # 0..1 duty fraction
    "humidity",        # % RH  (site-level distractor)
    "ambient_temp",    # deg C
    "operating_hours", # hours since last maintenance
]

FAILURE_TYPES = [
    "bearing_degradation",
    "motor_overheating",
    "pump_cavitation",
    "pressure_loss",
    "electrical_fault",
]

# --- Data splits ----------------------------------------------------------
# Chronological split with a purge gap so no training row's label window
# overlaps the validation/test period (temporal leakage guard).
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.15
# remainder -> test
PURGE_STEPS = HORIZON_STEPS


@dataclass(frozen=True)
class GeneratorConfig:
    """Deterministic knobs for the synthetic plant."""

    n_machines: int = 80
    days: int = 30
    seed: int = 42
    resolution_minutes: int = RESOLUTION_MINUTES
    start: str = "2025-03-01T00:00:00"

    # Expected number of failure events per machine over the whole horizon.
    failures_per_machine: float = 1.5
    # Fraction of failure events that develop very fast (sudden failures).
    sudden_failure_fraction: float = 0.15
    # Fraction of events with damped symptom amplitude (subtle degradation).
    subtle_failure_fraction: float = 0.30
    # Fraction of events that drop their most characteristic symptom entirely
    # (the "unusual pattern" hard case).
    atypical_failure_fraction: float = 0.10

    # Nuisance processes -- these exist to break naive thresholds.
    missing_rate: float = 0.0035          # probability a sample block is dropped
    spike_rate: float = 0.0009           # single-sample sensor glitches
    heatwave_days_fraction: float = 0.12  # site-wide hot spells
    load_surge_per_machine_day: float = 0.45  # sustained high-load episodes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    horizon_hours: int = HORIZON_HOURS
    lookback_steps: int = LOOKBACK_STEPS
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
