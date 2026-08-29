"""Machine archetypes, failure-mode physics and record schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MachineArchetype:
    """Nominal operating point for a class of machine."""

    kind: str
    rpm_nominal: float
    temp_base: float          # deg C above ambient at full load
    vib_base: float           # mm/s RMS at nominal load
    pressure_base: float      # bar (0 -> machine has no pressure loop)
    current_base: float       # A at full load
    voltage_nominal: float
    power_factor: float
    thermal_alpha: float      # EMA coefficient -> thermal inertia
    failure_modes: tuple[str, ...]


ARCHETYPES: dict[str, MachineArchetype] = {
    "PUMP": MachineArchetype(
        kind="PUMP", rpm_nominal=1450, temp_base=26.0, vib_base=1.6,
        pressure_base=4.2, current_base=18.0, voltage_nominal=400.0,
        power_factor=0.86, thermal_alpha=0.07,
        failure_modes=("bearing_degradation", "pump_cavitation",
                       "pressure_loss", "electrical_fault"),
    ),
    "MOTOR": MachineArchetype(
        kind="MOTOR", rpm_nominal=2950, temp_base=34.0, vib_base=1.1,
        pressure_base=0.0, current_base=26.0, voltage_nominal=400.0,
        power_factor=0.89, thermal_alpha=0.10,
        failure_modes=("bearing_degradation", "motor_overheating",
                       "electrical_fault"),
    ),
    "COMPRESSOR": MachineArchetype(
        kind="COMPRESSOR", rpm_nominal=1780, temp_base=38.0, vib_base=2.1,
        pressure_base=7.5, current_base=32.0, voltage_nominal=400.0,
        power_factor=0.84, thermal_alpha=0.06,
        failure_modes=("bearing_degradation", "pressure_loss",
                       "motor_overheating"),
    ),
    "CONVEYOR": MachineArchetype(
        kind="CONVEYOR", rpm_nominal=980, temp_base=18.0, vib_base=1.4,
        pressure_base=0.0, current_base=12.0, voltage_nominal=400.0,
        power_factor=0.82, thermal_alpha=0.09,
        failure_modes=("bearing_degradation", "motor_overheating",
                       "electrical_fault"),
    ),
}


@dataclass(frozen=True)
class FailureMode:
    """How a latent degradation severity `s` in [0, 1] shows up in sensors.

    Each coefficient is the effect at s == 1.0 before per-event damping.
    Multiplicative terms are fractions (0.15 -> +15%); additive terms are in
    the sensor's own unit.  `signature` names the channel a domain expert
    would look at first -- the atypical-event generator suppresses it.
    """

    name: str
    vib_mult: float = 0.0
    temp_add: float = 0.0
    current_mult: float = 0.0
    pressure_mult: float = 0.0
    voltage_mult: float = 0.0
    rpm_jitter_mult: float = 0.0
    signature: str = ""
    typical_hours: tuple[float, float] = (10.0, 60.0)


FAILURE_MODES: dict[str, FailureMode] = {
    # Friction rises: vibration dominates, heat and current follow, load steady.
    "bearing_degradation": FailureMode(
        name="bearing_degradation", vib_mult=3.10, temp_add=15.0,
        current_mult=0.16, rpm_jitter_mult=0.6, signature="vibration",
        typical_hours=(14.0, 72.0),
    ),
    # Cooling path fails: heat dominates, current follows, speed gets unstable.
    "motor_overheating": FailureMode(
        name="motor_overheating", vib_mult=0.55, temp_add=30.0,
        current_mult=0.13, rpm_jitter_mult=2.4, signature="temperature",
        typical_hours=(6.0, 30.0),
    ),
    # Cavitation: suction pressure collapses, hydraulic noise spikes.
    "pump_cavitation": FailureMode(
        name="pump_cavitation", vib_mult=2.20, temp_add=6.0,
        current_mult=0.10, pressure_mult=-0.30, rpm_jitter_mult=1.1,
        signature="pressure", typical_hours=(4.0, 24.0),
    ),
    # Leak / seal loss: pressure bleeds away, machine works *less* hard.
    "pressure_loss": FailureMode(
        name="pressure_loss", vib_mult=0.45, temp_add=3.0,
        current_mult=-0.09, pressure_mult=-0.48, rpm_jitter_mult=0.3,
        signature="pressure", typical_hours=(8.0, 40.0),
    ),
    # Winding / supply fault: current climbs, bus sags, heat follows.
    "electrical_fault": FailureMode(
        name="electrical_fault", vib_mult=0.60, temp_add=13.0,
        current_mult=0.32, voltage_mult=-0.06, rpm_jitter_mult=1.6,
        signature="current", typical_hours=(3.0, 20.0),
    ),
}


@dataclass
class FailureEvent:
    machine_id: str
    failure_type: str
    degradation_start: Any     # pd.Timestamp
    failure_time: Any
    repair_time: Any
    severity_scale: float      # 1.0 = textbook, <1 = subtle
    degradation_hours: float
    is_sudden: bool
    is_atypical: bool
    suppressed_channel: str = ""
    deg_start_step: int = 0
    fail_step: int = 0
    repair_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        for k in ("degradation_start", "failure_time", "repair_time"):
            d[k] = str(d[k])
        return d


@dataclass
class MaintenanceEvent:
    machine_id: str
    timestamp: Any
    kind: str                 # "preventive" | "corrective"
    failure_type: str = ""
    step: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["timestamp"] = str(d["timestamp"])
        return d


# Columns produced by the generator on top of the raw sensors.
LABEL_COLUMNS = [
    "label",                  # 1 if a failure starts within the horizon
    "time_to_failure_h",      # hours until next failure (NaN if none pending)
    "horizon_failure_type",   # failure type inside horizon, else ""
    "degradation_active",     # ground-truth latent flag (diagnostics only)
    "severity",               # ground-truth latent severity (diagnostics only)
    "is_downtime",            # machine stopped -> excluded from train/eval
]

# Ground-truth-only columns. Any model or agent touching these is leaking.
LEAKY_COLUMNS = {
    "label", "time_to_failure_h", "horizon_failure_type",
    "degradation_active", "severity",
}
