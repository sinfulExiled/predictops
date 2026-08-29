"""The controlled catalogue of maintenance actions.

The remediation agent may only propose actions from this list.  It cannot
invent one, and it cannot alter an action's parameters -- it selects an id.
That is the guard against an LLM proposing something physically dangerous or
operationally impossible.

`effects` describes what the action does to the raw sensor channels, and is
used by `simulation.machine_environment` to build the counterfactual.  Effects
are expressed as either a multiplier (`mul`) or an offset (`add`) applied to
the channel, ramped in over `settle_hours`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Intervention:
    id: str
    title: str
    detail: str
    applicable_to: tuple[str, ...]          # failure types, () = any
    machine_types: tuple[str, ...] = ()     # () = any
    effects: dict = field(default_factory=dict)
    settle_hours: float = 1.0
    downtime_hours: float = 0.0
    cost_usd: float = 0.0
    risk: str = "low"                       # low | medium | high
    requires_approval: bool = True
    is_diagnostic: bool = False             # gathers information, changes nothing
    preconditions: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


CATALOGUE: dict[str, Intervention] = {
    # --- load and speed management (reversible, low risk) -----------------
    "reduce_load_70": Intervention(
        id="reduce_load_70", title="Reduce load to 70%",
        detail=("Throttle the duty setpoint to 70% of current. Buys time on a "
                "developing fault without stopping production."),
        applicable_to=("bearing_degradation", "motor_overheating",
                       "electrical_fault", "pump_cavitation"),
        effects={"load": {"mul": 0.70}},
        settle_hours=1.0, cost_usd=180.0, risk="low", requires_approval=True,
        preconditions=("production schedule allows reduced throughput",),
    ),
    "reduce_load_50": Intervention(
        id="reduce_load_50", title="Reduce load to 50%",
        detail=("Halve the duty setpoint. Used when a fault is advanced and a "
                "shutdown slot is not yet available."),
        applicable_to=("bearing_degradation", "motor_overheating",
                       "electrical_fault", "pump_cavitation"),
        effects={"load": {"mul": 0.50}},
        settle_hours=1.0, cost_usd=420.0, risk="low", requires_approval=True,
        preconditions=("production schedule allows reduced throughput",),
    ),
    "reduce_speed_15": Intervention(
        id="reduce_speed_15", title="Reduce shaft speed 15%",
        detail=("Drop the drive frequency. Vibration falls faster than load "
                "with speed, so this helps a bearing more than throttling."),
        applicable_to=("bearing_degradation", "pump_cavitation"),
        effects={"rpm": {"mul": 0.85}, "vibration": {"mul": 0.72},
                 "current": {"mul": 0.88}},
        settle_hours=0.5, cost_usd=240.0, risk="low", requires_approval=True,
    ),

    # --- cooling and fluid path -------------------------------------------
    "boost_cooling": Intervention(
        id="boost_cooling", title="Increase cooling / clear airway",
        detail=("Raise coolant flow and clear the intake screen. Addresses "
                "heat rise that is not caused by mechanical friction."),
        applicable_to=("motor_overheating", "electrical_fault"),
        effects={"temperature": {"add": -8.0}},
        settle_hours=1.5, cost_usd=120.0, risk="low", requires_approval=False,
    ),
    "restore_suction": Intervention(
        id="restore_suction", title="Restore suction / re-prime",
        detail=("Check the suction line, vent trapped air and re-prime. "
                "Standard first response to cavitation."),
        applicable_to=("pump_cavitation", "pressure_loss"),
        machine_types=("PUMP", "COMPRESSOR"),
        effects={"pressure": {"mul": 1.28}, "vibration": {"mul": 0.78}},
        settle_hours=1.0, downtime_hours=0.5, cost_usd=350.0, risk="medium",
        requires_approval=True,
    ),
    "seal_pressure_circuit": Intervention(
        id="seal_pressure_circuit", title="Locate and seal pressure loss",
        detail="Pressure-test the circuit and replace the failing seal.",
        applicable_to=("pressure_loss",),
        machine_types=("PUMP", "COMPRESSOR"),
        effects={"pressure": {"mul": 1.45}},
        settle_hours=1.0, downtime_hours=2.5, cost_usd=1400.0, risk="medium",
        requires_approval=True,
    ),

    # --- electrical --------------------------------------------------------
    "inspect_electrical": Intervention(
        id="inspect_electrical", title="Inspect supply and terminations",
        detail=("Thermographic check of terminations and supply balance. "
                "Diagnostic: confirms or rules out an electrical fault."),
        applicable_to=("electrical_fault",),
        effects={}, is_diagnostic=True,
        settle_hours=0.0, cost_usd=200.0, risk="low", requires_approval=False,
    ),
    "correct_supply": Intervention(
        id="correct_supply", title="Correct supply / re-terminate",
        detail="Re-torque terminations and correct the phase imbalance.",
        applicable_to=("electrical_fault",),
        effects={"voltage": {"mul": 1.05}, "current": {"mul": 0.86},
                 "temperature": {"add": -6.0}},
        settle_hours=1.0, downtime_hours=1.5, cost_usd=900.0, risk="medium",
        requires_approval=True,
    ),

    # --- mechanical repair --------------------------------------------------
    "inspect_bearing": Intervention(
        id="inspect_bearing", title="Inspect bearing within 4 hours",
        detail=("Hands-on inspection: temperature gun, listening stick, "
                "grease condition. Diagnostic: confirms before committing to "
                "a replacement."),
        applicable_to=("bearing_degradation",),
        effects={}, is_diagnostic=True,
        settle_hours=0.0, cost_usd=150.0, risk="low", requires_approval=False,
    ),
    "relubricate_bearing": Intervention(
        id="relubricate_bearing", title="Re-lubricate bearing",
        detail=("Re-grease to specification. Helps early-stage wear; does "
                "nothing once the raceway is damaged."),
        applicable_to=("bearing_degradation",),
        effects={"vibration": {"mul": 0.82}, "temperature": {"add": -4.0}},
        settle_hours=1.0, cost_usd=90.0, risk="low", requires_approval=False,
    ),
    "replace_bearing": Intervention(
        id="replace_bearing", title="Replace bearing",
        detail=("Planned shutdown and bearing change. The definitive fix once "
                "inspection confirms degradation."),
        applicable_to=("bearing_degradation",),
        effects={"vibration": {"mul": 0.35}, "temperature": {"add": -12.0},
                 "current": {"mul": 0.88}},
        settle_hours=1.0, downtime_hours=4.0, cost_usd=2600.0, risk="high",
        requires_approval=True,
        preconditions=("inspection has confirmed bearing degradation",
                       "replacement bearing in stock"),
    ),

    # --- always available ---------------------------------------------------
    "increase_monitoring": Intervention(
        id="increase_monitoring", title="Raise sampling rate and re-assess in 1 h",
        detail=("No physical change. Used when evidence is thin and acting "
                "now would be premature."),
        applicable_to=(), effects={}, is_diagnostic=True,
        settle_hours=0.0, cost_usd=15.0, risk="low", requires_approval=False,
    ),
    "controlled_shutdown": Intervention(
        id="controlled_shutdown", title="Controlled shutdown",
        detail=("Stop the machine in a controlled way. Reserved for imminent "
                "failure where continued running risks secondary damage."),
        applicable_to=(),
        effects={"load": {"mul": 0.0}},
        settle_hours=0.25, downtime_hours=6.0, cost_usd=5200.0, risk="high",
        requires_approval=True,
        preconditions=("failure probability is high and the window is short",
                       "operations has agreed a stop slot"),
    ),
}


def applicable(failure_type: str | None, machine_type: str | None
               ) -> list[Intervention]:
    """Actions permitted for this diagnosis on this machine class."""
    out = []
    for iv in CATALOGUE.values():
        if iv.applicable_to and failure_type not in iv.applicable_to:
            continue
        if iv.machine_types and machine_type not in iv.machine_types:
            continue
        out.append(iv)
    return out


def get(intervention_id: str) -> Intervention:
    if intervention_id not in CATALOGUE:
        raise KeyError(
            f"'{intervention_id}' is not in the approved intervention "
            f"catalogue; permitted ids are {sorted(CATALOGUE)}")
    return CATALOGUE[intervention_id]
