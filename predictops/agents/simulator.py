"""Agent 6 -- Simulation.

Asks "what would happen if we did this?" for every proposed action, plus a
do-nothing control arm, and reports the risk the trained model assigns to each
counterfactual future.

The control arm matters: without it, "risk falls to 0.39" is unreadable,
because some of that fall could come from the rollout itself rather than from
the action.  Every number here is reported against the no-action arm and is
labelled `simulated`.
"""
from __future__ import annotations

import pandas as pd

from ..config import LOOKBACK_STEPS, STEPS_PER_HOUR
from ..ml.bundle import ModelBundle
from ..simulation.interventions import get
from ..simulation.machine_environment import rollout, score_rollout
from .base import Agent, AgentContext

# How far ahead to roll the machine when comparing actions.
DEFAULT_HORIZON_H = 3.0
# Extra history kept so the causal rolling features have a warm-up.
WARMUP_STEPS = LOOKBACK_STEPS * 2


class SimulationAgent(Agent):
    name = "simulator"
    brief = ("Roll the machine forward under each proposed action and a "
             "do-nothing control, and score every counterfactual.")

    def tools(self) -> list[str]:
        return ["machine_environment.rollout", "features.recompute",
                "bundle.score"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        bundle: ModelBundle = kwargs["bundle"]
        machine_id: str = kwargs["machine_id"]
        timestamp = kwargs["timestamp"]
        plan: list[dict] = kwargs["plan"]
        baseline_prob: float = float(kwargs.get("baseline_probability", 0.0))
        horizon_h: float = float(kwargs.get("horizon_hours", DEFAULT_HORIZON_H))

        df = ctx.data.df
        g = df[df.machine_id == machine_id].sort_values("timestamp")
        end = pd.Timestamp(timestamp)
        hist = g[g.timestamp <= end].tail(WARMUP_STEPS)
        if len(hist) < LOOKBACK_STEPS:
            raise ValueError(f"{machine_id} has too little history at {end}")

        # --- control arm: do nothing --------------------------------------
        control = score_rollout(rollout(hist, horizon_h, None), bundle, machine_id)
        control_prob = control["failure_probability_simulated"]

        # --- one arm per proposed action -----------------------------------
        arms = []
        for step in plan:
            iv = get(step["intervention_id"])
            if iv.is_diagnostic:
                arms.append({
                    "intervention_id": iv.id, "title": iv.title,
                    "is_simulated": True, "simulated": False,
                    "reason_not_simulated": (
                        "diagnostic action -- gathers information, changes no "
                        "telemetry, so there is nothing to simulate"),
                    "failure_probability_simulated": None,
                    "delta_vs_no_action": None,
                })
                continue

            res = score_rollout(rollout(hist, horizon_h, iv), bundle, machine_id)
            p = res["failure_probability_simulated"]
            arms.append({
                "intervention_id": iv.id,
                "title": iv.title,
                "simulated": True,
                "is_simulated": True,
                "failure_probability_simulated": p,
                "delta_vs_no_action": round(p - control_prob, 4),
                "delta_vs_now": round(p - baseline_prob, 4),
                "relative_reduction_pct": (
                    round((control_prob - p) / control_prob * 100, 1)
                    if control_prob > 1e-6 else 0.0),
                "projected_channels": res["channels"],
                "cost_usd": iv.cost_usd,
                "downtime_hours": iv.downtime_hours,
            })

        simulated = [a for a in arms if a.get("simulated")]
        best = min(simulated, key=lambda a: a["failure_probability_simulated"]) \
            if simulated else None

        # value for money, since a shutdown always "wins" on risk alone
        for a in simulated:
            reduction = control_prob - a["failure_probability_simulated"]
            a["risk_reduction_per_1k_usd"] = (
                round(reduction / max(a["cost_usd"], 1.0) * 1000, 4))
        efficient = (max(simulated, key=lambda a: a["risk_reduction_per_1k_usd"])
                     if simulated else None)

        improved = bool(best and best["delta_vs_no_action"] < -0.01)
        return {
            "_action": (f"Simulated {len(simulated)} action(s) plus a control "
                        f"arm over {horizon_h:.0f} h."),
            "_reason": (
                f"no action -> {control_prob:.3f}; best action "
                f"({best['intervention_id']}) -> "
                f"{best['failure_probability_simulated']:.3f}"
                if best else "no action could be simulated"),
            "_verification": (
                "All figures are model scores on synthetic telemetry and are "
                "labelled simulated; the control arm uses the identical "
                "rollout so the delta isolates the intervention."),
            "machine_id": machine_id,
            "timestamp": str(timestamp),
            "horizon_hours": horizon_h,
            "probability_now": round(baseline_prob, 4),
            "no_action": {
                "failure_probability_simulated": control_prob,
                "channels": control["channels"],
                "is_simulated": True,
            },
            "arms": arms,
            "best_by_risk": best["intervention_id"] if best else None,
            "best_by_value": efficient["intervention_id"] if efficient else None,
            "simulation_shows_improvement": improved,
            "caveat": (
                "Simulated futures use trend persistence for the underlying "
                "degradation and a first-order effect model for each action. "
                "Use them to rank actions, not as a forecast."),
        }
