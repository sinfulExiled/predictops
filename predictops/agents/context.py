"""Agent — Context.

Question it owns: **what do we already know about this machine?**

Split out of the investigator deliberately. The investigator answers "what
changed in the last few hours"; this answers "what is normal for *this*
machine, and what has recently been done to it". Those are different
questions with different data sources, and conflating them was hiding the
single most useful confound signal in the dataset: a machine that has just
been serviced runs hot and rough for a while, and that is not a fault.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import STEPS_PER_HOUR
from .base import Agent, AgentContext

# Hours after a service during which elevated readings are expected.
RUN_IN_HOURS = 8.0


class ContextAgent(Agent):
    name = "context"
    brief = ("Assemble the machine dossier: what is normal for this unit, "
             "what has been done to it, and what it has failed with before.")

    def tools(self) -> list[str]:
        return ["machines.lookup", "maintenance.history", "failures.history",
                "telemetry.baseline_stats"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        machine_id: str = kwargs["machine_id"]
        timestamp = pd.Timestamp(kwargs["timestamp"])
        df = ctx.data.df

        g = df[df.machine_id == machine_id].sort_values("timestamp")
        past = g[g.timestamp <= timestamp]
        if past.empty:
            raise ValueError(f"no history for {machine_id} at {timestamp}")
        now = past.iloc[-1]

        meta = ctx.data.machines[ctx.data.machines.machine_id == machine_id]
        machine_type = (str(meta["machine_type"].iloc[0]) if len(meta)
                        else machine_id.split("-")[0])

        # --- what is normal for THIS machine -------------------------------
        # Healthy reference: this machine's own history, excluding downtime.
        # Deliberately uses only data before `timestamp`.
        healthy = past[(past.is_downtime == 0) & (past.sensor_dropout == 0)]
        baseline = {}
        for ch in ("vibration", "temperature", "current", "pressure",
                   "temp_excess"):
            if ch not in healthy.columns or healthy.empty:
                continue
            s = healthy[ch].dropna()
            if len(s) < 24:
                continue
            baseline[ch] = {
                "median": round(float(s.median()), 3),
                "p95": round(float(s.quantile(0.95)), 3),
                "current": round(float(now[ch]), 3),
                "z_vs_own_history": round(
                    float((now[ch] - s.mean()) / (s.std() + 1e-9)), 2),
            }

        # --- maintenance ----------------------------------------------------
        mnt = ctx.data.maintenance[
            ctx.data.maintenance.machine_id == machine_id].copy()
        recent_service, hours_since_service = None, None
        if len(mnt):
            mnt["timestamp"] = pd.to_datetime(mnt["timestamp"])
            done = mnt[mnt.timestamp <= timestamp].sort_values("timestamp")
            if len(done):
                last = done.iloc[-1]
                hours_since_service = round(
                    (timestamp - last.timestamp).total_seconds() / 3600.0, 2)
                recent_service = {
                    "kind": str(last.kind),
                    "at": str(last.timestamp),
                    "hours_ago": hours_since_service,
                    "failure_type": str(last.get("failure_type", "") or ""),
                }
        in_run_in = bool(hours_since_service is not None
                         and hours_since_service <= RUN_IN_HOURS)

        # --- failure history -------------------------------------------------
        fails = ctx.data.failures[
            ctx.data.failures.machine_id == machine_id].copy()
        prior = []
        if len(fails):
            fails["failure_time"] = pd.to_datetime(fails["failure_time"])
            for _, ev in fails[fails.failure_time <= timestamp].iterrows():
                prior.append({
                    "failure_type": str(ev.failure_type),
                    "at": str(ev.failure_time),
                    "days_ago": round(
                        (timestamp - ev.failure_time).total_seconds() / 86400, 2),
                })
        type_counts: dict[str, int] = {}
        for p in prior:
            type_counts[p["failure_type"]] = type_counts.get(
                p["failure_type"], 0) + 1
        recurring = sorted(type_counts.items(), key=lambda kv: -kv[1])[:1]

        # --- operating regime -------------------------------------------------
        day = past.tail(24 * STEPS_PER_HOUR)
        regime = {
            "load_median_24h": round(float(day["load"].median()), 3),
            "load_now": round(float(now["load"]), 3),
            "ambient_median_24h": round(float(day["ambient_temp"].median()), 2),
            "ambient_now": round(float(now["ambient_temp"]), 2),
            "operating_hours_since_service": round(
                float(now["operating_hours"]), 1),
            "dropouts_in_last_24h": int(day["sensor_dropout"].sum()),
        }

        notes = []
        if in_run_in:
            notes.append(
                f"serviced {hours_since_service:.1f} h ago -- elevated "
                f"vibration and temperature are expected during the first "
                f"{RUN_IN_HOURS:.0f} h of run-in")
        if recurring:
            notes.append(
                f"has failed with {recurring[0][0].replace('_', ' ')} "
                f"{recurring[0][1]}x before -- a repeat is more likely than "
                "the fleet base rate")
        if regime["dropouts_in_last_24h"] > 6:
            notes.append(
                f"{regime['dropouts_in_last_24h']} sensor dropouts in 24 h -- "
                "instrumentation on this unit is unreliable")
        if not notes:
            notes.append("nothing unusual in this machine's recent record")

        return {
            "_action": f"Assembled the dossier for {machine_id}.",
            "_reason": (
                f"{machine_type}, {regime['operating_hours_since_service']:.0f} h "
                f"since service, {len(prior)} prior failure(s)"),
            "machine_id": machine_id,
            "machine_type": machine_type,
            "own_baseline": baseline,
            "recent_service": recent_service,
            "hours_since_service": hours_since_service,
            "in_run_in_period": in_run_in,
            "prior_failures": prior,
            "prior_failure_types": type_counts,
            "recurring_mode": recurring[0][0] if recurring else None,
            "operating_regime": regime,
            "notes": notes,
        }
