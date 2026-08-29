"""Agent — Investigation.

Question it owns: **what actually changed?**

Narrowed on purpose. It used to gather the facts *and* rank the failure modes,
which meant the interpretation was baked into whoever collected the evidence —
there was no way for a second reading to compete. Now it establishes the
shared factual record and stops. `agents.hypothesis` argues over it.

Every evidence item carries the exact function, channel and window that
produced it, so `agents.verifier` can re-derive it from raw telemetry and
compare. An item the verifier cannot reproduce is reported as unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import STEPS_PER_HOUR
from .base import Agent, AgentContext
from .evidence import DEFAULT_HOURS, EVIDENCE_HORIZONS, EvidenceBuilder

# Re-exported: the verifier and the advocates import these from here for
# backwards compatibility with the original single-investigator layout.
from .evidence import RECOMPUTE_FNS, abs_change, pct_change  # noqa: F401

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass
class SignatureLibrary:
    """Compact summaries of the windows that preceded past failures."""

    vectors: np.ndarray
    types: list[str]
    machines: list[str]
    channels: list[str]

    @staticmethod
    def _summarise(window: pd.DataFrame, channels: list[str]) -> np.ndarray:
        out = []
        for c in channels:
            s = window[c].to_numpy(dtype=float)
            s = s[np.isfinite(s)]
            if len(s) < 4:
                out.extend([0.0, 0.0, 0.0])
                continue
            z = (s - s.mean()) / (s.std() + 1e-9)
            slope = float(np.polyfit(np.arange(len(z)), z, 1)[0])
            out.extend([slope, float(s.std() / (abs(s.mean()) + 1e-9)),
                        float((s[-STEPS_PER_HOUR:].mean()
                               - s[:STEPS_PER_HOUR].mean())
                              / (abs(s.mean()) + 1e-9))])
        return np.asarray(out, dtype=float)

    @classmethod
    def build(cls, df: pd.DataFrame, failures: pd.DataFrame,
              channels: list[str], lookback: int, split: str = "train"
              ) -> "SignatureLibrary":
        f = failures.copy()
        f["failure_time"] = pd.to_datetime(f["failure_time"])
        vectors, types, machines = [], [], []
        for _, ev in f.iterrows():
            g = df[(df.machine_id == ev.machine_id) & (df["split"] == split)
                   & (df.timestamp < ev.failure_time)]
            if len(g) < lookback:
                continue
            w = g.sort_values("timestamp").iloc[-lookback:]
            vectors.append(cls._summarise(w, channels))
            types.append(str(ev.failure_type))
            machines.append(str(ev.machine_id))
        arr = np.vstack(vectors) if vectors else np.zeros((0, len(channels) * 3))
        return cls(arr, types, machines, channels)

    def nearest(self, window: pd.DataFrame, k: int = 5) -> list[dict]:
        if len(self.vectors) == 0:
            return []
        v = self._summarise(window, self.channels)
        d = np.linalg.norm(self.vectors - v, axis=1)
        order = np.argsort(d)[:k]
        scale = float(np.median(d)) + 1e-9
        return [{"failure_type": self.types[int(i)],
                 "machine_id": self.machines[int(i)],
                 "distance": round(float(d[int(i)]), 4),
                 "similarity": round(float(np.exp(-d[int(i)] / scale)), 4)}
                for i in order]


class InvestigationAgent(Agent):
    name = "investigator"
    brief = ("Establish the factual record: what moved, by how much, in what "
             "operating context, and what it resembles.")
    system_prompt = (
        "You are a reliability engineer summarising what a machine's telemetry "
        "did over the last few hours, for colleagues who will argue about what "
        "it means. State only what the evidence says. Do not diagnose, and do "
        "not state any number that is not in the evidence."
    )

    def tools(self) -> list[str]:
        return ["telemetry.window", "evidence.channel_movements",
                "model.attribution", "signature_library.nearest"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        service = kwargs["service"]
        machine_id: str = kwargs["machine_id"]
        timestamp = kwargs["timestamp"]
        library: SignatureLibrary = kwargs["library"]

        raw = service.raw_window(ctx.data.df, machine_id, timestamp)
        builder = EvidenceBuilder(raw)
        builder.channel_movements(EVIDENCE_HORIZONS)

        context = {
            "load_change_pct_3h": round(
                builder.measure("load", "pct_change"), 2),
            "ambient_change_c_3h": round(
                builder.measure("ambient_temp", "abs_change"), 2),
        }
        context["load_stable"] = bool(abs(context["load_change_pct_3h"]) < 8.0)
        context["ambient_rising"] = bool(context["ambient_change_c_3h"] > 3.0)

        neighbours = library.nearest(raw, k=5)
        attribution = service.attribution(ctx.data.df, machine_id, timestamp)

        moved = [e["channel"] for e in builder.items
                 if e["direction"] in ("up", "down")]
        summary = (f"{len(builder.items)} channel movement(s) recorded"
                   + (f": {', '.join(moved)}" if moved else
                      " -- no channel moved materially"))

        # The advocates extend this same record rather than starting a
        # second, competing one. It travels through the shared workflow state
        # because it is a live object, not JSON.
        ctx.state["evidence_builder"] = builder

        return {
            "_action": f"Recorded the factual state of {machine_id}.",
            "_reason": summary,
            "_verification": (
                f"{len(builder.items)} evidence item(s), each recomputable."),
            "machine_id": machine_id,
            "timestamp": str(timestamp),
            "evidence": builder.items,
            "operating_context": context,
            "model_attribution": attribution,
            "similar_past_failures": neighbours,
            "summary": summary,
        }

    def narrate(self, ctx: AgentContext, findings: dict):
        ev = "\n".join(f"{e['id']}: {e['claim']}" for e in findings["evidence"])
        prompt = (f"Machine {findings['machine_id']} at "
                  f"{findings['timestamp']}.\n\nEvidence:\n{ev}\n\n"
                  f"Operating context: {findings['operating_context']}\n\n"
                  "Summarise what changed, in one or two sentences. Do not "
                  "diagnose.")
        fallback = {"summary": findings["summary"]}
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA,
                                      fallback)
        findings["summary"] = res.data.get("summary", fallback["summary"])
        return findings, res
