#!/usr/bin/env python
"""Export representative agent trajectories in a readable form.

    python export_trajectories.py                 # latest incident + research run
    python export_trajectories.py --run-id main

Writes artifacts/reports/AGENT_TRAJECTORIES.md — one section per agent, showing
its instructions, what it did, which tools answered, the feedback that shaped
its next step, retries, and the human checkpoints it produced.
"""
from __future__ import annotations

import argparse
import json
import textwrap

from predictops.agents.assistant import AssistantAgent
from predictops.agents.context import ContextAgent
from predictops.agents.data_scientist import DataScientistAgent
from predictops.agents.hypothesis import (
    Adjudicator,
    ConfoundAdvocate,
    DegradationAdvocate,
)
from predictops.agents.investigator import InvestigationAgent
from predictops.agents.model_researcher import ModelResearchAgent
from predictops.agents.predictor import PredictionAgent
from predictops.agents.remediation import RemediationAgent
from predictops.agents.simulator import SimulationAgent
from predictops.agents.verifier import VerificationAgent
from predictops.config import REPORT_DIR
from predictops.experiments.registry import ExperimentStore

# Workflow order, so the export reads as the run actually happened.
AGENTS = {
    a.name: a for a in (
        DataScientistAgent(), ModelResearchAgent(), PredictionAgent(),
        ContextAgent(), InvestigationAgent(), DegradationAdvocate(),
        ConfoundAdvocate(), Adjudicator(), RemediationAgent(),
        SimulationAgent(), VerificationAgent(), AssistantAgent(),
    )
}

# What is worth showing from each agent's output, rather than dumping all of it.
HIGHLIGHTS = {
    "data_scientist": ["leakage_audit", "class_balance", "recommended_plan"],
    "model_researcher": ["selection", "candidates_run"],
    "predictor": ["failure_probability", "alert", "threshold",
                  "prediction_window_hours", "failure_type", "confidence",
                  "confidence_basis"],
    "context": ["machine_type", "in_run_in_period", "hours_since_service",
                "prior_failure_types", "recurring_mode", "notes"],
    "investigator": ["evidence", "operating_context", "summary"],
    "degradation_advocate": ["score", "conclusion", "factors",
                             "would_change_my_mind", "ranked_types"],
    "confound_advocate": ["score", "conclusion",
                          "alternative_explanations", "would_change_my_mind"],
    "adjudicator": ["decision", "degradation_score", "confound_score",
                    "margin", "rationale", "changed_the_model_verdict",
                    "recommend_physical_work"],
    "remediation": ["diagnosis", "mode", "plan", "approval_gate"],
    "simulator": ["no_action", "arms", "best_by_risk", "best_by_value",
                  "simulation_shows_improvement"],
    "verifier": ["verdict", "headline", "checks", "safe_to_act",
                 "action_guidance"],
    "assistant": ["intent", "answer", "citations", "grounded", "refused",
                  "action"],
}


def _fence(obj, limit: int = 2600) -> str:
    blob = json.dumps(obj, indent=2, default=str)
    if len(blob) > limit:
        blob = blob[:limit] + "\n  ... (truncated; full record in the registry)"
    return f"```json\n{blob}\n```"


def _wrap(text: str, width: int = 78) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def render(store: ExperimentStore, run_ids: list[str]) -> str:
    out: list[str] = []
    out.append("# Agent Trajectories")
    out.append("")
    out.append(
        "One section per agent: the instructions it runs under, a real "
        "execution, the tools that answered it, and the feedback that shaped "
        "what happened next. Records come straight from "
        "`artifacts/experiments/experiments.db`.")
    out.append("")
    out.append(
        "Every agent follows the same contract — `investigate()` computes "
        "findings with deterministic tools, then an optional `narrate()` pass "
        "phrases them. No number in any output below passed through a language "
        "model.")
    out.append("")

    steps: list[dict] = []
    for rid in run_ids:
        steps.extend(store.trajectory(rid))

    seen: set[str] = set()
    for name, agent in AGENTS.items():
        mine = [s for s in steps if s["agent"] == name]
        if not mine:
            continue
        step = mine[-1]
        seen.add(name)

        out.append("---")
        out.append("")
        out.append(f"## {name}")
        out.append("")
        out.append(f"**Brief.** {_wrap(agent.brief)}")
        out.append("")
        if getattr(agent, "system_prompt", ""):
            out.append("**Instructions given to the language model** "
                       "(narrative pass only):")
            out.append("")
            out.append("> " + _wrap(agent.system_prompt).replace("\n", "\n> "))
            out.append("")
        out.append(f"**Tools available.** "
                   + ", ".join(f"`{t}`" for t in agent.tools()))
        out.append("")
        out.append(f"**Run** `{step['run_id']}` · step {step['step']} · "
                   f"{step['duration_s']:.2f}s · retries {step['retry_count']}")
        out.append("")
        out.append(f"**Input.** `{step['input_summary'] or '(none)'}`")
        out.append("")
        out.append(f"**Action.** {_wrap(step['action'])}")
        out.append("")
        if step.get("reason"):
            out.append(f"**Reason.** {_wrap(step['reason'])}")
            out.append("")
        if step.get("verification"):
            out.append(f"**Self-reported check.** {_wrap(step['verification'])}")
            out.append("")

        output = step.get("output") or {}
        keys = [k for k in HIGHLIGHTS.get(name, []) if k in output]
        shown = {k: output[k] for k in keys} if keys else output
        out.append("**Result.**")
        out.append("")
        out.append(_fence(shown))
        out.append("")

        if name == "assistant":
            out.append("**Constraint.** The assistant cannot originate a fact. "
                       "It routes the question deterministically, retrieves "
                       "from computed artifacts with a citation per number, and "
                       "discards any LLM rephrasing that introduces a number "
                       "not in those facts. Requests to approve, schedule or "
                       "carry out physical work are refused before retrieval "
                       "runs.")
            out.append("")
        if name == "adjudicator":
            out.append("**Human checkpoint.** An `overturned` or "
                       "`insufficient_evidence` decision stops the workflow "
                       "proposing any physical work; a `contested` decision "
                       "downgrades the plan to inspection only. The "
                       "remediation agent is gated on this value.")
            out.append("")
        if name in ("degradation_advocate", "confound_advocate"):
            out.append("**Constraint.** This agent extends the shared evidence "
                       "record built by the investigator; it cannot introduce a "
                       "measurement of its own, and the verifier re-derives "
                       "every item it cites from raw telemetry.")
            out.append("")
        if name == "verifier":
            out.append("**Human checkpoint.** A `FAIL` verdict clears the "
                       "`safe_to_act` flag, and any action marked "
                       "`requires_approval` stays behind the approval gate "
                       "regardless of the model's confidence.")
            out.append("")
        if name == "remediation":
            gate = output.get("approval_gate", {})
            if gate.get("required"):
                out.append(f"**Human checkpoint.** {gate.get('statement', '')} "
                           f"Awaiting approval for: "
                           f"{', '.join(gate.get('actions', []))}.")
                out.append("")

        if len(mine) > 1:
            out.append(f"_{len(mine)} executions of this agent are recorded "
                       f"across the exported runs._")
            out.append("")

    missing = set(AGENTS) - seen
    if missing:
        out.append("---")
        out.append("")
        out.append(f"_No recorded execution for: {', '.join(sorted(missing))}. "
                   "Run `python run_experiments.py` and `python run_pipeline.py` "
                   "to populate them._")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", action="append", default=None,
                   help="may be given more than once")
    p.add_argument("--out", default=str(REPORT_DIR / "AGENT_TRAJECTORIES.md"))
    args = p.parse_args()

    store = ExperimentStore()
    run_ids = args.run_id
    if not run_ids:
        runs = store.runs()
        research = [r for r in runs if r.startswith(("main", "research"))][:1]
        incidents = [r for r in runs if r.startswith("incident")][:1]
        # include a run that exercised the assistant, so every agent appears
        assistant = [r for r in runs
                     if any(s["agent"] == "assistant"
                            for s in store.trajectory(r))][:1]
        run_ids = research + incidents + assistant or runs[:2]
    if not run_ids:
        raise SystemExit("no runs recorded; run run_experiments.py first")

    text = render(store, run_ids)
    from pathlib import Path
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"exported trajectories from {run_ids} -> {path}")


if __name__ == "__main__":
    main()
