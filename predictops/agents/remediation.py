"""Agent 5 -- Remediation.

Selects actions from the approved catalogue.  Three hard constraints, enforced
in code rather than asked for in a prompt:

1.  Only ids present in `simulation.interventions.CATALOGUE` may be proposed;
    anything else raises.
2.  An action gated on a precondition that is not met is proposed only in the
    order that satisfies it -- e.g. `replace_bearing` requires a confirmed
    inspection, so an unconfirmed case gets `inspect_bearing` first.
3.  High-risk and downtime-causing actions are always flagged
    `requires_approval`, and the plan carries an explicit approval gate.

The agent proposes.  A human approves.  Nothing here actuates anything.
"""
from __future__ import annotations

from ..simulation.interventions import CATALOGUE, Intervention, applicable, get
from .base import Agent, AgentContext

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "ordering_rationale": {"type": "string"},
    },
    "required": ["summary", "ordering_rationale"],
    "additionalProperties": False,
}

# Above this probability, the situation warrants proposing a stop.
URGENT_PROBABILITY = 0.85
# Below this, evidence is too thin to justify a physical action.
WEAK_EVIDENCE = 0.45


class RemediationAgent(Agent):
    name = "remediation"
    brief = ("Propose an ordered plan of approved interventions, matched to "
             "the diagnosis and its confidence.")
    system_prompt = (
        "You are a maintenance planner. You are given a diagnosis and a list of "
        "already-selected approved actions in order. Summarise the plan for a "
        "supervisor in plain language. Do not add, remove or reorder actions, "
        "and do not invent any action that is not in the list."
    )

    def tools(self) -> list[str]:
        return ["catalogue.applicable", "catalogue.get",
                "preconditions.evaluate"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        prediction: dict = kwargs["prediction"]
        investigation: dict = kwargs["investigation"]
        adjudication: dict = kwargs.get("adjudication") or {}
        context: dict = kwargs.get("context") or {}

        ftype = (investigation.get("likely_failure_type")
                 or prediction.get("failure_type"))
        machine_type = (context.get("machine_type")
                        or str(prediction.get("machine_id", "")).split("-")[0])
        prob = float(prediction.get("failure_probability", 0.0))
        eta = prediction.get("prediction_window_hours", {}).get("eta_hours")
        margin = investigation.get("hypothesis_margin", "close_call")
        top = (investigation.get("ranked_hypotheses") or [{}])[0]
        support = float(top.get("score", 0.0))

        permitted = applicable(ftype, machine_type)
        permitted_ids = [iv.id for iv in permitted]

        plan: list[dict] = []
        notes: list[str] = []

        def add(iv: Intervention, why: str, order: int) -> None:
            plan.append({
                "order": order,
                "intervention_id": iv.id,
                "title": iv.title,
                "detail": iv.detail,
                "why": why,
                "risk": iv.risk,
                "cost_usd": iv.cost_usd,
                "downtime_hours": iv.downtime_hours,
                "requires_approval": iv.requires_approval,
                "is_diagnostic": iv.is_diagnostic,
                "preconditions": list(iv.preconditions),
                "expected_effect": iv.effects or "no telemetry change (diagnostic)",
            })

        # --- the case did not survive challenge: propose nothing physical --
        decision = adjudication.get("decision")
        if decision == "overturned":
            add(get("increase_monitoring"),
                (f"the benign explanation "
                 f"({adjudication.get('leading_benign_explanation') or 'operating context'}) "
                 f"outweighed the degradation case by "
                 f"{abs(adjudication.get('margin', 0)):.2f}; sending a crew "
                 "would be a false callout"), 1)
            notes.append(
                "Adjudication overturned the model's flag -- monitoring only. "
                + adjudication.get("rationale", ""))
            return self._finish(plan, notes, ftype, prob, permitted_ids,
                                "monitor_only", adjudication)
        if decision == "insufficient_evidence":
            add(get("increase_monitoring"),
                "neither the degradation nor the benign case cleared the "
                "evidence floor", 1)
            notes.append("Adjudication found no case strong enough to act on.")
            return self._finish(plan, notes, ftype, prob, permitted_ids,
                                "monitor_only", adjudication)
        if decision == "contested":
            diag = next((iv for iv in permitted if iv.is_diagnostic
                         and iv.id != "increase_monitoring"), None)
            add(get("increase_monitoring"), (
                "the two readings of the evidence are within "
                f"{abs(adjudication.get('margin', 0)):.2f} of each other"), 1)
            if diag is not None:
                add(diag, ("a physical check is the cheapest way to break the "
                           "tie, and it is reversible"), 2)
            notes.append(
                "Contested case: inspect before committing to any repair. "
                + adjudication.get("rationale", ""))
            return self._finish(plan, notes, ftype, prob, permitted_ids,
                                "diagnose", adjudication)

        # --- evidence too thin: do not touch the machine -------------------
        if support < WEAK_EVIDENCE or margin == "no_supported_hypothesis":
            add(get("increase_monitoring"),
                (f"hypothesis support is only {support:.2f} and the margin over "
                 "the next candidate is small; a physical action is not yet "
                 "justified"), 1)
            notes.append("Evidence is inconclusive -- monitoring only.")
            return self._finish(plan, notes, ftype, prob, permitted_ids,
                                "monitor_only", adjudication)

        # --- 1. buy time, reversibly ---------------------------------------
        holding = next((iv for iv in permitted
                        if iv.id in ("reduce_speed_15", "reduce_load_70")), None)
        if holding is not None:
            add(holding,
                (f"reversible and reduces the driver of {ftype.replace('_', ' ')} "
                 f"while the inspection is arranged"), len(plan) + 1)

        # --- 2. confirm before committing ----------------------------------
        diagnostic = next((iv for iv in permitted if iv.is_diagnostic
                           and iv.id != "increase_monitoring"), None)
        if diagnostic is not None:
            deadline = (f"within {max(int((eta or 4) - 1), 1)} h"
                        if eta else "within 4 h")
            add(diagnostic,
                f"confirms the diagnosis before any irreversible work; {deadline}",
                len(plan) + 1)

        # --- 3. the definitive repair, gated on that confirmation ----------
        repair = next((iv for iv in permitted
                       if not iv.is_diagnostic and iv.risk in ("medium", "high")
                       and iv.id != "controlled_shutdown"), None)
        if repair is not None:
            gate = ("after inspection confirms the fault"
                    if diagnostic is not None else "once approved")
            add(repair, f"the definitive fix, to be carried out {gate}",
                len(plan) + 1)
            if repair.preconditions:
                notes.append(
                    f"{repair.title} is gated on: "
                    + "; ".join(repair.preconditions))

        # --- 4. stop the machine, only when genuinely urgent ---------------
        if prob >= URGENT_PROBABILITY and (eta is None or eta <= 3.0):
            add(get("controlled_shutdown"),
                (f"probability {prob:.2f} with an estimated {eta:.1f} h to "
                 "failure leaves little margin; a controlled stop avoids "
                 "secondary damage"), len(plan) + 1)
            notes.append("Shutdown proposed because the window is short -- "
                         "requires operations sign-off.")
        else:
            notes.append(
                f"Shutdown not proposed: probability {prob:.2f} and estimated "
                f"{eta if eta is not None else 'unknown'} h to failure leave "
                "room for the reversible steps above.")

        if not plan:
            add(get("increase_monitoring"),
                "no catalogued action applies to this diagnosis", 1)
            notes.append("No catalogued action matched this failure type.")

        mode = "act" if any(not p["is_diagnostic"] for p in plan) else "diagnose"
        return self._finish(plan, notes, ftype, prob, permitted_ids, mode,
                            adjudication)

    @staticmethod
    def _finish(plan, notes, ftype, prob, permitted_ids, mode,
                adjudication=None) -> dict:
        for p in plan:
            if p["intervention_id"] not in CATALOGUE:
                raise ValueError(
                    f"proposed action '{p['intervention_id']}' is not in the "
                    "approved catalogue")
        needs_approval = [p["intervention_id"] for p in plan
                          if p["requires_approval"]]
        return {
            "_action": f"Proposed {len(plan)} approved action(s), mode={mode}.",
            "_reason": (f"diagnosis {ftype}, probability {prob:.2f}; "
                        f"{len(needs_approval)} action(s) need human approval"),
            "_verification": ("Every action id was checked against the approved "
                              "catalogue."),
            "diagnosis": ftype,
            "mode": mode,
            "gated_by_adjudication": (adjudication or {}).get("decision"),
            "plan": plan,
            "notes": notes,
            "permitted_actions": permitted_ids,
            "approval_gate": {
                "required": bool(needs_approval),
                "actions": needs_approval,
                "status": "awaiting_human_approval" if needs_approval else "none",
                "statement": ("PredictOps proposes; it does not act. No action "
                              "is executed without a named human approver."),
            },
            "estimated_cost_usd": round(sum(p["cost_usd"] for p in plan), 2),
            "estimated_downtime_hours": round(
                sum(p["downtime_hours"] for p in plan), 2),
        }

    def narrate(self, ctx: AgentContext, findings: dict):
        steps = "\n".join(f"{p['order']}. {p['title']} -- {p['why']}"
                          for p in findings["plan"])
        prompt = (f"Diagnosis: {findings['diagnosis']}\n"
                  f"Selected actions, in order:\n{steps}\n"
                  f"Notes: {findings['notes']}\n\n"
                  "Summarise this plan for a shift supervisor and explain why "
                  "it is ordered this way.")
        fallback = {
            "summary": "; ".join(p["title"] for p in findings["plan"]),
            "ordering_rationale": (
                "Reversible holding actions first, confirmation before "
                "irreversible work, shutdown only if the window is short."),
        }
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA,
                                      fallback)
        findings["summary"] = res.data.get("summary", fallback["summary"])
        findings["ordering_rationale"] = res.data.get(
            "ordering_rationale", fallback["ordering_rationale"])
        return findings, res
