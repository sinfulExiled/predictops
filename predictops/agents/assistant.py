"""Agent — Assistant.

Question it owns: **what does the operator want to know, and can we ground it?**

A chat surface is the easiest place in a system like this to undo everything
else. If it answers from a language model's memory of the conversation, then
the careful work upstream — evidence that recomputes, thresholds tuned on
validation, an adjudicator that decides on arithmetic — is invisible and
unverifiable at the one place a human actually reads.

So this assistant cannot originate a fact. It works in three fixed steps:

  1. `route()`    a deterministic intent classifier over the question text
  2. `retrieve()` pulls the answer out of computed artifacts, and records a
                  citation for every number it uses
  3. `narrate()`  optionally asks an LLM to phrase the retrieved facts, and
                  **discards the rephrasing if it contains a number that is not
                  in the facts** -- the same check the verifier applies to the
                  investigation narrative

With no API key the templated answer is served directly, so the assistant is
fully functional with zero credentials.

It may trigger *analysis* (score a machine, run the investigation workflow,
simulate an intervention) because those compute and change nothing physical.
It may never approve, execute or schedule work: `REFUSED_INTENTS` is checked
before anything else, and the approval gate stays in the hands of a human in
the UI.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import REPORT_DIR
from ..simulation.interventions import CATALOGUE
from .base import Agent, AgentContext

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

# Phrasings that ask the assistant to authorise or carry out physical work.
# Checked first, so no amount of retrieval can talk past them.
#
# A refusal filter with holes is worse than none, so the verb and the object
# are matched independently with a bounded gap between them. That catches
# "go ahead with the shutdown" and "please just execute that repair", which a
# pattern expecting the object immediately after the verb both missed.
_ACT_VERBS = (r"approve|authoris|authoriz|sign[- ]off|execute|carry out|"
              r"perform|go ahead|proceed|dispatch|schedule|initiate|trigger|"
              r"actuate|shut ?down|stop|halt")
_ACT_OBJECTS = (r"repair|replacement|replace|shutdown|shut ?down|maintenance|"
                r"work order|work|fix|bearing|machine|pump|motor|compressor|"
                r"conveyor|crew|technician|engineer|intervention|action|plan")

REFUSED_PATTERNS = [
    # a bare authorisation, with or without an object
    r"\b(approve|authoris\w*|authoriz\w*|sign[- ]?off)\b",
    # verb ... object, within a short span
    rf"\b({_ACT_VERBS})\b[^.?!]{{0,30}}\b({_ACT_OBJECTS})\b",
    r"\b(send|dispatch)\s+(a\s+|the\s+)?(crew|technician|engineer)\b",
    r"\bdo it\b",
]

MACHINE_RE = re.compile(r"\b(PUMP|MOTOR|COMPRESSOR|CONVEYOR)[-\s]?(\d{1,3})\b", re.I)


@dataclass
class Citation:
    source: str
    record: str = ""
    field: str = ""
    value: object = None

    def to_dict(self) -> dict:
        return {"source": self.source, "record": self.record,
                "field": self.field, "value": self.value}


@dataclass
class Answer:
    intent: str
    answer: str
    citations: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    grounded: bool = True
    action: str | None = None
    action_result: dict | None = None
    refused: bool = False

    def to_dict(self) -> dict:
        return {"intent": self.intent, "answer": self.answer,
                "citations": [c.to_dict() for c in self.citations],
                "facts": self.facts, "grounded": self.grounded,
                "action": self.action, "action_result": self.action_result,
                "refused": self.refused}


def _machine_in(text: str, known: set[str]) -> str | None:
    m = MACHINE_RE.search(text or "")
    if not m:
        return None
    candidate = f"{m.group(1).upper()}-{int(m.group(2)):03d}"
    return candidate if candidate in known else None


def route(question: str) -> str:
    """Deterministic intent classification. Order is precedence."""
    q = (question or "").lower().strip()
    if not q:
        return "empty"
    for pat in REFUSED_PATTERNS:
        if re.search(pat, q):
            return "refused_action"
    rules = [
        ("ablation", r"\b(ablation|adjudicat\w*|contest|advocate|disagree)\b.*"
                     r"\b(help|improve|worth|earn|work|result)\b|\bablation\b"),
        ("evaluation", r"\b(evaluat\w+|benchmark|how good|accuracy|precision|"
                       r"recall|f1|baseline vs|compare.*baseline|scenario)\b"),
        ("model_choice", r"\b(which model|what model|why xgboost|why not (the )?"
                         r"(tft|lstm)|model select\w*|experiment|changelog|bake.?off)\b"),
        # no trailing \b, so "thresholds" matches as well as "threshold"
        ("thresholds", r"\bthreshold|\btrigger|\bcut ?off|when does it alert"),
        ("investigate", r"\b(investigat\w+|why (is|was)|explain|diagnos\w+|"
                        r"what.?s wrong|evidence)\b"),
        ("simulate", r"\b(simulat\w+|what if|counterfactual|if we (reduce|cut|"
                     r"lower)|would happen)\b"),
        ("interventions", r"\b(intervention|what can we do|options|catalogue|"
                          r"catalog|remediat\w+|fix|action)\b"),
        # fleet before machine_status: "which machines are at risk" is a fleet
        # question even though it contains "risk"
        ("fleet", r"\b(fleet|all machines|which machines|every machine|"
                  r"highest risk|at risk|overview|summary)\b"),
        ("machine_status", r"\b(status|risk|probability|how is|score)\b"),
        ("capabilities", r"\b(what can you|help|how do i|what do you do)\b"),
    ]
    for intent, pat in rules:
        if re.search(pat, q):
            return intent
    if MACHINE_RE.search(q):
        return "machine_status"
    return "ungrounded"


class AssistantAgent(Agent):
    name = "assistant"
    brief = ("Answer operator questions strictly from computed artifacts, "
             "citing every number, and refuse anything it cannot ground.")
    system_prompt = (
        "You are the assistant inside a predictive-maintenance tool. You are "
        "given a set of FACTS retrieved from the system's own records, and a "
        "draft answer. Rewrite the draft so it reads naturally for a "
        "maintenance planner. You may not introduce any number, machine name "
        "or claim that is not in the FACTS. If the draft says the system does "
        "not know something, keep that."
    )

    def tools(self) -> list[str]:
        return ["assistant.route", "fleet.scores", "registry.experiments",
                "reports.evaluation", "reports.ablation", "catalogue.list",
                "engine.run_incident"]

    # -- retrieval -----------------------------------------------------------
    def _fleet(self, engine) -> tuple[str, list, dict]:
        df = engine.fleet_scores()
        live = df.dropna(subset=["failure_probability"])
        high = live[live.status == "high"]
        at = df.attrs.get("timestamp")
        lines = [f"At {at}, {len(high)} of {len(live)} machines are above the "
                 f"alert threshold."]
        for _, r in high.head(5).iterrows():
            lines.append(f"  {r.machine_id} — {r.failure_probability:.1%} "
                         f"(confidence {r.confidence:.0%})")
        if high.empty:
            top = live.head(3)
            lines.append("Highest scores, all below the threshold:")
            for _, r in top.iterrows():
                lines.append(f"  {r.machine_id} — {r.failure_probability:.1%}")
        cites = [Citation("fleet scoring (live)", f"snapshot {at}",
                          "failure_probability", None)]
        facts = {"timestamp": str(at), "n_high": int(len(high)),
                 "n_live": int(len(live)),
                 "top": live.head(5)[["machine_id", "failure_probability"]]
                 .to_dict("records")}
        return "\n".join(lines), cites, facts

    def _machine(self, engine, mid: str) -> tuple[str, list, dict]:
        df = engine.fleet_scores()
        row = df[df.machine_id == mid]
        if row.empty:
            return f"{mid} is not in the fleet.", [], {}
        r = row.iloc[0]
        at = df.attrs.get("timestamp")
        if pd.isna(r.failure_probability):
            return f"{mid} is stopped at {at}, so it is not scored.", [], {}
        svc = engine.service
        text = (f"{mid} is at {r.failure_probability:.1%} failure probability "
                f"as of {at}, against an alert threshold of "
                f"{svc.alert_threshold:.3f} — status {r.status}. "
                f"Confidence {r.confidence:.0%}, meaning that share of "
                f"validation cases scoring at least this high were genuinely "
                f"followed by a failure. Latest readings: vibration "
                f"{r.vibration}, temperature {r.temperature}, load {r.load}.")
        cites = [Citation("fleet scoring (live)", f"{mid} @ {at}",
                          "failure_probability", float(r.failure_probability)),
                 Citation("model bundle", engine.bundle.kind, "threshold",
                          round(svc.alert_threshold, 4))]
        return text, cites, {"machine_id": mid,
                             "failure_probability": float(r.failure_probability),
                             "status": str(r.status)}

    def _model_choice(self, ctx) -> tuple[str, list, dict]:
        exps = ctx.store.all(ctx.store.latest_run_id())
        if not exps:
            return "No experiments have been recorded yet.", [], {}
        scored = [(e, e.metrics.get("val", {}).get("pr_auc")) for e in exps]
        scored = [(e, v) for e, v in scored if isinstance(v, (int, float))]
        best = max(scored, key=lambda ev: ev[1])
        kept = [e for e, _ in scored if e.decision == "kept"]
        lines = [f"{len(exps)} candidates were trained and recorded. Selection "
                 f"is on validation PR-AUC; test metrics are never used to "
                 f"choose."]
        for e, v in scored:
            lines.append(f"  {e.stage}: {e.name} — val PR-AUC {v:.4f}, "
                         f"test F1 {e.metrics['row']['f1']:.4f} ({e.decision})")
        lines.append(f"Highest validation score: {best[0].name} at "
                     f"{best[1]:.4f}.")
        cites = [Citation("artifacts/experiments/experiments.db",
                          f"run {e.run_id} / {e.stage}", "val.pr_auc", round(v, 4))
                 for e, v in scored]
        return "\n".join(lines), cites, {"n_experiments": len(exps),
                                         "best": best[0].name}

    def _thresholds(self, engine) -> tuple[str, list, dict]:
        d = engine.service.describe()
        text = (f"The model service exposes two thresholds. Below "
                f"{d['investigate_threshold']:.3f} a machine is normal and no "
                f"agent runs. Between {d['investigate_threshold']:.3f} and "
                f"{d['alert_threshold']:.3f} it is investigated but not "
                f"alarmed. At or above {d['alert_threshold']:.3f} it is an "
                f"alert. The alert threshold was tuned on the validation split "
                f"and frozen before the test split was scored.")
        cites = [Citation("model bundle", d["kind"], "alert_threshold",
                          d["alert_threshold"]),
                 Citation("model bundle", d["kind"], "investigate_threshold",
                          d["investigate_threshold"])]
        return text, cites, d

    def _evaluation(self) -> tuple[str, list, dict]:
        path = REPORT_DIR / "evaluation.json"
        if not path.exists():
            return ("No evaluation has been run yet — `python evaluate.py` "
                    "produces it."), [], {}
        d = json.loads(path.read_text())
        a, b = d["agent"], d["baseline"]
        text = (
            f"On {d['suite']['n_cases']} fixed scenarios "
            f"({d['suite']['n_positive']} real warning windows, "
            f"{d['suite']['n_negative']} nuisance cases), the agent solution "
            f"scores {a['alert_accuracy']:.1%} alert accuracy against the "
            f"threshold baseline's {b['alert_accuracy']:.1%}. F1 "
            f"{b['f1']:.3f} → {a['f1']:.3f}. Precision "
            f"{b['precision']:.3f} → {a['precision']:.3f}, recall "
            f"{b['recall']:.3f} → {a['recall']:.3f}. False alarms on nuisance "
            f"cases fall from {b['false_alarm_rate_on_nuisance_cases']:.1%} to "
            f"{a['false_alarm_rate_on_nuisance_cases']:.1%}. Recall is the "
            f"known weakness: it misses more than half the warning windows at "
            f"this operating point.")
        cites = [Citation("artifacts/reports/evaluation.json", "agent", k, a[k])
                 for k in ("alert_accuracy", "f1", "precision", "recall")]
        return text, cites, {"agent": a, "baseline": b}

    def _ablation(self) -> tuple[str, list, dict]:
        path = REPORT_DIR / "ablation_adjudication.json"
        if not path.exists():
            return ("No ablation on disk — `python ablate_adjudication.py` "
                    "produces it."), [], {}
        d = json.loads(path.read_text())
        s = d["summary"]
        text = (
            f"It was measured, and the answer is no. Across the threshold "
            f"sweep the adjudicator changed {s['verdicts_changed']} verdicts "
            f"and is worth {s['delta_f1']:+.4f} F1: best model-only "
            f"{s['best_model_only_f1']:.4f} against best adjudicated "
            f"{s['best_adjudicated_f1']:.4f}. The reason is that the model "
            f"already rejects every nuisance case well below the "
            f"investigation trigger, so the confound advocate never gets a "
            f"case to overturn. It is kept as a safety mechanism rather than "
            f"an accuracy one: a contested verdict routes to inspection and "
            f"can never authorise repair.")
        cites = [Citation("artifacts/reports/ablation_adjudication.json",
                          "summary", k, s[k]) for k in
                 ("delta_f1", "verdicts_changed", "best_model_only_f1",
                  "best_adjudicated_f1")]
        return text, cites, s

    def _interventions(self, ftype: str | None) -> tuple[str, list, dict]:
        items = [iv for iv in CATALOGUE.values()
                 if not ftype or not iv.applicable_to
                 or ftype in iv.applicable_to]
        lines = [f"{len(items)} approved actions"
                 + (f" apply to {ftype.replace('_', ' ')}" if ftype else "")
                 + ". Nothing outside this catalogue can be proposed:"]
        for iv in items[:8]:
            lines.append(f"  {iv.id} — {iv.title} (${iv.cost_usd:,.0f}, "
                         f"{iv.risk} risk"
                         + (", needs approval" if iv.requires_approval else "")
                         + ")")
        cites = [Citation("predictops/simulation/interventions.py", iv.id,
                          "cost_usd", iv.cost_usd) for iv in items[:8]]
        return "\n".join(lines), cites, {"n": len(items)}

    # -- the agent contract ---------------------------------------------------
    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        question: str = kwargs.get("question", "")
        engine = kwargs["engine"]
        allow_actions: bool = bool(kwargs.get("allow_actions", True))

        intent = route(question)
        known = set(ctx.data.df["machine_id"].unique())
        mid = _machine_in(question, known)

        if intent == "refused_action":
            ans = Answer(
                intent=intent, refused=True, grounded=True,
                answer=("I can't approve, schedule or carry out physical work. "
                        "This system proposes and simulates; a named human "
                        "approves in the Remediation Simulator, and nothing "
                        "here actuates a machine. I can show you the proposed "
                        "plan and what the simulation says it would buy."))
            return {"_action": "Refused an action request.",
                    "_reason": "outside the assistant's authority", **ans.to_dict()}

        if intent == "empty":
            ans = Answer(intent=intent, answer="Ask me something.",
                         grounded=False)
            return {"_action": "No question asked.", **ans.to_dict()}

        if intent == "capabilities":
            ans = Answer(
                intent=intent, grounded=True,
                answer=("I answer from this system's own records, and I cite "
                        "every number. Try: 'which machines are at risk', "
                        "'why is MOTOR-045 flagged', 'what can we do about "
                        "bearing degradation', 'which model was selected and "
                        "why', 'did the adjudicator help', 'how good is it "
                        "against the baseline', 'what are the thresholds'. I "
                        "can run an investigation for you. I cannot approve or "
                        "carry out work."))
            return {"_action": "Listed capabilities.", **ans.to_dict()}

        action, action_result = None, None
        try:
            if intent in ("investigate", "simulate") and mid and allow_actions:
                report = engine.run_incident(mid, engine.busiest_timestamp(),
                                             save=False)
                action = "run_incident"
                adj, pred = report.adjudication, report.prediction
                ev = report.investigation.get("evidence", [])
                sim = report.simulation
                if intent == "simulate":
                    arms = [a for a in sim.get("arms", []) if a.get("simulated")]
                    best = (min(arms, key=lambda a: a["failure_probability_simulated"])
                            if arms else None)
                    ctl = sim["no_action"]["failure_probability_simulated"]
                    text = (f"Simulated {sim['horizon_hours']:.0f} h ahead for "
                            f"{mid}. Doing nothing gives {ctl:.1%}. "
                            + (f"The best action is {best['intervention_id']} at "
                               f"{best['failure_probability_simulated']:.1%} "
                               f"({best['delta_vs_no_action']:+.1%} against the "
                               f"control arm)." if best else
                               "No proposed action changes telemetry, so there "
                               "is nothing to simulate.")
                            + " These are model scores on synthetic "
                              "counterfactual telemetry, not a forecast.")
                    cites = [Citation("simulation (live)", mid,
                                      "no_action.failure_probability_simulated",
                                      ctl)]
                    facts = {"control": ctl,
                             "best": best["intervention_id"] if best else None}
                else:
                    lines = [f"{mid} scores {pred['failure_probability']:.1%}. "
                             f"The adjudicator's verdict is {adj['decision']} "
                             f"(degradation {adj['degradation_score']:.2f} vs "
                             f"benign {adj['confound_score']:.2f})."]
                    if ev:
                        lines.append("Evidence:")
                        for e in ev[:6]:
                            lines.append(f"  [{e['id']}] {e['claim']}")
                    lines.append(f"Verification: {report.verification['verdict']}"
                                 f" — {report.verification['headline']}")
                    if not adj.get("recommend_physical_work"):
                        lines.append("No physical work is authorised on this "
                                     "case.")
                    text = "\n".join(lines)
                    cites = [Citation("investigation (live)", f"{mid} {e['id']}",
                                      e["metric"], e["value"]) for e in ev[:6]]
                    facts = {"probability": pred["failure_probability"],
                             "decision": adj["decision"]}
                action_result = {"machine_id": mid,
                                 "decision": adj["decision"],
                                 "verification": report.verification["verdict"]}
                ans = Answer(intent=intent, answer=text, citations=cites,
                             facts=facts, action=action,
                             action_result=action_result)
                return {"_action": f"Ran the workflow for {mid}.",
                        "_reason": adj["decision"], **ans.to_dict()}

            if intent in ("investigate", "simulate") and not mid:
                ans = Answer(
                    intent=intent, grounded=False,
                    answer=("Name a machine and I'll run it — for example "
                            "'why is MOTOR-045 flagged'."))
                return {"_action": "Asked which machine.", **ans.to_dict()}

            if intent == "machine_status" and mid:
                text, cites, facts = self._machine(engine, mid)
            elif intent in ("machine_status", "fleet"):
                text, cites, facts = self._fleet(engine)
            elif intent == "model_choice":
                text, cites, facts = self._model_choice(ctx)
            elif intent == "thresholds":
                text, cites, facts = self._thresholds(engine)
            elif intent == "evaluation":
                text, cites, facts = self._evaluation()
            elif intent == "ablation":
                text, cites, facts = self._ablation()
            elif intent == "interventions":
                ft = next((f for f in
                           ("bearing_degradation", "motor_overheating",
                            "pump_cavitation", "pressure_loss",
                            "electrical_fault")
                           if f.replace("_", " ") in question.lower()), None)
                text, cites, facts = self._interventions(ft)
            else:
                ans = Answer(
                    intent="ungrounded", grounded=False,
                    answer=("I don't have a computed answer for that. I can "
                            "only report what this system has actually "
                            "measured — fleet risk, the evidence behind an "
                            "alert, the model bake-off, the evaluation, the "
                            "ablation, and the approved intervention "
                            "catalogue."))
                return {"_action": "Declined: no grounding.",
                        "_reason": "no artifact answers this", **ans.to_dict()}
        except Exception as exc:  # noqa: BLE001
            ans = Answer(intent=intent, grounded=False,
                         answer=f"I could not retrieve that: {exc}")
            return {"_action": "Retrieval failed.", **ans.to_dict()}

        ans = Answer(intent=intent, answer=text, citations=cites, facts=facts)
        return {"_action": f"Answered a '{intent}' question.",
                "_reason": f"{len(cites)} citation(s)", **ans.to_dict()}

    # -- narration, with the same no-invented-numbers guard -------------------
    def narrate(self, ctx: AgentContext, findings: dict):
        if not findings.get("grounded", True) or findings.get("refused"):
            return findings, None
        draft = findings["answer"]
        facts = json.dumps(findings.get("facts", {}), default=str)
        cites = json.dumps([c["value"] for c in findings.get("citations", [])],
                           default=str)
        prompt = (f"FACTS: {facts}\nCITED VALUES: {cites}\n\n"
                  f"DRAFT ANSWER:\n{draft}\n\nRewrite the draft.")
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA,
                                      {"answer": draft})
        candidate = res.data.get("answer", draft)

        # Same guard as verification check C8: a rephrasing may not introduce a
        # number that is not already in the draft or the cited values.
        allowed = set(re.findall(r"\d+\.?\d*", draft + " " + facts + " " + cites))
        introduced = [n for n in re.findall(r"\d+\.?\d*", candidate)
                      if n not in allowed]
        if introduced:
            findings["narration_rejected"] = introduced[:5]
        else:
            findings["answer"] = candidate
        return findings, res
