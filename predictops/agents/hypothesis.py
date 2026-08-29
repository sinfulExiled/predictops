"""Agents — the hypothesis advocates, and the adjudicator between them.

The model says "87%". That is a score, not an explanation, and on this plant
the same score is produced by two very different situations:

* a bearing genuinely degrading, and
* a machine being worked harder on a hot afternoon with a noisy transducer.

So two agents argue the case from the **same evidence base**, each with a
different prior and a different set of things it goes looking for:

    DegradationAdvocate   "this machine is developing a fault"
    ConfoundAdvocate      "this is operations, weather, or instrumentation"

They may not disagree about the facts -- both build evidence through
`agents.evidence`, and the verifier re-derives every item from raw telemetry.
They disagree about what the facts *mean*, and each must state what would
change its mind.

`Adjudicator` then decides on computed scores, not rhetoric. This is what
makes a lower investigation threshold safe: the model can afford to flag more
candidates, because a case only becomes an alert if the degradation argument
actually survives the confound argument.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.schemas import FAILURE_MODES
from ..ml.diagnosis import (
    canonical_channel,
    expected_signature,
    observed_directions,
)
from .base import Agent, AgentContext
from .evidence import EvidenceBuilder

# A confound argument this much stronger than the degradation argument
# overturns the model's flag.
OVERTURN_MARGIN = 0.15
# ...but only if the degradation case is itself marginal. Measured on the
# scenario suite, an unconstrained overturn rule rejected a genuine failure one
# hour out (degradation 0.56, confound 0.87) because the duty happened to rise
# at the same time. A coincidental benign explanation must not be allowed to
# veto a strong case -- it may only break a weak one.
DEGRADATION_FLOOR = 0.55
# Below this, neither side has made a case worth acting on.
WEAK_CASE = 0.30

SCHEMA = {
    "type": "object",
    "properties": {
        "argument": {"type": "string"},
        "would_change_my_mind": {"type": "string"},
    },
    "required": ["argument", "would_change_my_mind"],
    "additionalProperties": False,
}


@dataclass
class Case:
    """One side's argument, with the numbers behind it."""

    position: str
    score: float
    evidence: list = field(default_factory=list)
    factors: dict = field(default_factory=dict)
    conclusion: str = ""
    would_change_my_mind: str = ""
    failure_type: str | None = None
    ranked_types: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------
class DegradationAdvocate(Agent):
    """Argues that a real fault is developing, and which one."""

    name = "degradation_advocate"
    brief = ("Argue that the machine is developing a fault, name the mode, "
             "and state what would refute it.")
    system_prompt = (
        "You are a reliability engineer arguing that a machine is developing a "
        "fault. You are given computed evidence and scores. Make the strongest "
        "honest case, using only those numbers, and state plainly what "
        "evidence would change your mind. Do not invent measurements."
    )

    def tools(self) -> list[str]:
        return ["evidence.channel_movements", "evidence.monotonicity",
                "signature.match", "history.nearest_failures",
                "model.attribution"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        builder: EvidenceBuilder = kwargs["builder"]
        prediction: dict = kwargs["prediction"]
        context: dict = kwargs["context"]
        neighbours: list = kwargs.get("neighbours", [])

        observed = observed_directions(builder.items)
        load_pct = builder.measure("load", "pct_change")
        load_flat = abs(load_pct) < 8.0

        # --- what a developing fault looks like ---------------------------
        # 1. the movement persists rather than coming back down
        persistence = {}
        for ch in ("vibration", "temp_excess", "current"):
            if ch in builder.window.columns:
                persistence[ch] = round(builder.measure(ch, "monotonicity"), 3)
        steady_climb = max(persistence.values()) if persistence else 0.0
        if steady_climb >= 0.66:
            worst = max(persistence, key=persistence.get)
            builder.add(
                f"{worst.replace('_', ' ').capitalize()} moved in one direction "
                f"across {persistence[worst]:.0%} of the hourly steps -- a "
                "sustained trend, not an excursion",
                worst, "monotonicity", persistence[worst], "fraction", "up")

        # 2. it is happening while the duty is flat
        if load_flat and any(observed.get(c) == "up"
                             for c in ("vibration", "current", "temperature")):
            builder.add(
                f"Load held flat ({load_pct:+.1f}%) while those channels rose, "
                "so the rise is not explained by the duty",
                "load", "pct_change", load_pct, "%", "flat")

        # 3. it resembles this machine's own past failures
        votes: dict[str, float] = {}
        for nb in neighbours:
            votes[nb["failure_type"]] = votes.get(nb["failure_type"], 0.0) + \
                nb["similarity"]
        total = sum(votes.values()) or 1.0
        history_vote = {k: v / total for k, v in votes.items()}

        # 4. the fitted classifier's opinion
        clf = {}
        if prediction.get("failure_type"):
            clf[prediction["failure_type"]] = float(
                prediction.get("failure_type_confidence", 0.0))
            for alt in prediction.get("failure_type_alternatives", []):
                clf[alt["failure_type"]] = float(alt["probability"])

        # --- rank the modes ------------------------------------------------
        ranked = []
        for ftype in FAILURE_MODES:
            sig = expected_signature(ftype)
            if not sig:
                continue
            matched = [c for c, d in sig.items() if observed.get(c) == d]
            sig_score = len(matched) / max(len(sig), 1)
            score = (0.50 * clf.get(ftype, 0.0) + 0.30 * sig_score
                     + 0.20 * history_vote.get(ftype, 0.0))
            supporting = [
                e["id"] for e in builder.items
                if canonical_channel(e["channel"]) in sig
                and observed.get(canonical_channel(e["channel"]))
                == sig[canonical_channel(e["channel"])]]
            if not supporting and sig_score == 0 and score < 0.15:
                continue
            ranked.append({
                "failure_type": ftype, "score": round(score, 4),
                "classifier_probability": round(clf.get(ftype, 0.0), 4),
                "signature_match": round(sig_score, 3),
                "matched_channels": matched,
                "expected_signature": sig,
                "historical_vote": round(history_vote.get(ftype, 0.0), 4),
                "evidence_ids": supporting,
            })
        ranked.sort(key=lambda h: -h["score"])
        top = ranked[0] if ranked else None

        # --- the strength of the overall position --------------------------
        recurring_bonus = 0.10 if (
            top and context.get("recurring_mode") == top["failure_type"]) else 0.0
        factors = {
            "model_probability": round(
                float(prediction.get("failure_probability", 0.0)), 4),
            "best_mode_score": round(top["score"], 4) if top else 0.0,
            "trend_persistence": round(steady_climb, 3),
            "load_is_flat": bool(load_flat),
            "machine_has_failed_this_way_before": bool(recurring_bonus),
        }
        score = float(np.clip(
            0.45 * factors["model_probability"]
            + 0.25 * factors["best_mode_score"]
            + 0.20 * steady_climb
            + (0.10 if load_flat else 0.0)
            + recurring_bonus, 0.0, 1.0))

        refute = ("A load or ambient increase of the same shape and timing as "
                  "the channel movements, or a single-sample spike rather than "
                  "a sustained trend, would undercut this.")
        conclusion = (
            f"{top['failure_type'].replace('_', ' ')} is developing"
            if top else "a fault is developing but the mode is unclear")

        case = Case(
            position="degradation", score=round(score, 4),
            evidence=[e["id"] for e in builder.items],
            factors=factors, conclusion=conclusion,
            would_change_my_mind=refute,
            failure_type=top["failure_type"] if top else None,
            ranked_types=ranked)
        return {
            "_action": f"Argued degradation at {score:.2f}.",
            "_reason": conclusion,
            **case.to_dict(),
        }

    def narrate(self, ctx: AgentContext, findings: dict):
        prompt = (
            f"Position: a fault is developing. Score {findings['score']}.\n"
            f"Factors: {findings['factors']}\n"
            f"Ranked modes: {findings['ranked_types'][:3]}\n\n"
            "Write the argument in two sentences and state what would change "
            "your mind.")
        fallback = {"argument": findings["conclusion"],
                    "would_change_my_mind": findings["would_change_my_mind"]}
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA,
                                      fallback)
        findings["argument"] = res.data.get("argument", fallback["argument"])
        findings["would_change_my_mind"] = res.data.get(
            "would_change_my_mind", fallback["would_change_my_mind"])
        return findings, res


# --------------------------------------------------------------------------
class ConfoundAdvocate(Agent):
    """Argues the reading is innocent: duty, weather, instrumentation, run-in."""

    name = "confound_advocate"
    brief = ("Argue that the reading has a benign explanation -- production "
             "load, ambient heat, a sensor glitch, or post-service run-in.")
    system_prompt = (
        "You are a sceptical operations engineer. Your job is to find the "
        "innocent explanation for an alarm before a crew is sent out. You are "
        "given computed evidence. Make the strongest honest case that nothing "
        "is wrong, using only those numbers, and state what would change your "
        "mind. Do not invent measurements."
    )

    def tools(self) -> list[str]:
        return ["evidence.channel_movements", "evidence.peak_ratio",
                "evidence.monotonicity", "context.dossier",
                "telemetry.load_profile"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        builder: EvidenceBuilder = kwargs["builder"]
        context: dict = kwargs["context"]
        prediction: dict = kwargs["prediction"]

        explanations: list[dict] = []
        score_parts: dict[str, float] = {}

        # --- 1. the machine is simply working harder ----------------------
        load_pct = builder.measure("load", "pct_change")
        if load_pct >= 8.0:
            builder.add(
                f"Load rose {load_pct:.0f}% over the same window, which raises "
                "vibration, current and temperature on a healthy machine",
                "load", "pct_change", load_pct, "%", "up")
            strength = float(np.clip(load_pct / 40.0, 0.0, 1.0))
            score_parts["production_load"] = strength
            explanations.append({
                "explanation": "production load increase",
                "strength": round(strength, 3),
                "detail": f"duty up {load_pct:.0f}% across the window"})

        # --- 2. it is hot outside ------------------------------------------
        ambient_delta = builder.measure("ambient_temp", "abs_change")
        if ambient_delta >= 3.0:
            builder.add(
                f"Ambient temperature rose {ambient_delta:.1f} deg C over the "
                "same window",
                "ambient_temp", "abs_change", ambient_delta, "deg C", "up")
            temp_only = all(
                canonical_channel(e["channel"]) in ("temperature", "load")
                for e in builder.items if e["direction"] == "up")
            strength = float(np.clip(ambient_delta / 12.0, 0.0, 1.0)) * (
                1.0 if temp_only else 0.45)
            score_parts["ambient"] = strength
            explanations.append({
                "explanation": "ambient temperature rise",
                "strength": round(strength, 3),
                "detail": (f"ambient +{ambient_delta:.1f} C"
                           + ("; only thermal channels moved" if temp_only
                              else "; non-thermal channels also moved, so this "
                                   "cannot be the whole story"))})

        # --- 3. the transducer is lying ------------------------------------
        spiky = {}
        for ch in ("vibration", "temperature", "current", "pressure"):
            if ch in builder.window.columns:
                spiky[ch] = builder.measure(ch, "peak_ratio")
        worst_ch = max(spiky, key=spiky.get) if spiky else None
        if worst_ch and spiky[worst_ch] >= 1.8:
            mono = builder.measure(worst_ch, "monotonicity")
            if mono < 0.5:
                builder.add(
                    f"{worst_ch.capitalize()} peaks at {spiky[worst_ch]:.1f}x "
                    f"its median but moves consistently in only {mono:.0%} of "
                    "hourly steps -- an isolated spike, not a trend",
                    worst_ch, "peak_ratio", spiky[worst_ch], "ratio", "up")
                strength = float(np.clip((spiky[worst_ch] - 1.5), 0.0, 1.0))
                score_parts["sensor_glitch"] = strength
                explanations.append({
                    "explanation": "transducer spike",
                    "strength": round(strength, 3),
                    "detail": (f"{worst_ch} peak/median {spiky[worst_ch]:.1f}x, "
                               f"monotonicity only {mono:.0%}")})

        # --- 4. it was just serviced ---------------------------------------
        if context.get("in_run_in_period"):
            hrs = context.get("hours_since_service")
            strength = 0.55
            score_parts["run_in"] = strength
            explanations.append({
                "explanation": "post-service run-in",
                "strength": round(strength, 3),
                "detail": f"serviced {hrs:.1f} h ago; elevated readings expected"})

        # --- 5. instrumentation is unreliable on this unit ------------------
        drops = context.get("operating_regime", {}).get("dropouts_in_last_24h", 0)
        if drops > 6:
            strength = float(np.clip(drops / 30.0, 0.0, 0.5))
            score_parts["unreliable_instrumentation"] = strength
            explanations.append({
                "explanation": "unreliable instrumentation",
                "strength": round(strength, 3),
                "detail": f"{drops} dropouts in the last 24 h"})

        score = float(np.clip(max(score_parts.values()) if score_parts else 0.0,
                              0.0, 1.0))
        # A very high model score is itself evidence against the benign reading.
        model_p = float(prediction.get("failure_probability", 0.0))
        score = float(np.clip(score * (1.0 - 0.45 * model_p), 0.0, 1.0))

        explanations.sort(key=lambda e: -e["strength"])
        conclusion = (
            f"{explanations[0]['explanation']} explains this reading"
            if explanations else
            "no benign explanation is available -- nothing in the operating "
            "context accounts for the movement")

        case = Case(
            position="confound", score=round(score, 4),
            evidence=[e["id"] for e in builder.items],
            factors={"components": {k: round(v, 3)
                                    for k, v in score_parts.items()},
                     "model_probability": round(model_p, 4)},
            conclusion=conclusion,
            would_change_my_mind=(
                "A sustained multi-channel trend continuing after the load and "
                "ambient returned to normal would defeat this."))
        return {
            "_action": f"Argued a benign explanation at {score:.2f}.",
            "_reason": conclusion,
            "alternative_explanations": explanations,
            **case.to_dict(),
        }

    def narrate(self, ctx: AgentContext, findings: dict):
        prompt = (
            f"Position: nothing is wrong. Score {findings['score']}.\n"
            f"Candidate explanations: {findings['alternative_explanations']}\n\n"
            "Write the argument in two sentences and state what would change "
            "your mind.")
        fallback = {"argument": findings["conclusion"],
                    "would_change_my_mind": findings["would_change_my_mind"]}
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA,
                                      fallback)
        findings["argument"] = res.data.get("argument", fallback["argument"])
        findings["would_change_my_mind"] = res.data.get(
            "would_change_my_mind", fallback["would_change_my_mind"])
        return findings, res


# --------------------------------------------------------------------------
class Adjudicator(Agent):
    """Decides between the two cases, on their computed scores."""

    name = "adjudicator"
    brief = ("Weigh the degradation case against the benign case and decide "
             "whether this is an alert, on the numbers.")

    def tools(self) -> list[str]:
        return ["cases.compare", "thresholds.apply"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        degradation: dict = kwargs["degradation"]
        confound: dict = kwargs["confound"]
        prediction: dict = kwargs["prediction"]

        d = float(degradation.get("score", 0.0))
        c = float(confound.get("score", 0.0))
        margin = d - c
        model_alert = bool(prediction.get("alert", False))
        flagged = bool(prediction.get("investigate", model_alert))

        if not flagged:
            decision, why = "no_alert", (
                "the model did not flag this machine for investigation")
        elif max(d, c) < WEAK_CASE:
            decision, why = "insufficient_evidence", (
                f"neither case reaches the {WEAK_CASE:.2f} floor "
                f"(degradation {d:.2f}, benign {c:.2f})")
        elif margin <= -OVERTURN_MARGIN and d < DEGRADATION_FLOOR:
            decision, why = "overturned", (
                f"the benign explanation is stronger by {abs(margin):.2f} "
                f"(benign {c:.2f} vs degradation {d:.2f}), and the degradation "
                f"case is below the {DEGRADATION_FLOOR:.2f} floor, so the flag "
                "does not stand")
        elif margin <= -OVERTURN_MARGIN:
            decision, why = "contested", (
                f"the benign case is stronger ({c:.2f} vs {d:.2f}) but the "
                f"degradation case still clears the {DEGRADATION_FLOOR:.2f} "
                "floor, so this is not dismissed -- it is checked")
        elif margin >= OVERTURN_MARGIN:
            decision, why = "alert", (
                f"the degradation case survives by {margin:.2f} "
                f"(degradation {d:.2f} vs benign {c:.2f})")
        else:
            decision, why = "contested", (
                f"the two cases are within {OVERTURN_MARGIN:.2f} of each other "
                f"(degradation {d:.2f}, benign {c:.2f}) -- too close to call "
                "without a physical check")

        alert = decision in ("alert", "contested")
        # A contested case is worth looking at, but not worth a repair.
        act = decision == "alert"

        return {
            "_action": f"Adjudicated: {decision}.",
            "_reason": why,
            "_verification": (
                f"decided on computed scores; margin {margin:+.3f}"),
            "decision": decision,
            "alert": alert,
            "recommend_physical_work": act,
            "degradation_score": round(d, 4),
            "confound_score": round(c, 4),
            "margin": round(margin, 4),
            "model_said_alert": model_alert,
            "model_flagged_for_investigation": flagged,
            "changed_the_model_verdict": bool(alert != model_alert),
            "rationale": why,
            "failure_type": (degradation.get("failure_type")
                             if alert else None),
            "leading_benign_explanation": (
                (confound.get("alternative_explanations") or [{}])[0]
                .get("explanation") if c > 0 else None),
        }
