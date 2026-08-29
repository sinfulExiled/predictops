"""Agent 7 -- Verification.

The adversarial pass.  It assumes the upstream agents are wrong and tries to
show it, using the raw telemetry rather than their summaries.

Checks, each returning pass / fail / warn with the numbers behind it:

 C1  every evidence item recomputes from raw telemetry to its stated value
 C2  the evidence directions match the signature of the diagnosed failure mode
 C3  the prediction clears the model's own decision threshold
 C4  confidence is backed by the measured reliability curve, not asserted
 C5  a benign alternative explanation (load surge, hot weather) is considered
     and explicitly accepted or rejected on the numbers
 C6  every proposed action exists in the approved catalogue
 C7  the simulation actually beat its own do-nothing control
 C8  the narrative contains no number that is absent from the evidence
 C9  irreversible or high-risk actions carry an approval gate
C10  the competing hypotheses were resolved on evidence, not on rhetoric

Checks are **scoped to the claim actually being made**.  Most of them
interrogate the reasoning behind an alert; on a machine the model is not
alerting about there is no diagnosis, no plan and nothing to simulate, so
those checks report `n/a` rather than failing.  An earlier version did fail
them, which marked 25 of 45 evaluation cases FAIL for the crime of being
healthy.

A failed check is never smoothed over: the run is marked FAIL and the report
says which claim could not be supported.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..ml.bundle import window_for
from ..ml.diagnosis import expected_signature, observed_directions
from .base import Agent, AgentContext
from .investigator import RECOMPUTE_FNS

# Evidence values are stored rounded to this many decimals, so a re-derivation
# can only ever be compared at that precision -- a tighter tolerance would flag
# the rounding itself as a mismatch.
EVIDENCE_DECIMALS = 4
# Confidence below this is not enough to act on a physical intervention.
MIN_ACTIONABLE_CONFIDENCE = 0.35


class VerificationAgent(Agent):
    name = "verifier"
    brief = ("Re-derive every claim from raw telemetry and challenge the "
             "diagnosis, the plan and the simulation.")

    def tools(self) -> list[str]:
        return ["telemetry.window", "evidence.recompute",
                "signature.compare", "catalogue.validate",
                "narrative.scan_numbers"]

    @staticmethod
    def _check(cid, name, status, detail, **extra) -> dict:
        return {"id": cid, "check": name, "status": status,
                "detail": detail, **extra}

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        bundle = kwargs["bundle"]
        prediction: dict = kwargs["prediction"]
        investigation: dict = kwargs["investigation"]
        remediation: dict = kwargs["remediation"]
        simulation: dict = kwargs["simulation"]
        adjudication: dict = kwargs.get("adjudication") or {}

        machine_id = prediction["machine_id"]
        timestamp = prediction["timestamp"]
        checks: list[dict] = []

        # Scope. Most of these checks interrogate the reasoning behind an
        # alert. When the model is *not* alerting there is no diagnosis, no
        # repair plan and nothing to simulate, so running them anyway produced
        # a wall of meaningless failures on healthy machines. Those checks are
        # reported "n/a" instead, and only the claim actually being made --
        # "this machine is fine" -- is verified.
        alerting = bool(prediction.get(
            "alert", float(prediction.get("failure_probability", 0.0))
            >= float(prediction.get("threshold", 0.5))))

        # Independent read of the raw window -- not the investigator's copy.
        _, raw = window_for(ctx.data.df, machine_id, timestamp,
                            bundle.channels, None, bundle.lookback)

        # --- C1: recompute every evidence item ----------------------------
        evidence = investigation.get("evidence", [])
        mismatches = []
        for e in evidence:
            spec = e.get("recompute")
            if not spec or spec.get("fn") not in RECOMPUTE_FNS:
                mismatches.append({"id": e.get("id"),
                                   "problem": "no recompute recipe"})
                continue
            fn = RECOMPUTE_FNS[spec["fn"]]
            try:
                got = fn(raw, spec["channel"], spec["hours"])
            except Exception as exc:  # noqa: BLE001
                mismatches.append({"id": e["id"], "problem": str(exc)})
                continue
            claimed = float(e["value"])
            if abs(round(got, EVIDENCE_DECIMALS)
                   - round(claimed, EVIDENCE_DECIMALS)) > 1e-9:
                mismatches.append({"id": e["id"], "claimed": claimed,
                                   "recomputed": round(got, EVIDENCE_DECIMALS)})
        c1_status = ("fail" if mismatches else
                     "pass" if evidence else
                     ("n/a" if not alerting else "warn"))
        checks.append(self._check(
            "C1", "evidence recomputes from raw telemetry", c1_status,
            (f"{len(evidence)} item(s) re-derived independently; "
             f"{len(mismatches)} mismatch(es)") if evidence
            else ("no channel moved materially, so no evidence was offered "
                  "-- consistent with a quiet machine" if not alerting
                  else "no evidence items were offered"),
            mismatches=mismatches, n_evidence=len(evidence)))

        # --- C2: does the evidence match the diagnosis? -------------------
        ftype = investigation.get("likely_failure_type")
        sig = expected_signature(ftype) if ftype else {}
        observed = observed_directions(evidence)
        matched = [c for c, d in sig.items() if observed.get(c) == d]
        contradicted = [c for c, d in sig.items()
                        if c in observed and observed[c] != d
                        and observed[c] != "flat"]
        ratio = len(matched) / max(len(sig), 1)
        c2_status = (
            "n/a" if not alerting or not sig else
            "pass" if ratio >= 0.5 and not contradicted else
            "warn" if ratio >= 0.34 else "fail")
        checks.append(self._check(
            "C2", "evidence is consistent with the diagnosis", c2_status,
            "no diagnosis was asserted (score is below the alert threshold), "
            "so there is nothing to corroborate" if c2_status == "n/a" else
            (f"{len(matched)}/{len(sig)} signature channels for "
             f"{ftype} are present"
             + (f"; contradicted: {contradicted}" if contradicted else "")),
            expected=sig, observed=observed, match_ratio=round(ratio, 3)))

        # --- C3: does it clear the model's own threshold? -----------------
        prob = float(prediction.get("failure_probability", 0.0))
        thr = float(prediction.get("threshold", 0.5))
        checks.append(self._check(
            "C3",
            "prediction clears the decision threshold" if alerting
            else "prediction is correctly held below the threshold",
            "pass",
            f"probability {prob:.4f} vs validation-tuned threshold {thr:.4f}"
            + ("" if alerting else " -- no alert raised"),
            probability=prob, threshold=thr))

        # --- C4: is the confidence measured or asserted? ------------------
        conf = float(prediction.get("confidence", 0.0))
        curve_backed = bool(getattr(bundle.reliability, "edges", []))
        checks.append(self._check(
            "C4", "confidence is empirically grounded",
            "pass" if curve_backed and conf > 0 else "warn",
            (f"confidence {conf:.3f} read from the validation reliability "
             "curve" if curve_backed else
             "no reliability curve fitted; confidence falls back to the raw "
             "score and should not be read as a precision estimate"),
            confidence=conf, basis=prediction.get("confidence_basis", "")))

        # --- C5: is there an innocent explanation? ------------------------
        octx = investigation.get("operating_context", {})
        load_pct = float(octx.get("load_change_pct_3h", 0.0))
        ambient = float(octx.get("ambient_change_c_3h", 0.0))
        alternatives = []
        if abs(load_pct) >= 8.0:
            alternatives.append({
                "explanation": "production load change",
                "evidence": f"load moved {load_pct:+.1f}% over 3 h",
                "verdict": "plausible -- symptoms may be duty-driven"})
        else:
            alternatives.append({
                "explanation": "production load change",
                "evidence": f"load moved only {load_pct:+.1f}% over 3 h",
                "verdict": "rejected -- load is flat"})
        if ambient > 3.0:
            alternatives.append({
                "explanation": "high ambient temperature",
                "evidence": f"ambient rose {ambient:+.1f} C over 3 h",
                "verdict": ("plausible for a heat-only signature"
                            if ftype == "motor_overheating"
                            else "does not explain the non-thermal channels")})
        else:
            alternatives.append({
                "explanation": "high ambient temperature",
                "evidence": f"ambient moved {ambient:+.1f} C over 3 h",
                "verdict": "rejected -- ambient is stable"})
        unresolved = [a for a in alternatives if a["verdict"].startswith("plausible")]
        checks.append(self._check(
            "C5", "benign alternatives considered",
            "n/a" if not alerting else ("pass" if not unresolved else "warn"),
            (f"{len(alternatives)} alternative(s) evaluated; "
             f"{len(unresolved)} still plausible"),
            alternatives=alternatives))

        # --- C6: are the proposed actions permitted? ----------------------
        from ..simulation.interventions import CATALOGUE
        plan = remediation.get("plan", [])
        illegal = [p["intervention_id"] for p in plan
                   if p["intervention_id"] not in CATALOGUE]
        checks.append(self._check(
            "C6", "all proposed actions are in the approved catalogue",
            "fail" if illegal else ("pass" if plan else "n/a"),
            (f"{len(plan)} action(s) checked" if plan else "no actions proposed"),
            illegal=illegal))

        # --- C7: did the simulation actually help? ------------------------
        improved = bool(simulation.get("simulation_shows_improvement"))
        control = simulation.get("no_action", {}).get(
            "failure_probability_simulated")
        arms = [a for a in simulation.get("arms", []) if a.get("simulated")]
        best = (min(arms, key=lambda a: a["failure_probability_simulated"])
                if arms else None)
        checks.append(self._check(
            "C7", "simulation beats its own do-nothing control",
            "pass" if improved else ("warn" if arms else "n/a"),
            (f"control {control:.3f} vs best action "
             f"{best['failure_probability_simulated']:.3f} "
             f"({best['delta_vs_no_action']:+.3f})"
             if best and control is not None else
             "no action produced a simulatable telemetry change"),
            control=control,
            best=best["intervention_id"] if best else None))

        # --- C8: does the narrative quote unsupported numbers? ------------
        narrative = " ".join(str(investigation.get(k, ""))
                             for k in ("conclusion", "narrative"))
        allowed = {round(abs(float(e["value"])), 1) for e in evidence}
        allowed |= {round(abs(float(e["value"])), 0) for e in evidence}
        # Fraction-valued evidence (monotonicity, ratios) is quoted as a
        # percentage in its own claim, so 0.83 legitimately appears as "83%".
        allowed |= {round(abs(float(e["value"])) * 100, 0) for e in evidence}
        allowed |= {round(abs(float(e["value"])) * 100, 1) for e in evidence}
        # The window length ("over the last 3 hours") is part of the claim's
        # phrasing, not a measurement being asserted.
        allowed |= {float(e.get("recompute", {}).get("hours", 0.0))
                    for e in evidence}
        allowed |= {round(prob, 2), round(conf, 2), round(thr, 2)}
        allowed |= {round(h["score"], 2)
                    for h in investigation.get("ranked_hypotheses", [])}
        allowed |= {round(h["signature_match"], 2)
                    for h in investigation.get("ranked_hypotheses", [])}
        quoted = {abs(float(m)) for m in
                  re.findall(r"[-+]?\d+\.?\d*", narrative)}
        unsupported = sorted(
            q for q in quoted
            if not any(abs(q - a) <= max(0.05, abs(a) * 0.02) for a in allowed)
            and q > 1.0)
        checks.append(self._check(
            "C8", "narrative quotes only supported numbers",
            "pass" if not unsupported else "fail",
            (f"{len(quoted)} number(s) in the narrative; "
             f"{len(unsupported)} not traceable to evidence"),
            unsupported=unsupported[:8]))

        # --- C9: approval gate on consequential actions -------------------
        needs = [p for p in plan if p["requires_approval"]]
        gated = remediation.get("approval_gate", {}).get("required", False)
        checks.append(self._check(
            "C9", "consequential actions require human approval",
            "pass" if (not needs or gated) else "fail",
            (f"{len(needs)} action(s) require approval; gate "
             f"{'present' if gated else 'MISSING'}")))

        # --- C10: was the disagreement actually resolved on evidence? -----
        if adjudication:
            d = float(adjudication.get("degradation_score", 0.0))
            c = float(adjudication.get("confound_score", 0.0))
            decision = adjudication.get("decision", "")
            overturned = decision == "overturned"
            contested = decision == "contested"
            c10 = ("warn" if contested else "pass")
            detail = (
                f"degradation {d:.2f} vs benign {c:.2f}, margin "
                f"{adjudication.get('margin', 0):+.2f} -> {decision}")
            if overturned:
                detail += (
                    "; the model's flag was overturned by the benign case ("
                    f"{adjudication.get('leading_benign_explanation')})")
            if contested:
                detail += "; too close to call without a physical check"
            checks.append(self._check(
                "C10", "the competing hypotheses were resolved on evidence",
                c10, detail,
                degradation_score=d, confound_score=c,
                decision=decision,
                changed_the_model_verdict=bool(
                    adjudication.get("changed_the_model_verdict"))))

        # --- verdict --------------------------------------------------------
        fails = [c for c in checks if c["status"] == "fail"]
        warns = [c for c in checks if c["status"] == "warn"]
        skipped = [c for c in checks if c["status"] == "n/a"]
        scope = "alert" if alerting else "no-alert"
        if fails:
            verdict, headline = "FAIL", (
                f"{len(fails)} check(s) failed -- this result must not be "
                "presented as verified")
        elif warns:
            verdict, headline = "PASS_WITH_WARNINGS", (
                f"all applicable checks passed; {len(warns)} caveat(s) to read "
                "before acting")
        elif alerting:
            verdict, headline = "PASS", "all checks passed"
        else:
            verdict, headline = "PASS", (
                "no alert was raised and the quiet reading is consistent with "
                "the telemetry")

        # Act-ability is a separate question from correctness.
        actionable = (verdict != "FAIL" and alerting
                      and conf >= MIN_ACTIONABLE_CONFIDENCE)

        return {
            "_action": f"Ran {len(checks)} verification checks -> {verdict}.",
            "_reason": headline,
            "_verification": verdict,
            "verdict": verdict,
            "headline": headline,
            "scope": scope,
            "checks": checks,
            "n_passed": sum(c["status"] == "pass" for c in checks),
            "n_warnings": len(warns),
            "n_failed": len(fails),
            "n_not_applicable": len(skipped),
            "failed_checks": [c["id"] for c in fails],
            "warned_checks": [c["id"] for c in warns],
            "safe_to_act": bool(actionable),
            "action_guidance": (
                "Proceed to human approval." if actionable else
                "No action required: the machine is not predicted to fail "
                "within the horizon." if not alerting else
                "Do not act on this result: "
                + ("verification failed" if fails else
                   f"confidence {conf:.2f} is below the "
                   f"{MIN_ACTIONABLE_CONFIDENCE} actionable floor")),
        }
