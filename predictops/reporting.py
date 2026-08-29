"""Human-readable incident report.

The deliverable a maintenance planner reads at 06:00. It has to answer six
questions in order -- what, when, why, what to do, what that buys, how sure --
and it has to be honest about the last one. Every claim carries its provenance
(`measured`, `model`, `simulated`) so a reader knows which numbers are
observations and which are projections.
"""
from __future__ import annotations

import textwrap

RULE = "=" * 78
THIN = "-" * 78


def _wrap(text: str, indent: str = "  ", width: int = 78) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _bar(value: float, width: int = 28) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def render_incident(report: dict) -> str:
    pred = report["prediction"]
    inv = report["investigation"]
    rem = report["remediation"]
    sim = report["simulation"]
    ver = report["verification"]

    prob = pred["failure_probability"]
    win = pred.get("prediction_window_hours") or {}
    lines: list[str] = []

    # --- header ---------------------------------------------------------
    lines.append(RULE)
    lines.append(f"PREDICTOPS INCIDENT REPORT{'':<20}{report['machine_id']:>32}")
    lines.append(f"{'assessed at ' + str(report['timestamp']):>78}")
    lines.append(RULE)
    lines.append("")

    # --- WHAT ------------------------------------------------------------
    band = pred["risk_band"].upper()
    lines.append(f"  FAILURE PROBABILITY   {prob:>6.1%}   [{_bar(prob)}]  {band}")
    lines.append(f"  alert threshold       {pred['threshold']:>6.1%}"
                 f"   (tuned on validation, frozen before test)")
    lines.append("")

    # --- WHEN ------------------------------------------------------------
    if win.get("eta_hours") is not None:
        lines.append(f"  EXPECTED WINDOW       {win['window_low_h']:.1f}-"
                     f"{win['window_high_h']:.1f} hours  "
                     f"(point estimate {win['eta_hours']:.1f} h)   [model]")
    else:
        lines.append("  EXPECTED WINDOW       not estimated "
                     "(no warning-window model output for this row)")
    ftype = inv.get("likely_failure_type") or pred.get("failure_type")
    if ftype:
        lines.append(f"  LIKELY FAILURE        {ftype.replace('_', ' ')}"
                     f"   ({pred.get('failure_type_confidence', 0):.0%} "
                     "classifier confidence)   [model]")
    lines.append("")

    # --- WHY --------------------------------------------------------------
    lines.append(THIN)
    lines.append("  WHY")
    lines.append(THIN)
    for e in inv.get("evidence", []):
        lines.append(f"  [{e['id']}] {e['claim']}   [measured]")
    ctx = inv.get("operating_context", {})
    if ctx:
        lines.append("")
        lines.append(f"       operating context: load {ctx.get('load_change_pct_3h', 0):+.1f}% "
                     f"/ 3 h, ambient {ctx.get('ambient_change_c_3h', 0):+.1f} C / 3 h")
    sims = inv.get("similar_past_failures", [])
    if sims:
        top = sims[0]
        same = sum(1 for s in sims if s["failure_type"] == top["failure_type"])
        lines.append(f"       nearest historical match: {top['failure_type']} "
                     f"on {top['machine_id']} "
                     f"({same} of {len(sims)} neighbours agree)   [measured]")
    mh = inv.get("maintenance_history", {})
    if mh:
        lines.append(f"       {mh.get('operating_hours_since_service', 0):.0f} h "
                     f"since last service; "
                     f"{mh.get('past_failures_on_this_machine', 0)} prior "
                     "failure(s) on this machine")
    lines.append("")
    if inv.get("narrative"):
        lines.append(_wrap(inv["narrative"], indent="  "))
        lines.append("")

    adj = report.get("adjudication") or {}
    deg = report.get("degradation_case") or {}
    con = report.get("confound_case") or {}
    if adj:
        lines.append(THIN)
        lines.append("  COMPETING READINGS")
        lines.append(THIN)
        lines.append(f"  a fault is developing {deg.get('score', 0):>34.2f}")
        lines.append(f"       {deg.get('conclusion', '')[:66]}")
        lines.append(f"  nothing is wrong      {con.get('score', 0):>34.2f}")
        lines.append(f"       {con.get('conclusion', '')[:66]}")
        lines.append("")
        lines.append(f"  DECISION  {adj.get('decision', '').upper():<20}"
                     f"margin {adj.get('margin', 0):+.2f}")
        lines.append(_wrap(adj.get("rationale", ""), indent="       "))
        if adj.get("changed_the_model_verdict"):
            lines.append(_wrap(
                "This overruled the model's own verdict.", indent="       "))
        lines.append("")

    ranked = inv.get("ranked_hypotheses", [])
    if len(ranked) > 1:
        lines.append("  Hypotheses considered:")
        for h in ranked[:4]:
            lines.append(f"    {h['score']:.2f}  {h['failure_type']:<24}"
                         f"signature {h['signature_match']:.0%}, "
                         f"classifier {h['classifier_probability']:.0%}, "
                         f"history {h['historical_vote']:.0%}")
        lines.append("")

    # --- WHAT TO DO -------------------------------------------------------
    lines.append(THIN)
    lines.append("  RECOMMENDED ACTION")
    lines.append(THIN)
    for step in rem.get("plan", []):
        tag = "diagnostic" if step["is_diagnostic"] else step["risk"] + " risk"
        lines.append(f"  {step['order']}. {step['title']}   ({tag}, "
                     f"${step['cost_usd']:,.0f}"
                     + (f", {step['downtime_hours']:.1f} h downtime"
                        if step["downtime_hours"] else "") + ")")
        lines.append(_wrap(step["why"], indent="     "))
        for pre in step.get("preconditions", []):
            lines.append(f"       requires: {pre}")
    for note in rem.get("notes", []):
        lines.append("")
        lines.append(_wrap(note, indent="  "))
    lines.append("")
    lines.append(f"  Estimated cost ${rem.get('estimated_cost_usd', 0):,.0f}"
                 f"   downtime {rem.get('estimated_downtime_hours', 0):.1f} h")
    lines.append("")

    # --- WHAT THAT BUYS ----------------------------------------------------
    lines.append(THIN)
    lines.append("  IF WE ACT  (simulated -- not a forecast)")
    lines.append(THIN)
    control = sim.get("no_action", {}).get("failure_probability_simulated")
    if control is not None:
        lines.append(f"  do nothing, {sim['horizon_hours']:.0f} h from now"
                     f"{control:>28.1%}   [simulated]")
    for arm in sim.get("arms", []):
        if not arm.get("simulated"):
            lines.append(f"  {arm['title'][:44]:<44}"
                         f"{'not simulated':>22}   ({arm['reason_not_simulated'].split('--')[0].strip()})")
            continue
        p = arm["failure_probability_simulated"]
        lines.append(f"  {arm['title'][:44]:<44}{p:>22.1%}   "
                     f"({arm['delta_vs_no_action']:+.1%} vs no action)")
    if sim.get("best_by_value"):
        lines.append("")
        lines.append(f"  best risk reduction per dollar: {sim['best_by_value']}")
    lines.append("")
    lines.append(_wrap(sim.get("caveat", ""), indent="  "))
    lines.append("")

    # --- HOW SURE ----------------------------------------------------------
    lines.append(THIN)
    lines.append(f"  VERIFICATION   {ver['verdict']}")
    lines.append(THIN)
    for c in ver.get("checks", []):
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL",
                "n/a": " -- "}[c["status"]]
        lines.append(f"  {mark}  {c['id']}  {c['check']}")
        lines.append(f"         {c['detail']}")
    lines.append("")
    lines.append(f"  Confidence {pred['confidence']:.0%} "
                 f"-- {pred['confidence_basis']}.")
    lines.append("")
    lines.append(_wrap(ver["action_guidance"], indent="  "))
    gate = rem.get("approval_gate", {})
    if gate.get("required"):
        lines.append("")
        lines.append(_wrap(
            f"APPROVAL REQUIRED for: {', '.join(gate['actions'])}. "
            + gate.get("statement", ""), indent="  "))
    lines.append("")
    lines.append(THIN)
    lines.append(f"  model {pred['model']['kind']} on "
                 f"{pred['model']['feature_set']} features, "
                 f"{pred['model']['lookback_steps']}-step lookback"
                 f"   |   {report.get('duration_s', 0):.1f}s   |   "
                 f"run {report.get('run_id', '')}")
    lines.append(RULE)
    return "\n".join(lines)


def render_fleet(df, limit: int = 15) -> str:
    lines = [RULE, f"PREDICTOPS FLEET STATUS{'':<20}"
             f"{str(df.attrs.get('timestamp', '')):>34}", RULE,
             f"  {'MACHINE':<16}{'RISK':>8}{'CONF':>8}  {'STATUS':<8}"
             f"{'VIB':>8}{'TEMP':>8}{'LOAD':>7}", THIN]
    for _, r in df.head(limit).iterrows():
        p = r.get("failure_probability")
        if p is None or p != p:
            lines.append(f"  {r['machine_id']:<16}{'--':>8}{'--':>8}  "
                         f"{'down':<8}")
            continue
        lines.append(
            f"  {r['machine_id']:<16}{p:>8.1%}{r.get('confidence', 0):>8.1%}  "
            f"{r.get('status', ''):<8}{r.get('vibration', 0):>8.2f}"
            f"{r.get('temperature', 0):>8.1f}{r.get('load', 0):>7.2f}")
    lines.append(RULE)
    return "\n".join(lines)
