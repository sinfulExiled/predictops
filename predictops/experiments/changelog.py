"""Improvement changelog, generated from the experiment registry.

Nothing here is written by hand.  Every row is read back out of SQLite, so a
number in the changelog exists only if an experiment actually produced it.
That is the point: the changelog and the results cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

from ..config import REPORT_DIR
from .registry import Experiment, ExperimentStore


def _fmt(v, nd=4) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _row_metric(e: Experiment, key: str, default=None):
    return e.metrics.get("row", {}).get(key, default)


def _val_metric(e: Experiment, key: str, default=None):
    return e.metrics.get("val", {}).get(key, default)


def _event_metric(e: Experiment, key: str, default=None):
    return e.metrics.get("event", {}).get(key, default)


def build_changelog(store: ExperimentStore, run_id: str | None = None) -> str:
    run_id = run_id or store.latest_run_id()
    exps = store.all(run_id)
    if not exps:
        return "_No experiments recorded._"

    base = next((e for e in exps if e.model == "threshold_baseline"), exps[0])
    base_f1 = _row_metric(base, "f1", 0.0) or 0.0

    lines: list[str] = []
    lines.append("## Improvement Changelog")
    lines.append("")
    lines.append(f"Run `{run_id}`. Every figure below was read back from the "
                 "experiment registry (`artifacts/experiments/experiments.db`); "
                 "none is hand-entered.")
    lines.append("")
    lines.append("Models are **selected on validation PR-AUC**. Test F1 is "
                 "reported for the same frozen threshold and is never used to "
                 "choose between candidates.")
    lines.append("")
    lines.append("| Stage | What was tried and why | Val PR-AUC | Test F1 | "
                 "vs baseline | Decision / learning |")
    lines.append("|---|---|---|---|---|---|")

    for e in exps:
        f1 = _row_metric(e, "f1")
        delta = (f"{(f1 - base_f1):+.4f}"
                 if isinstance(f1, (int, float)) and e.id != base.id else "--")
        why = e.hypothesis.replace("\n", " ").strip()
        learning = (e.learning or "").replace("\n", " ").strip()
        lines.append(
            f"| **{e.stage}** | {e.name}<br/><sub>{why}</sub> | "
            f"{_fmt(_val_metric(e, 'pr_auc'))} | {_fmt(f1)} | {delta} | "
            f"`{e.decision}` {learning} |")

    lines.append("")
    lines.append("### Operational view")
    lines.append("")
    lines.append("Row-level F1 is the model-selection metric. What a "
                 "maintenance planner actually feels is the next table: how "
                 "many real failures were caught, how much warning they got, "
                 "and how many times the crew was sent out for nothing.")
    lines.append("")
    lines.append("| Stage | Model | Events caught | Mean early warning (h) | "
                 "False alarms / machine / day | Precision | Recall |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in exps:
        det = _event_metric(e, "detected")
        tot = _event_metric(e, "n_events")
        caught = (f"{det}/{tot} ({_event_metric(e, 'detection_rate', 0) * 100:.0f}%)"
                  if det is not None else "--")
        lines.append(
            f"| {e.stage} | {e.model} | {caught} | "
            f"{_fmt(_event_metric(e, 'mean_early_warning_h'), 2)} | "
            f"{_fmt(_event_metric(e, 'false_alarms_per_machine_day'), 3)} | "
            f"{_fmt(_row_metric(e, 'precision'), 3)} | "
            f"{_fmt(_row_metric(e, 'recall'), 3)} |")

    # --- what actually moved the needle -----------------------------------
    lines.append("")
    lines.append("### Where the improvement came from")
    lines.append("")
    contributions = []
    ordered = [e for e in exps if _val_metric(e, "pr_auc") is not None]
    for prev, cur in zip(ordered, ordered[1:]):
        gain = _val_metric(cur, "pr_auc") - _val_metric(prev, "pr_auc")
        contributions.append((gain, prev, cur))
    for gain, prev, cur in sorted(contributions, key=lambda t: -t[0]):
        verdict = "**gain**" if gain > 0.01 else (
            "no change" if abs(gain) <= 0.01 else "regression")
        lines.append(f"- `{prev.name}` -> `{cur.name}`: "
                     f"{gain:+.4f} val PR-AUC -- {verdict}")

    best = store.best(run_id, metric="f1")
    kept = [e for e in exps if e.decision == "kept"]
    removed = [e for e in exps if e.decision == "removed"]
    lines.append("")
    lines.append(f"**Kept:** {', '.join(e.name for e in kept) or 'none'}  ")
    lines.append(f"**Removed:** {', '.join(e.name for e in removed) or 'none'}")
    if best is not None:
        lines.append("")
        lines.append(
            f"**Best test F1:** `{best.name}` at {_fmt(_row_metric(best, 'f1'))} "
            f"({(_row_metric(best, 'f1') - base_f1):+.4f} over the baseline's "
            f"{_fmt(base_f1)}).")

    total_s = sum(e.duration_s for e in exps)
    lines.append("")
    lines.append(f"_Total experiment compute: {total_s / 60:.1f} min across "
                 f"{len(exps)} recorded runs._")
    return "\n".join(lines)


def write_changelog(store: ExperimentStore, run_id: str | None = None,
                    path: Path | None = None) -> Path:
    path = path or (REPORT_DIR / "IMPROVEMENT_CHANGELOG.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_changelog(store, run_id), encoding="utf-8")
    return path
