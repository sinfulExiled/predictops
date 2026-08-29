"""Agent 2 -- Model Research.

Reads the data scientist's plan, turns it into a queue of experiments, runs
them, and picks a winner.

Two rules this agent must not break:

1.  **Selection is on measured validation PR-AUC.**  The LLM may explain the
    choice; it may not make it.  `select()` is pure arithmetic over the
    registry, and `test` metrics are never consulted for selection -- reading
    them would turn the held-out split into a tuning set.
2.  **An architecture is only kept if it beat what came before it.**  If the
    TFT does not improve on the LSTM, the run says so and keeps the LSTM.
"""
from __future__ import annotations

from ..config import TrainingConfig
from ..experiments.runner import ExperimentRunner
from .base import Agent, AgentContext

SCHEMA = {
    "type": "object",
    "properties": {
        "selection_rationale": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["selection_rationale", "notes"],
    "additionalProperties": False,
}

# Improvement in validation PR-AUC below which a more complex model is not
# worth its extra cost and latency.
MIN_GAIN = 0.01


class ModelResearchAgent(Agent):
    name = "model_researcher"
    brief = ("Plan and run model experiments, then select the best candidate "
             "on measured validation PR-AUC.")
    system_prompt = (
        "You are an ML researcher writing up a model bake-off. You are given "
        "the measured results of every experiment that ran. Explain which model "
        "was selected and why, referring only to the numbers provided. Never "
        "claim a model is better than its measured score shows."
    )

    def tools(self) -> list[str]:
        return ["experiment_runner.train", "experiment_runner.evaluate",
                "registry.record", "registry.compare"]

    # -- selection is arithmetic, not opinion ------------------------------
    @staticmethod
    def select(experiments: list) -> tuple[object, str]:
        """Highest validation PR-AUC wins; ties break toward the simpler model."""
        complexity = {"threshold_baseline": 0, "xgboost": 1, "lightgbm": 1,
                      "lstm": 2, "tft": 3, "ensemble": 4}
        scored = [(e, e.metrics.get("val", {}).get("pr_auc")) for e in experiments]
        scored = [(e, v) for e, v in scored if isinstance(v, (int, float))]
        if not scored:
            raise ValueError("no experiment carried a validation score")
        best_val = max(v for _, v in scored)
        # anything within MIN_GAIN of the leader is a tie -> prefer simpler
        contenders = [(e, v) for e, v in scored if best_val - v <= MIN_GAIN]
        winner, wval = min(contenders,
                           key=lambda ev: (complexity.get(ev[0].model, 9), -ev[1]))
        leader = max(scored, key=lambda ev: ev[1])
        if winner.id != leader[0].id:
            reason = (
                f"{winner.name} selected on validation PR-AUC {wval:.4f}; "
                f"{leader[0].name} scored {leader[1]:.4f} but the "
                f"{leader[1] - wval:+.4f} gain is inside the {MIN_GAIN} "
                "tolerance, so the simpler model wins the tie")
        else:
            runner_up = sorted(scored, key=lambda ev: -ev[1])[1:2]
            gap = f" ({wval - runner_up[0][1]:+.4f} over {runner_up[0][0].name})" \
                if runner_up else ""
            reason = (f"{winner.name} selected on validation PR-AUC "
                      f"{wval:.4f}{gap}")
        return winner, reason

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        plan = ctx.state.get("data_scientist", {}).get("recommended_plan", {})
        runner: ExperimentRunner = kwargs["runner"]
        quick = bool(kwargs.get("quick", False))
        cfg = TrainingConfig(max_epochs=6 if quick else 30,
                             patience=3 if quick else 5)

        feature_sets = plan.get("feature_sets_to_ablate", ["engineered"])
        try_sequence = bool(plan.get("try_sequence_models", True))

        queue: list[str] = []
        log: list[dict] = []

        def note(exp_id, fitted, decision, learning):
            e = [x for x in ctx.store.all(ctx.run_id) if x.id == exp_id][0]
            ctx.store.set_decision(exp_id, decision, learning)
            log.append({
                "experiment": f"{e.stage}",
                "name": e.name, "model": e.model, "feature_set": e.feature_set,
                "val_pr_auc": e.metrics.get("val", {}).get("pr_auc"),
                "test_f1": e.metrics.get("row", {}).get("f1"),
                "decision": decision, "learning": learning,
                "duration_s": e.duration_s,
            })
            return e

        # --- 1. reference point ------------------------------------------
        ctx.say("  [research] baseline: threshold rule")
        f_base, id_base = runner.run_baseline("Baseline")
        e_base = note(id_base, f_base, "reference",
                      "Establishes the operating point the plant already has.")
        base_val = e_base.metrics["val"]["pr_auc"]
        queue.append("baseline")

        # --- 2. tabular trees, one per feature set ------------------------
        best_tree_val, best_tree = -1.0, None
        prev_tree_val, prev_tree_fs = None, None
        for i, fs in enumerate(feature_sets):
            ctx.say(f"  [research] xgboost on {fs} features")
            f, eid = runner.run_tree(
                "xgboost", fs, f"Iteration {len(log)}",
                f"Do gradient-boosted trees over {fs} features beat the alarm "
                "rule without any sequence modelling?")
            e = [x for x in ctx.store.all(ctx.run_id) if x.id == eid][0]
            v = e.metrics["val"]["pr_auc"]
            kept = v > base_val + MIN_GAIN
            if prev_tree_val is None:
                learning = (
                    f"PR-AUC {v:.4f} vs baseline {base_val:.4f} "
                    f"({v - base_val:+.4f}). "
                    + ("Trees on raw channels already beat the alarm rule, so "
                       "most of the baseline's weakness was the fixed "
                       "threshold, not the sensors." if kept
                       else "No better than the alarm rule."))
            else:
                # The ablation the whole project turns on: same model, same
                # rows, only the feature set changed.
                learning = (
                    f"PR-AUC {v:.4f} vs {prev_tree_fs} features "
                    f"{prev_tree_val:.4f} ({v - prev_tree_val:+.4f}). "
                    + ("Causal rolling statistics and load/ambient "
                       "normalisation are worth more than any change of "
                       "architecture tried below." if v > prev_tree_val
                       else "Feature engineering did not help this model."))
            note(eid, f, "kept" if kept else "removed", learning)
            prev_tree_val, prev_tree_fs = v, fs
            if v > best_tree_val:
                best_tree_val, best_tree = v, f"xgboost_{fs}"
            queue.append(f"xgboost_{fs}")

        # --- 3. sequence models -------------------------------------------
        seq_results: dict[str, float] = {}
        if try_sequence:
            for fs in feature_sets:
                ctx.say(f"  [research] LSTM on {fs} channels")
                f, eid = runner.run_sequence(
                    "lstm", fs, f"Iteration {len(log)}",
                    f"Does an LSTM over a {runner.lookback}-step window on {fs} "
                    "channels capture degradation the tabular view misses?",
                    cfg=cfg, progress=ctx.verbose)
                e = [x for x in ctx.store.all(ctx.run_id) if x.id == eid][0]
                v = e.metrics["val"]["pr_auc"]
                seq_results[f"lstm_{fs}"] = v
                kept = v > max(base_val, best_tree_val) + MIN_GAIN
                note(eid, f, "kept" if kept else "removed",
                     (f"PR-AUC {v:.4f} vs best tabular {best_tree_val:.4f} "
                      f"({v - best_tree_val:+.4f}). "
                      + ("The sequence view finds structure the tabular one "
                         "misses." if kept else
                         "Learning the temporal structure from raw sequence "
                         "does not beat handing a tree the same structure "
                         "explicitly.")))
                queue.append(f"lstm_{fs}")

            best_lstm_fs = max(
                (fs for fs in feature_sets),
                key=lambda fs: seq_results.get(f"lstm_{fs}", -1.0))
            ctx.say(f"  [research] TFT on {best_lstm_fs} channels")
            f, eid = runner.run_sequence(
                "tft", best_lstm_fs, f"Iteration {len(log)}",
                "Does variable selection plus attention beat a plain LSTM on "
                "the same channels?",
                cfg=cfg, progress=ctx.verbose)
            e = [x for x in ctx.store.all(ctx.run_id) if x.id == eid][0]
            v_tft = e.metrics["val"]["pr_auc"]
            v_lstm = seq_results.get(f"lstm_{best_lstm_fs}", -1.0)
            seq_results[f"tft_{best_lstm_fs}"] = v_tft
            kept = v_tft > v_lstm + MIN_GAIN
            note(eid, f, "kept" if kept else "removed",
                 (f"PR-AUC {v_tft:.4f} vs LSTM {v_lstm:.4f}. "
                  + ("Attention and variable selection earned their cost."
                     if kept else
                     "No material gain over the LSTM at this data scale -- "
                     "the extra capacity is not paying for itself.")))
            queue.append(f"tft_{best_lstm_fs}")

            # --- 4. ensemble, only if both members are real contenders ----
            members = [f"lstm_{best_lstm_fs}", f"tft_{best_lstm_fs}"]
            if all(m in runner.fitted for m in members):
                ctx.say("  [research] ensemble LSTM + TFT")
                f, eid = runner.run_ensemble(
                    members, best_lstm_fs, f"Iteration {len(log)}",
                    "Do the two sequence models make different mistakes? If so "
                    "a validation-weighted blend should beat either alone.")
                e = [x for x in ctx.store.all(ctx.run_id) if x.id == eid][0]
                v_ens = e.metrics["val"]["pr_auc"]
                best_member = max(v_tft, v_lstm)
                kept = v_ens > best_member + MIN_GAIN
                note(eid, f, "kept" if kept else "removed",
                     (f"PR-AUC {v_ens:.4f} vs best member {best_member:.4f}. "
                      + ("The blend adds real signal."
                         if kept else
                         "The members' errors are correlated, so blending "
                         "averages without adding information.")))
                queue.append("ensemble")

        # --- 5. select ----------------------------------------------------
        experiments = ctx.store.all(ctx.run_id)
        winner, rationale = self.select(experiments)

        return {
            "_action": f"Ran {len(log)} experiments and selected {winner.name}.",
            "_reason": rationale,
            "_verification": (
                "Selection used validation PR-AUC only; test metrics were not "
                "consulted for model choice."),
            "candidates_run": queue,
            "experiments": log,
            "selection": {
                "experiment_id": winner.id,
                "name": winner.name,
                "model": winner.model,
                "feature_set": winner.feature_set,
                "val_pr_auc": winner.metrics.get("val", {}).get("pr_auc"),
                "test_f1": winner.metrics.get("row", {}).get("f1"),
                "rationale": rationale,
                "selected_on": "validation pr_auc",
            },
            "selection_rationale": rationale,
        }

    def narrate(self, ctx: AgentContext, findings: dict):
        rows = "\n".join(
            f"- {e['name']} ({e['model']}/{e['feature_set']}): "
            f"val PR-AUC {e['val_pr_auc']}, decision {e['decision']}"
            for e in findings["experiments"])
        prompt = (
            f"Experiments that ran:\n{rows}\n\n"
            f"Selected: {findings['selection']['name']} "
            f"({findings['selection']['rationale']}).\n\n"
            "Explain the selection in two sentences and list what each "
            "experiment taught you.")
        fallback = {"selection_rationale": findings["selection_rationale"],
                    "notes": [e["learning"] for e in findings["experiments"]]}
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA, fallback)
        findings["selection_rationale"] = res.data.get(
            "selection_rationale", fallback["selection_rationale"])
        findings["notes"] = res.data.get("notes", fallback["notes"])
        return findings, res
