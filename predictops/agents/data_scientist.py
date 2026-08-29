"""Agent 1 -- Data Scientist.

Profiles the dataset before anything is trained, and turns the profile into
concrete instructions for the Model Research agent.  Its recommendations are
derived from measurements on the **training split only** -- profiling on the
full frame would itself be a leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZON_HOURS, LOOKBACK_STEPS, RESOLUTION_MINUTES
from ..data.schemas import LEAKY_COLUMNS
from ..ml.features import RAW_FEATURES, feature_columns
from .base import Agent, AgentContext

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "risks", "recommendations"],
    "additionalProperties": False,
}


def _point_biserial(x: pd.Series, y: np.ndarray) -> float:
    """Correlation between a continuous feature and the binary label."""
    v = x.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 50:
        return 0.0
    v, yy = v[ok], y[ok]
    if v.std() < 1e-12 or yy.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(v, yy)[0, 1])


class DataScientistAgent(Agent):
    name = "data_scientist"
    brief = ("Profile the telemetry, find leakage and imbalance risks, and "
             "recommend the feature set and training window.")
    system_prompt = (
        "You are a senior data scientist reviewing an industrial telemetry "
        "dataset before any model is trained. You are given a computed profile. "
        "Summarise the three findings that most affect modelling choices. "
        "Do not invent statistics; use only the numbers provided."
    )

    def tools(self) -> list[str]:
        return ["pandas.profile", "corr.point_biserial", "split.audit",
                "leakage.scan"]

    def investigate(self, ctx: AgentContext, **kwargs) -> dict:
        df = ctx.data.df
        train = ctx.data.split("train")

        # --- inventory ----------------------------------------------------
        n_rows, n_machines = len(df), df["machine_id"].nunique()
        span = (df["timestamp"].max() - df["timestamp"].min())
        resolution = int(RESOLUTION_MINUTES)

        raw_present = [c for c in RAW_FEATURES if c in df.columns]
        engineered = feature_columns(df, "engineered")

        # --- missing data --------------------------------------------------
        # The frame reaching the agent is already imputed, so counting NaNs
        # here would report a misleading zero.  `sensor_dropout` marks exactly
        # the rows the historian failed to deliver, and a dropout blanks every
        # sensor channel at once, so it is the honest per-channel figure.
        dropout_rate = float(df.get("sensor_dropout", pd.Series([0])).mean())
        blanked = ["temperature", "vibration", "pressure", "rpm", "current",
                   "voltage", "power", "load", "humidity"]
        missing = {c: round(dropout_rate, 5) for c in blanked if c in df.columns}
        downtime_rate = float(df["is_downtime"].mean())

        # --- class balance -------------------------------------------------
        pos_rate = float(train["label"].mean())
        imbalance_ratio = round((1 - pos_rate) / max(pos_rate, 1e-9), 1)

        # --- leakage scan ---------------------------------------------------
        leaks_offered = sorted(set(engineered) & LEAKY_COLUMNS)
        split_bounds = {
            s: {"start": str(g["timestamp"].min()), "end": str(g["timestamp"].max()),
                "rows": int(len(g)), "positives": int(g["label"].sum())}
            for s, g in df.groupby("split")
        }
        chronological = (
            split_bounds.get("train", {}).get("end", "")
            < split_bounds.get("val", {}).get("start", "z")
            < split_bounds.get("test", {}).get("start", "z"))

        # --- which features actually carry signal (train split only) --------
        y = train["label"].to_numpy(dtype=float)
        scored = sorted(
            ((c, abs(_point_biserial(train[c], y))) for c in engineered),
            key=lambda kv: -kv[1])
        top = [{"feature": c, "abs_corr": round(v, 4)} for c, v in scored[:12]]
        dead = [c for c, v in scored if v < 0.005]

        raw_best = max((abs(_point_biserial(train[c], y))
                        for c in raw_present), default=0.0)
        eng_best = scored[0][1] if scored else 0.0
        engineering_gain = round(eng_best - raw_best, 4)

        # --- temporal structure --------------------------------------------
        sample = train[train.machine_id == train.machine_id.iloc[0]]
        autocorr = {}
        for c in ("vibration", "temperature", "current"):
            s = sample[c].dropna()
            if len(s) > 100:
                autocorr[c] = round(float(s.autocorr(lag=6)), 4)

        # --- turn measurements into instructions ----------------------------
        risks, recommendations = [], []
        if leaks_offered:
            risks.append(f"ground-truth columns exposed as features: {leaks_offered}")
        else:
            risks.append("no ground-truth column appears in the feature list")
        if not chronological:
            risks.append("splits are not chronological -- future may leak into train")
        else:
            risks.append(
                "splits are chronological with a purge gap; the test period is "
                f"{split_bounds.get('test', {}).get('start', '?')} onwards")
        if imbalance_ratio > 20:
            risks.append(f"severe class imbalance ({imbalance_ratio}:1 negative:positive)")
        if missing:
            risks.append(f"sensor dropouts blank all channels together on "
                         f"{dropout_rate:.2%} of rows; imputation is "
                         "forward-fill with a train-median fallback")
        if downtime_rate > 0:
            risks.append(f"{downtime_rate:.2%} of rows are downtime and must be "
                         "excluded from both training and scoring")

        recommendations.append(
            "use PR-AUC (not accuracy or ROC-AUC) to select models: at "
            f"{pos_rate:.2%} positives, accuracy is uninformative")
        recommendations.append(
            "subsample negatives for training only; score on the untouched "
            "distribution")
        recommendations.append(
            "ablate raw against engineered features rather than trusting these "
            f"correlations: the best engineered feature gains only "
            f"{engineering_gain:+.3f} marginally, but marginal correlation "
            "cannot see the interaction that matters here (vibration rising "
            "*while load stays flat*)")
        if any(v > 0.85 for v in autocorr.values()):
            recommendations.append(
                "channels are strongly autocorrelated at 1 h lag -- sequence "
                "models are worth testing against a tabular baseline")
        recommendations.append(
            f"lookback of {LOOKBACK_STEPS} samples "
            f"({LOOKBACK_STEPS * resolution / 60:.0f} h) covers the "
            f"{HORIZON_HOURS} h horizon without spanning typical downtime blocks")
        if dead:
            recommendations.append(
                f"{len(dead)} features carry almost no marginal signal "
                f"(|corr| < 0.005), e.g. {dead[:3]}")

        findings = {
            "_action": "Profiled the dataset and derived the modelling plan.",
            "_reason": (f"{n_rows:,} rows across {n_machines} machines at "
                        f"{resolution} min resolution; {pos_rate:.2%} positive."),
            "inventory": {
                "rows": n_rows, "machines": n_machines,
                "span_days": round(span.total_seconds() / 86400, 2),
                "resolution_minutes": resolution,
                "raw_channels": raw_present,
                "n_engineered_features": len(engineered),
            },
            "data_quality": {
                "missing_by_channel": missing,
                "sensor_dropout_rate": round(dropout_rate, 5),
                "downtime_rate": round(downtime_rate, 5),
            },
            "class_balance": {
                "train_positive_rate": round(pos_rate, 5),
                "imbalance_ratio": imbalance_ratio,
                "train_positives": int(train["label"].sum()),
            },
            "leakage_audit": {
                "leaky_columns_in_feature_list": leaks_offered,
                "splits_chronological": bool(chronological),
                "split_bounds": split_bounds,
                "verdict": ("clean" if (not leaks_offered and chronological)
                            else "review required"),
            },
            "temporal_structure": {"autocorr_lag_1h": autocorr},
            "feature_signal": {
                "top_features": top,
                "n_low_signal_features": len(dead),
                "engineering_gain_over_raw": engineering_gain,
            },
            "risks": risks,
            "recommendations": recommendations,
            "recommended_plan": {
                "selection_metric": "pr_auc",
                "primary_report_metric": "f1",
                "feature_sets_to_ablate": ["raw", "engineered"],
                "try_sequence_models": bool(any(v > 0.85
                                                for v in autocorr.values())),
                "lookback_steps": LOOKBACK_STEPS,
                "balance_training_data": imbalance_ratio > 20,
            },
        }
        return findings

    def narrate(self, ctx: AgentContext, findings: dict):
        prompt = (
            "Computed dataset profile (JSON):\n"
            f"{findings['inventory']}\n{findings['class_balance']}\n"
            f"{findings['leakage_audit']['verdict']}\n"
            f"top features: {findings['feature_signal']['top_features'][:5]}\n\n"
            "Write a one-sentence headline, then list the risks and "
            "recommendations you would give the modelling team."
        )
        fallback = {"headline": findings["_reason"],
                    "risks": findings["risks"],
                    "recommendations": findings["recommendations"]}
        res = ctx.provider.structured(self.system_prompt, prompt, SCHEMA, fallback)
        findings["headline"] = res.data.get("headline", fallback["headline"])
        findings["risks"] = res.data.get("risks", findings["risks"])
        findings["recommendations"] = res.data.get("recommendations",
                                                   findings["recommendations"])
        return findings, res
