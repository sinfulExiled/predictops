"""Agent 8 -- Orchestrator.

Owns the workflow and the shared state.  Agents exchange structured dicts
through `AgentContext.state`; nothing is passed as free text.

    dataset -> data scientist -> model research -> [bundle]
            -> prediction -> investigation -> remediation
            -> simulation -> verification -> report

The research half is expensive and runs once; the incident half runs per
machine.  `PredictOpsEngine` separates the two so a dashboard can score a
machine in milliseconds without retraining anything.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPORT_DIR, TrainingConfig
from ..experiments.registry import ExperimentStore
from ..experiments.runner import ExperimentRunner
from ..llm.provider import LLMProvider, get_provider
from ..ml.bundle import BUNDLE_DIR, ModelBundle
from ..ml.dataset import PreparedData, prepare
from ..ml.diagnosis import (
    ReliabilityCurve,
    train_failure_type_classifier,
    train_time_to_failure_regressor,
)
from ..ml.features import feature_columns
from ..ml.service import ModelService
from .base import AgentContext, AgentResult
from .context import ContextAgent
from .data_scientist import DataScientistAgent
from .hypothesis import Adjudicator, ConfoundAdvocate, DegradationAdvocate
from .investigator import InvestigationAgent, SignatureLibrary
from .model_researcher import ModelResearchAgent
from .predictor import PredictionAgent
from .remediation import RemediationAgent
from .simulator import SimulationAgent
from .verifier import VerificationAgent


@dataclass
class IncidentReport:
    machine_id: str
    timestamp: str
    prediction: dict
    context: dict
    investigation: dict
    degradation_case: dict
    confound_case: dict
    adjudication: dict
    remediation: dict
    simulation: dict
    verification: dict
    trajectory: list[dict] = field(default_factory=list)
    llm_usage: dict = field(default_factory=dict)
    duration_s: float = 0.0
    run_id: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id, "machine_id": self.machine_id,
            "timestamp": self.timestamp, "prediction": self.prediction,
            "context": self.context, "investigation": self.investigation,
            "degradation_case": self.degradation_case,
            "confound_case": self.confound_case,
            "adjudication": self.adjudication,
            "remediation": self.remediation,
            "simulation": self.simulation, "verification": self.verification,
            "llm_usage": self.llm_usage, "duration_s": round(self.duration_s, 2),
            "trajectory": self.trajectory,
        }

    def save(self, directory: Path = REPORT_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        safe = self.timestamp.replace(":", "").replace(" ", "T")
        path = directory / f"incident_{self.machine_id}_{safe}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


class PredictOpsEngine:
    """Workflow and state manager."""

    def __init__(self, data: PreparedData | None = None,
                 store: ExperimentStore | None = None,
                 provider: LLMProvider | None = None,
                 run_id: str | None = None, verbose: bool = True):
        self.data = data if data is not None else prepare()
        self.store = store or ExperimentStore()
        self.provider = provider or get_provider()
        self.run_id = run_id or f"pipeline-{datetime.now():%Y%m%d-%H%M%S}"
        self.verbose = verbose
        self.ctx = AgentContext(run_id=self.run_id, data=self.data,
                                store=self.store, provider=self.provider,
                                verbose=verbose)
        self.bundle: ModelBundle | None = None
        self.service: ModelService | None = None
        self._busiest = None
        self.library: SignatureLibrary | None = None

    # -- research half ------------------------------------------------------
    def research(self, quick: bool = False) -> dict:
        """Profile, experiment, select, and assemble the deployable bundle."""
        ds = DataScientistAgent().execute(self.ctx)
        self.ctx.state["data_scientist"] = ds.output

        runner = ExperimentRunner(self.data, self.store, self.run_id)
        mr = ModelResearchAgent().execute(self.ctx, runner=runner, quick=quick)
        self.ctx.state["model_researcher"] = mr.output

        self.bundle = self._assemble_bundle(runner, mr.output)
        return {"data_scientist": ds.output, "model_research": mr.output}

    def _assemble_bundle(self, runner: ExperimentRunner,
                         research: dict) -> ModelBundle:
        sel = research["selection"]
        fitted = None
        for key in (f"{sel['model']}_{sel['feature_set']}", sel["model"],
                    "ensemble", "baseline"):
            if key in runner.fitted:
                fitted = runner.fitted[key]
                break
        if fitted is None:
            raise RuntimeError(f"selected model {sel} was not retained")

        scaler, cols, tr, va, te, ev_va, ev_te = runner.sequence_bits(
            fitted.feature_set)
        # The tree scores rows in its own feature set -- hardcoding "engineered"
        # here would feed a raw-feature model the wrong columns.
        tabular_cols = feature_columns(self.data.df, fitted.feature_set)

        bundle = ModelBundle(
            kind=fitted.kind, feature_set=fitted.feature_set,
            channels=list(fitted.channels or cols),
            tabular_columns=tabular_cols,
            threshold=fitted.threshold, scaler=fitted.scaler or scaler,
            lookback=runner.lookback,
            torch_model=fitted.torch_model, tree_model=fitted.tree_model,
            ensemble=({"members": (fitted.extra or {}).get("members"),
                       "member_kinds": (fitted.extra or {}).get("member_kinds"),
                       "weight_first": (fitted.extra or {}).get("weight_first")}
                      if fitted.kind == "ensemble" else None),
            metrics=fitted.metrics,
            selection_rationale=sel.get("rationale", ""))

        # confidence must be measured, so fit the curve on validation scores
        val_scores = self._score_split(bundle, runner, "val")
        bundle.reliability = ReliabilityCurve.fit(val_scores["y"],
                                                  val_scores["p"])

        # Failure type and ETA are independent models and always get the full
        # engineered feature set, whichever feature set the risk model won on.
        diagnosis_cols = feature_columns(self.data.df, "engineered")
        self.ctx.say("  [engine] training diagnosis models "
                     "(failure type, time-to-failure)")
        bundle.type_classifier = train_failure_type_classifier(
            self.data.df, diagnosis_cols)
        bundle.ttf_regressor = train_time_to_failure_regressor(
            self.data.df, diagnosis_cols)

        bundle.save(BUNDLE_DIR)
        self.service = ModelService(bundle)
        self.library = SignatureLibrary.build(
            self.data.df, self.data.failures, bundle.channels, bundle.lookback)
        return bundle

    def _score_split(self, bundle: ModelBundle, runner: ExperimentRunner,
                     split: str) -> dict:
        from ..ml.training import predict_windows
        _, _, _, va, te, _, _ = runner.sequence_bits(bundle.feature_set)
        wi = va if split == "val" else te
        if bundle.is_sequence():
            p = predict_windows(bundle.torch_model, wi)
            return {"y": wi.labels, "p": p}
        x, y, m, t, _ = self.data.tabular(split, bundle.feature_set)
        return {"y": y, "p": bundle.score_rows(x)}

    def load_bundle(self, path: Path = BUNDLE_DIR) -> ModelBundle:
        self.bundle = ModelBundle.load(path)
        self.service = ModelService(self.bundle)
        self.library = SignatureLibrary.build(
            self.data.df, self.data.failures, self.bundle.channels,
            self.bundle.lookback)
        return self.bundle

    # -- incident half ------------------------------------------------------
    def run_incident(self, machine_id: str, timestamp, save: bool = True,
                     horizon_hours: float | None = None) -> IncidentReport:
        """One machine, one moment, through the full agent workflow.

        The order matters. Facts are established once (investigator), both
        readings of them are argued independently, and only then does anything
        decide. Running remediation before adjudication would mean planning a
        repair for a fault that had not yet survived challenge.
        """
        if self.service is None:
            self.load_bundle()
        t0 = time.time()
        start_step = self.ctx.step

        # 1. what is likely to happen  (model service, not a model agent)
        pred = PredictionAgent().execute(
            self.ctx, service=self.service, machine_id=machine_id,
            timestamp=timestamp)

        # 2. what do we already know about this machine
        con = ContextAgent().execute(
            self.ctx, machine_id=machine_id, timestamp=timestamp)

        # 3. what actually changed  (the shared factual record)
        inv = InvestigationAgent().execute(
            self.ctx, service=self.service, machine_id=machine_id,
            timestamp=timestamp, library=self.library)
        builder = self.ctx.state["evidence_builder"]

        # 4. two readings of the same facts, argued independently
        deg = DegradationAdvocate().execute(
            self.ctx, builder=builder, prediction=pred.output,
            context=con.output,
            neighbours=inv.output.get("similar_past_failures", []))
        conf = ConfoundAdvocate().execute(
            self.ctx, builder=builder, context=con.output,
            prediction=pred.output)

        # 5. decide between them, on the numbers
        adj = Adjudicator().execute(
            self.ctx, degradation=deg.output, confound=conf.output,
            prediction=pred.output)

        # the evidence record now includes whatever the advocates went looking
        # for, so downstream agents and the verifier see the complete set
        investigation = dict(inv.output)
        investigation["evidence"] = builder.items
        investigation["ranked_hypotheses"] = deg.output.get("ranked_types", [])
        investigation["likely_failure_type"] = adj.output.get("failure_type")
        investigation["conclusion"] = adj.output.get("rationale", "")
        investigation["narrative"] = " ".join(
            e["claim"] + "." for e in builder.items)

        # 6. what can we do -- only if the case survived
        rem = RemediationAgent().execute(
            self.ctx, prediction=pred.output, investigation=investigation,
            adjudication=adj.output, context=con.output)

        # 7. what would that buy
        sim = SimulationAgent().execute(
            self.ctx, bundle=self.bundle, machine_id=machine_id,
            timestamp=timestamp, plan=rem.output["plan"],
            baseline_probability=pred.output["failure_probability"],
            **({"horizon_hours": horizon_hours} if horizon_hours else {}))

        # 8. does any of this hold up
        ver = VerificationAgent().execute(
            self.ctx, bundle=self.bundle, prediction=pred.output,
            investigation=investigation, remediation=rem.output,
            simulation=sim.output, adjudication=adj.output)

        traj = [r for r in self.store.trajectory(self.run_id)
                if r["step"] > start_step]
        report = IncidentReport(
            machine_id=machine_id, timestamp=str(timestamp),
            prediction=pred.output, context=con.output,
            investigation=investigation,
            degradation_case=deg.output, confound_case=conf.output,
            adjudication=adj.output,
            remediation=rem.output, simulation=sim.output,
            verification=ver.output, trajectory=traj,
            llm_usage=self.ctx.usage.to_dict(),
            duration_s=time.time() - t0, run_id=self.run_id)
        if save:
            report.save()
        return report

    # -- helpers -------------------------------------------------------------
    def busiest_timestamp(self, split: str = "test"):
        """The moment with the most machines at risk.

        The fleet view used to open on the last sample of the test period,
        where nothing happens to be failing -- a wall of "normal" that tells a
        reader nothing about what the system does. Opening on the busiest
        moment is not cherry-picking: every timestamp is scoreable and the
        control lets you move to any of them.
        """
        if self._busiest is not None:
            return self._busiest
        if self.service is None:
            self.load_bundle()
        x, y, m, t, _ = self.data.tabular(split, self.bundle.feature_set)
        if self.bundle.is_sequence():
            self._busiest = pd.Timestamp(pd.Series(t).max())
            return self._busiest
        p = self.bundle.score_rows(x)
        f = pd.DataFrame({"timestamp": pd.to_datetime(pd.Series(t)),
                          "hit": p >= self.service.alert_threshold,
                          "watch": p >= self.service.investigate_threshold})
        by = f.groupby("timestamp").agg(hit=("hit", "sum"),
                                        watch=("watch", "sum"))
        by = by.sort_values(["hit", "watch"], ascending=False)
        self._busiest = pd.Timestamp(by.index[0]) if len(by) else None
        return self._busiest

    def fleet_scores(self, at: pd.Timestamp | None = None,
                     split: str = "test") -> pd.DataFrame:
        """Score every machine at one timestamp -- powers the fleet view."""
        if self.service is None:
            self.load_bundle()
        from ..ml.bundle import window_for
        from ..ml.training import static_index

        if at is not None:
            at = pd.Timestamp(at)
        else:
            at = self.busiest_timestamp(split)
            if at is None:
                at = self.data.df[self.data.df["split"] == split]["timestamp"].max()
        rows = []
        for mid, g in self.data.df.groupby("machine_id"):
            try:
                x, raw = window_for(self.data.df, mid, at, self.bundle.channels,
                                    self.bundle.scaler, self.bundle.lookback)
            except ValueError:
                continue
            # Scorability must use the same rule the deployed model was
            # trained and evaluated under, which differs by model kind:
            # a sequence model consumes the whole lookback, so no step in it
            # may be downtime (`preprocessing.build_windows`); a tabular model
            # consumes one row, so only that row must be usable
            # (`preprocessing.usable_mask`). `fleet.overview` already splits
            # this way. Applying only the final-step check here scored a
            # sequence model on a lookback straddling an outage -- an input
            # distribution it never saw -- which is how a pump 40 minutes back
            # from a stop reached the assistant as the plant's highest risk at
            # 69.5% while the dashboard showed the same pump as `down`.
            unusable = (int(raw["is_downtime"].max()) == 1
                        if self.bundle.is_sequence()
                        else int(raw["is_downtime"].iloc[-1]) == 1
                        or int(raw.get("sensor_dropout",
                                       pd.Series([0])).iloc[-1]) == 1)
            if unusable:
                rows.append({"machine_id": mid, "failure_probability": np.nan,
                             "status": "down"})
                continue
            if self.bundle.is_sequence():
                p = float(self.bundle.score_windows(
                    x[None, ...], np.array([static_index(mid)]))[0])
            else:
                r = self.data.df[(self.data.df.machine_id == mid)
                                 & (self.data.df.timestamp == at)]
                p = float(self.bundle.score_rows(
                    r[self.bundle.tabular_columns].to_numpy(
                        dtype=np.float32))[0])
            rows.append({
                "machine_id": mid, "failure_probability": round(p, 4),
                "confidence": round(self.bundle.confidence(p), 4),
                "alert": bool(p >= self.service.alert_threshold),
                "status": ("high" if p >= self.service.alert_threshold
                           else "watch"
                           if p >= self.service.investigate_threshold
                           else "normal"),
                "machine_type": mid.split("-")[0],
                "vibration": round(float(raw["vibration"].iloc[-1]), 3),
                "temperature": round(float(raw["temperature"].iloc[-1]), 2),
                "load": round(float(raw["load"].iloc[-1]), 3),
            })
        out = pd.DataFrame(rows)
        out.attrs["timestamp"] = str(at)
        return out.sort_values("failure_probability", ascending=False,
                               na_position="last")
