"""End-to-end: dataset -> features -> model -> incident report.

This is the test that would catch an integration break that every unit test
misses: a scaler applied in the wrong order, a bundle that round-trips badly,
an agent whose output the next agent cannot read.
"""
from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
import pytest

from predictops.agents.orchestrator import PredictOpsEngine
from predictops.config import GeneratorConfig
from predictops.data.generator import PlantGenerator
from predictops.data.preprocessing import Scaler
from predictops.evaluation.scenarios import build_suite, summarise
from predictops.experiments.registry import Experiment, ExperimentStore
from predictops.llm.provider import MockProvider, get_provider
from predictops.ml.bundle import ModelBundle
from predictops.ml.dataset import prepare
from predictops.ml.diagnosis import (
    ReliabilityCurve,
    train_failure_type_classifier,
    train_time_to_failure_regressor,
)
from predictops.ml.evaluation import pick_threshold
from predictops.ml.features import feature_columns, sequence_channels
from predictops.ml.trees import train_tree
from predictops.reporting import render_incident


@pytest.fixture(scope="module")
def data():
    return prepare()


@pytest.fixture(scope="module")
def engine(data, tmp_path_factory):
    """A real, small, end-to-end deployable engine."""
    cols = feature_columns(data.df, "engineered")
    x_tr, y_tr, *_ = data.tabular("train", "engineered")
    x_va, y_va, *_ = data.tabular("val", "engineered")
    tree = train_tree("xgboost", x_tr, y_tr, x_va, y_va, cols, n_rounds=60)
    p_va = tree.predict_proba(x_va)

    chans = [c for c in sequence_channels("engineered") if c in data.df.columns]
    bundle = ModelBundle(
        kind="xgboost", feature_set="engineered", channels=chans,
        tabular_columns=cols, threshold=pick_threshold(y_va, p_va),
        scaler=Scaler.fit(data.split("train"), chans), tree_model=tree,
        reliability=ReliabilityCurve.fit(y_va, p_va),
        selection_rationale="fixture model for the end-to-end test")
    bundle.type_classifier = train_failure_type_classifier(data.df, cols)
    bundle.ttf_regressor = train_time_to_failure_regressor(data.df, cols)
    bundle.save(tmp_path_factory.mktemp("bundle"))

    # The registry is persistent SQLite, so a fixed run id would accumulate
    # trajectory rows across pytest sessions and break the per-run assertions.
    eng = PredictOpsEngine(data=data, store=ExperimentStore(),
                           provider=MockProvider(),
                           run_id=f"pytest-e2e-{uuid.uuid4().hex[:8]}",
                           verbose=False)
    eng.bundle = bundle
    # Wire the service explicitly. Without this `run_incident` falls back to
    # loading a bundle from disk, so the test passed only on a machine that had
    # already run the training pipeline.
    from predictops.ml.service import ModelService
    eng.service = ModelService(bundle)
    from predictops.agents.investigator import SignatureLibrary
    eng.library = SignatureLibrary.build(data.df, data.failures,
                                         bundle.channels, bundle.lookback)
    return eng


def _a_warning_window(data):
    d = data.df[(data.df.split == "test") & (data.df.label == 1)
                & (data.df.is_downtime == 0) & (data.df.sensor_dropout == 0)]
    r = d.sort_values("time_to_failure_h").iloc[len(d) // 2]
    return str(r.machine_id), r.timestamp


# --- the full loop ---------------------------------------------------------
def test_end_to_end_incident_produces_a_complete_report(engine, data):
    mid, ts = _a_warning_window(data)
    report = engine.run_incident(mid, ts, save=False)

    for section in ("prediction", "context", "investigation",
                    "degradation_case", "confound_case", "adjudication",
                    "remediation", "simulation", "verification"):
        assert getattr(report, section), f"{section} is empty"

    assert report.prediction["machine_id"] == mid
    assert 0.0 <= report.prediction["failure_probability"] <= 1.0
    assert report.investigation["evidence"]
    assert report.remediation["plan"]
    assert "no_action" in report.simulation
    assert report.adjudication["decision"] in (
        "alert", "contested", "overturned", "insufficient_evidence",
        "no_alert")
    assert report.verification["verdict"] in (
        "PASS", "PASS_WITH_WARNINGS", "FAIL")
    assert len(report.trajectory) == 9, "one trajectory row per agent expected"
    assert [t["agent"] for t in report.trajectory] == [
        "predictor", "context", "investigator", "degradation_advocate",
        "confound_advocate", "adjudicator", "remediation", "simulator",
        "verifier"]


def test_report_is_json_serialisable(engine, data):
    mid, ts = _a_warning_window(data)
    report = engine.run_incident(mid, ts, save=False)
    blob = json.dumps(report.to_dict(), default=str)
    assert len(blob) > 2000
    assert json.loads(blob)["machine_id"] == mid


def test_rendered_report_answers_all_six_questions(engine, data):
    mid, ts = _a_warning_window(data)
    text = render_incident(engine.run_incident(mid, ts, save=False).to_dict())
    for heading in ("PREDICTOPS INCIDENT REPORT", "FAILURE PROBABILITY",
                    "EXPECTED WINDOW", "WHY", "RECOMMENDED ACTION",
                    "IF WE ACT", "VERIFICATION"):
        assert heading in text, f"report is missing '{heading}'"
    assert "simulated" in text.lower()
    assert mid in text


def test_fleet_scoring_covers_every_machine(engine, data):
    fleet = engine.fleet_scores()
    assert len(fleet) == data.df["machine_id"].nunique()
    live = fleet.dropna(subset=["failure_probability"])
    assert live["failure_probability"].between(0, 1).all()
    assert set(live["status"]) <= {"high", "watch", "normal"}


# --- bundle persistence ----------------------------------------------------
def test_bundle_round_trips(engine, tmp_path, data):
    path = engine.bundle.save(tmp_path / "b")
    loaded = ModelBundle.load(path)
    assert loaded.kind == engine.bundle.kind
    assert loaded.channels == engine.bundle.channels
    assert loaded.threshold == pytest.approx(engine.bundle.threshold)

    x, _, *_ = data.tabular("test", "engineered")
    a = engine.bundle.score_rows(x[:200])
    b = loaded.score_rows(x[:200])
    assert np.allclose(a, b), "reloaded bundle scores differently"


# --- registry integrity ----------------------------------------------------
def test_registry_refuses_an_experiment_with_no_measured_metrics():
    store = ExperimentStore()
    with pytest.raises(ValueError, match="no measured metrics"):
        store.record(Experiment(run_id="pytest", stage="Bogus", name="made up",
                                model="vibes", feature_set="none"))


def test_trajectory_rows_carry_the_required_fields(engine, data):
    mid, ts = _a_warning_window(data)
    engine.run_incident(mid, ts, save=False)
    rows = engine.store.trajectory(engine.run_id)
    assert rows
    for r in rows:
        for field in ("agent", "step", "action", "reason", "tools_used",
                      "input_summary", "output", "retry_count", "duration_s"):
            assert field in r, field
        assert isinstance(r["tools_used"], list)


# --- provider abstraction --------------------------------------------------
def test_pipeline_runs_without_any_api_key(engine, data, monkeypatch):
    """The whole workflow must work with no LLM available."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = get_provider("auto")
    assert provider.name == "mock"

    engine.ctx.provider = provider
    mid, ts = _a_warning_window(data)
    report = engine.run_incident(mid, ts, save=False)
    assert report.verification["verdict"] in (
        "PASS", "PASS_WITH_WARNINGS", "FAIL")


def test_mock_provider_returns_the_deterministic_fallback():
    p = MockProvider()
    fallback = {"headline": "computed", "risks": ["a"], "recommendations": []}
    out = p.structured("sys", "prompt", {}, fallback)
    assert out.data == fallback
    assert out.used_fallback is True
    assert out.cost_usd == 0.0


# --- scenario suite --------------------------------------------------------
def test_scenario_suite_is_deterministic_and_hard(data):
    a = build_suite(data.df, data.failures)
    b = build_suite(data.df, data.failures)
    assert [s.id for s in a] == [s.id for s in b]
    assert [s.machine_id for s in a] == [s.machine_id for s in b]

    s = summarise(a)
    assert s["n_cases"] >= 20, "suite is too small to be informative"
    assert s["n_positive"] > 0 and s["n_negative"] > 0
    assert s["by_difficulty"].get("hard", 0) >= 8, \
        "not enough hard cases to separate the systems"


def test_every_scenario_is_actually_scoreable(engine, data):
    from predictops.ml.bundle import window_for
    for sc in build_suite(data.df, data.failures)[:12]:
        x, raw = window_for(data.df, sc.machine_id, pd.Timestamp(sc.timestamp),
                            engine.bundle.channels, engine.bundle.scaler,
                            engine.bundle.lookback)
        assert x.shape == (engine.bundle.lookback, len(engine.bundle.channels))
        assert np.isfinite(x).all()
