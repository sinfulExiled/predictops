"""Agent contracts, the hypothesis contest, and verification.

The load-bearing test in this file is
`test_verifier_catches_fabricated_evidence`: it plants a false number in the
evidence and asserts the run is marked FAIL. If that ever goes green for the
wrong reason, the project's central claim -- that agents cannot invent
evidence here -- is unsupported.

Close behind it is `test_advocates_cannot_disagree_about_the_facts`: the two
advocates are allowed to disagree about meaning, never about measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictops.agents.base import AgentContext
from predictops.agents.context import ContextAgent
from predictops.agents.evidence import (
    EvidenceBuilder,
    RECOMPUTE_FNS,
    monotonicity,
    pct_change,
    peak_ratio,
)
from predictops.agents.hypothesis import (
    Adjudicator,
    ConfoundAdvocate,
    DegradationAdvocate,
)
from predictops.agents.investigator import InvestigationAgent, SignatureLibrary
from predictops.agents.predictor import PredictionAgent
from predictops.agents.remediation import RemediationAgent
from predictops.agents.simulator import SimulationAgent
from predictops.agents.verifier import VerificationAgent
from predictops.data.preprocessing import Scaler
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import MockProvider
from predictops.ml.bundle import ModelBundle
from predictops.ml.dataset import prepare
from predictops.ml.diagnosis import (
    ReliabilityCurve,
    train_failure_type_classifier,
    train_time_to_failure_regressor,
)
from predictops.ml.evaluation import pick_threshold
from predictops.ml.features import feature_columns, sequence_channels
from predictops.ml.service import ModelService
from predictops.ml.trees import train_tree
from predictops.simulation.interventions import CATALOGUE, get


@pytest.fixture(scope="module")
def data():
    return prepare()


@pytest.fixture(scope="module")
def service(data):
    """A small but genuine model service: a real fitted tree plus diagnosis."""
    cols = feature_columns(data.df, "engineered")
    x_tr, y_tr, *_ = data.tabular("train", "engineered")
    x_va, y_va, *_ = data.tabular("val", "engineered")
    tree = train_tree("xgboost", x_tr, y_tr, x_va, y_va, cols, n_rounds=60)
    p_va = tree.predict_proba(x_va)

    chans = [c for c in sequence_channels("engineered") if c in data.df.columns]
    b = ModelBundle(
        kind="xgboost", feature_set="engineered", channels=chans,
        tabular_columns=cols, threshold=pick_threshold(y_va, p_va),
        scaler=Scaler.fit(data.split("train"), chans),
        tree_model=tree, reliability=ReliabilityCurve.fit(y_va, p_va))
    b.type_classifier = train_failure_type_classifier(data.df, cols)
    b.ttf_regressor = train_time_to_failure_regressor(data.df, cols)
    return ModelService(b)


@pytest.fixture(scope="module")
def library(data, service):
    return SignatureLibrary.build(data.df, data.failures,
                                  service.bundle.channels,
                                  service.bundle.lookback)


@pytest.fixture(scope="module")
def case(data):
    """A test-split row inside a real warning window."""
    d = data.df[(data.df.split == "test") & (data.df.label == 1)
                & (data.df.is_downtime == 0) & (data.df.sensor_dropout == 0)]
    assert len(d), "no positive test rows to investigate"
    r = d.sort_values("time_to_failure_h").iloc[len(d) // 2]
    return str(r.machine_id), r.timestamp


@pytest.fixture
def ctx(data):
    return AgentContext(run_id="pytest", data=data, store=ExperimentStore(),
                        provider=MockProvider(), verbose=False)


def _run_to_adjudication(ctx, service, library, mid, ts):
    """Prediction -> context -> facts -> both advocates -> adjudication."""
    pred = PredictionAgent().execute(
        ctx, service=service, machine_id=mid, timestamp=ts).output
    con = ContextAgent().execute(
        ctx, machine_id=mid, timestamp=ts).output
    inv = InvestigationAgent().execute(
        ctx, service=service, machine_id=mid, timestamp=ts,
        library=library).output
    builder = ctx.state["evidence_builder"]
    deg = DegradationAdvocate().execute(
        ctx, builder=builder, prediction=pred, context=con,
        neighbours=inv.get("similar_past_failures", [])).output
    conf = ConfoundAdvocate().execute(
        ctx, builder=builder, context=con, prediction=pred).output
    adj = Adjudicator().execute(
        ctx, degradation=deg, confound=conf, prediction=pred).output
    inv = dict(inv)
    inv["evidence"] = builder.items
    inv["ranked_hypotheses"] = deg.get("ranked_types", [])
    inv["likely_failure_type"] = adj.get("failure_type")
    inv["narrative"] = " ".join(e["claim"] + "." for e in builder.items)
    return pred, con, inv, deg, conf, adj


def _full_case(ctx, service, library, mid, ts):
    pred, con, inv, deg, conf, adj = _run_to_adjudication(
        ctx, service, library, mid, ts)
    rem = RemediationAgent().execute(
        ctx, prediction=pred, investigation=inv, adjudication=adj,
        context=con).output
    sim = SimulationAgent().execute(
        ctx, bundle=service.bundle, machine_id=mid, timestamp=ts,
        plan=rem["plan"], baseline_probability=pred["failure_probability"]).output
    return pred, con, inv, deg, conf, adj, rem, sim


# --- the model service boundary --------------------------------------------
def test_service_exposes_two_thresholds(service):
    assert service.investigate_threshold < service.alert_threshold
    d = service.describe()
    assert d["kind"] and d["lookback_steps"] > 0


def test_prediction_output_schema(ctx, service, case):
    mid, ts = case
    out = PredictionAgent().execute(ctx, service=service, machine_id=mid,
                                    timestamp=ts).output
    for k in ("machine_id", "failure_probability", "alert", "investigate",
              "threshold", "confidence", "prediction_window_hours",
              "failure_type"):
        assert k in out, k
    assert 0.0 <= out["failure_probability"] <= 1.0
    assert out["alert"] == (out["failure_probability"] >= out["alert_threshold"])
    assert out["investigate"] == (
        out["failure_probability"] >= out["investigate_threshold"])


def test_confidence_comes_from_the_reliability_curve(service):
    assert service.bundle.confidence(0.99) >= service.bundle.confidence(0.01)
    assert service.bundle.reliability.edges, "no reliability curve was fitted"


def test_prediction_refuses_a_window_without_enough_history(ctx, service, data):
    mid = data.df.machine_id.iloc[0]
    first = data.df[data.df.machine_id == mid].timestamp.min()
    res = PredictionAgent().execute(ctx, service=service, machine_id=mid,
                                    timestamp=first)
    assert res.output.get("status") == "failed"
    assert "required" in res.output.get("error", "")


# --- context ----------------------------------------------------------------
def test_context_dossier_uses_only_the_past(ctx, data, case):
    mid, ts = case
    out = ContextAgent().execute(ctx, machine_id=mid, timestamp=ts).output
    for f in out["prior_failures"]:
        assert pd.Timestamp(f["at"]) <= pd.Timestamp(ts), \
            "context leaked a future failure into the dossier"
    if out["recent_service"]:
        assert pd.Timestamp(out["recent_service"]["at"]) <= pd.Timestamp(ts)
    assert out["machine_type"] in ("PUMP", "MOTOR", "COMPRESSOR", "CONVEYOR")


# --- shared evidence --------------------------------------------------------
def test_every_evidence_item_is_recomputable(ctx, service, library, case):
    mid, ts = case
    _, _, inv, _, _, _ = _run_to_adjudication(ctx, service, library, mid, ts)
    assert inv["evidence"], "no evidence at all"
    for e in inv["evidence"]:
        assert e["recompute"]["fn"] in RECOMPUTE_FNS
        assert "source" in e and "value" in e


def test_advocates_cannot_disagree_about_the_facts(ctx, service, library, case):
    """Both advocates extend one record; ids must stay unique and consistent."""
    mid, ts = case
    _, _, inv, deg, conf, _ = _run_to_adjudication(
        ctx, service, library, mid, ts)
    ids = [e["id"] for e in inv["evidence"]]
    assert len(ids) == len(set(ids)), "duplicate evidence ids"
    # each advocate cites only ids that exist in the shared record
    for cited in (deg["evidence"], conf["evidence"]):
        assert set(cited) <= set(ids)


def test_evidence_primitives_behave(data):
    """monotonicity separates a trend from an excursion; peak_ratio a spike."""
    ts = pd.date_range("2025-01-01", periods=36, freq="10min")
    trend = pd.DataFrame({"machine_id": "X", "timestamp": ts,
                          "v": np.linspace(1.0, 2.0, 36)})
    bump = pd.DataFrame({"machine_id": "X", "timestamp": ts,
                         "v": np.concatenate([np.linspace(1, 2, 18),
                                              np.linspace(2, 1, 18)])})
    # Use the full 6 h window: a 3 h window would only see the bump's
    # descending half, which is itself monotonic.
    assert monotonicity(trend, "v", 6.0) == pytest.approx(1.0)
    assert monotonicity(bump, "v", 6.0) < 0.8

    spike = trend.copy()
    spike.loc[spike.index[20], "v"] = 20.0
    assert peak_ratio(spike, "v", 3.0) > peak_ratio(trend, "v", 3.0)


# --- the hypothesis contest -------------------------------------------------
def test_both_advocates_produce_a_scored_case(ctx, service, library, case):
    mid, ts = case
    _, _, _, deg, conf, _ = _run_to_adjudication(ctx, service, library, mid, ts)
    for c in (deg, conf):
        assert 0.0 <= c["score"] <= 1.0
        assert c["conclusion"]
        assert c["would_change_my_mind"], "an advocate must state its refuter"


def test_adjudication_is_arithmetic(ctx, service, library, case):
    mid, ts = case
    _, _, _, deg, conf, adj = _run_to_adjudication(
        ctx, service, library, mid, ts)
    assert adj["margin"] == pytest.approx(
        adj["degradation_score"] - adj["confound_score"], abs=1e-4)
    assert adj["decision"] in ("alert", "contested", "overturned",
                               "insufficient_evidence", "no_alert")


def test_a_strong_confound_overturns_a_flagged_case(ctx):
    """The adjudicator must be willing to overrule the model."""
    adj = Adjudicator().execute(
        ctx,
        degradation={"score": 0.40, "failure_type": "bearing_degradation"},
        confound={"score": 0.80,
                  "alternative_explanations": [
                      {"explanation": "production load increase"}]},
        prediction={"alert": True, "investigate": True,
                    "failure_probability": 0.7}).output
    assert adj["decision"] == "overturned"
    assert adj["alert"] is False
    assert adj["changed_the_model_verdict"] is True
    assert adj["leading_benign_explanation"] == "production load increase"


def test_a_close_contest_is_reported_as_contested(ctx):
    adj = Adjudicator().execute(
        ctx, degradation={"score": 0.62, "failure_type": "bearing_degradation"},
        confound={"score": 0.58, "alternative_explanations": []},
        prediction={"alert": True, "investigate": True,
                    "failure_probability": 0.6}).output
    assert adj["decision"] == "contested"
    assert adj["alert"] is True
    assert adj["recommend_physical_work"] is False, \
        "a contested case must not authorise physical work"


# --- remediation ------------------------------------------------------------
def test_remediation_only_proposes_catalogue_actions(ctx, service, library,
                                                     case):
    mid, ts = case
    *_, rem, _ = _full_case(ctx, service, library, mid, ts)
    assert rem["plan"]
    for step in rem["plan"]:
        assert step["intervention_id"] in CATALOGUE


def test_an_overturned_case_proposes_no_physical_work(ctx, service, library,
                                                      case):
    """The whole point of the contest: a rejected case cannot send a crew."""
    mid, ts = case
    pred, con, inv, _, _, _ = _run_to_adjudication(
        ctx, service, library, mid, ts)
    overturned = {"decision": "overturned", "margin": -0.4,
                  "leading_benign_explanation": "production load increase",
                  "rationale": "benign case stronger"}
    rem = RemediationAgent().execute(
        ctx, prediction=pred, investigation=inv, adjudication=overturned,
        context=con).output
    assert rem["mode"] == "monitor_only"
    assert [p["intervention_id"] for p in rem["plan"]] == ["increase_monitoring"]
    assert all(p["is_diagnostic"] for p in rem["plan"])
    assert rem["estimated_downtime_hours"] == 0


def test_catalogue_rejects_an_invented_action():
    with pytest.raises(KeyError, match="approved intervention catalogue"):
        get("vent_the_reactor")


def test_high_risk_actions_are_all_gated():
    for iv in CATALOGUE.values():
        if iv.risk == "high" or iv.downtime_hours > 0:
            assert iv.requires_approval, f"{iv.id} is not gated"


def test_consequential_actions_carry_an_approval_gate(ctx, service, library,
                                                      case):
    mid, ts = case
    *_, rem, _ = _full_case(ctx, service, library, mid, ts)
    if [p for p in rem["plan"] if p["requires_approval"]]:
        assert rem["approval_gate"]["required"] is True
        assert rem["approval_gate"]["status"] == "awaiting_human_approval"


# --- simulation -------------------------------------------------------------
def test_simulation_has_a_control_arm_and_labels_everything(ctx, service,
                                                            library, case):
    mid, ts = case
    *_, sim = _full_case(ctx, service, library, mid, ts)
    assert "no_action" in sim, "no control arm -- deltas would be unreadable"
    assert sim["no_action"]["is_simulated"] is True
    for arm in sim["arms"]:
        assert arm["is_simulated"] is True
        if arm.get("simulated"):
            assert 0.0 <= arm["failure_probability_simulated"] <= 1.0
    assert "caveat" in sim


def test_load_reduction_does_not_increase_projected_load(data):
    from predictops.simulation.machine_environment import rollout
    mid = str(data.df.machine_id.iloc[0])
    g = data.df[data.df.machine_id == mid].sort_values("timestamp").tail(72)
    base = rollout(g, 3.0, None).frame.tail(18)["load"].mean()
    cut = rollout(g, 3.0, get("reduce_load_70")).frame.tail(18)["load"].mean()
    assert cut < base, "reduce_load_70 did not reduce load in the rollout"


# --- verification -----------------------------------------------------------
def _verify(ctx, service, pred, inv, rem, sim, adj):
    return VerificationAgent().execute(
        ctx, bundle=service.bundle, prediction=pred, investigation=inv,
        remediation=rem, simulation=sim, adjudication=adj).output


def test_verifier_passes_a_clean_case(ctx, service, library, case):
    mid, ts = case
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    ver = _verify(ctx, service, pred, inv, rem, sim, adj)
    assert ver["verdict"] in ("PASS", "PASS_WITH_WARNINGS"), ver["checks"]
    c1 = next(c for c in ver["checks"] if c["id"] == "C1")
    assert c1["status"] == "pass", c1


def test_verifier_catches_fabricated_evidence(ctx, service, library, case):
    """Plant a false number. The verifier must refuse to certify the result."""
    mid, ts = case
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    assert inv["evidence"], "need at least one evidence item to tamper with"

    tampered = dict(inv)
    tampered["evidence"] = [dict(e) for e in inv["evidence"]]
    tampered["evidence"][0]["value"] = float(
        tampered["evidence"][0]["value"]) + 999.0
    tampered["evidence"][0]["claim"] = "Vibration increased 999% over 3 hours"

    ver = _verify(ctx, service, pred, tampered, rem, sim, adj)
    c1 = next(c for c in ver["checks"] if c["id"] == "C1")
    assert c1["status"] == "fail", "verifier accepted a fabricated value"
    assert c1["mismatches"]
    assert ver["verdict"] == "FAIL"
    assert ver["safe_to_act"] is False


def test_verifier_catches_an_unsupported_number_in_the_narrative(
        ctx, service, library, case):
    mid, ts = case
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    tampered = dict(inv)
    tampered["narrative"] = ("Bearing wear is advanced; vibration is up 431% "
                             "and the unit has run 8123 hours since service.")
    ver = _verify(ctx, service, pred, tampered, rem, sim, adj)
    c8 = next(c for c in ver["checks"] if c["id"] == "C8")
    assert c8["status"] == "fail", "unsupported narrative numbers went unnoticed"


def test_verifier_flags_a_missing_approval_gate(ctx, service, library, case):
    mid, ts = case
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    if not any(p["requires_approval"] for p in rem["plan"]):
        pytest.skip("this plan proposes nothing that needs approval")
    broken = dict(rem)
    broken["approval_gate"] = {"required": False, "actions": [],
                               "status": "none"}
    ver = _verify(ctx, service, pred, inv, broken, sim, adj)
    c9 = next(c for c in ver["checks"] if c["id"] == "C9")
    assert c9["status"] == "fail"


def test_verifier_reports_the_hypothesis_contest(ctx, service, library, case):
    mid, ts = case
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    ver = _verify(ctx, service, pred, inv, rem, sim, adj)
    c10 = next(c for c in ver["checks"] if c["id"] == "C10")
    assert c10["decision"] == adj["decision"]
    assert c10["degradation_score"] == adj["degradation_score"]


# --- trajectories -----------------------------------------------------------
def test_every_agent_step_is_logged_to_the_trajectory(ctx, service, library,
                                                      case):
    mid, ts = case
    before = len(ctx.store.trajectory("pytest"))
    pred, con, inv, deg, conf, adj, rem, sim = _full_case(
        ctx, service, library, mid, ts)
    _verify(ctx, service, pred, inv, rem, sim, adj)
    after = ctx.store.trajectory("pytest")
    logged = [r["agent"] for r in after[before:]]
    assert logged == ["predictor", "context", "investigator",
                      "degradation_advocate", "confound_advocate",
                      "adjudicator", "remediation", "simulator", "verifier"]
    for row in after[before:]:
        for field in ("agent", "action", "tools_used", "output", "duration_s"):
            assert field in row
