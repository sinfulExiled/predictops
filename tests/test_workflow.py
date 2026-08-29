"""The composable workflow graph.

A canvas that lets you wire agents arbitrarily is only honest if the rejections
are real. These tests hold the line: validation is derived from the agents'
declared contracts, a rejected edge names the reason, and an executed graph
actually runs the agents it says it will.
"""
from __future__ import annotations

import pytest

from predictops.agents.orchestrator import PredictOpsEngine
from predictops.agents.workflow import (
    DEFAULT_EDGES,
    NODES,
    default_graph,
    describe_nodes,
    execute,
    validate,
    validate_edge,
)
from predictops.experiments.registry import ExperimentStore
from predictops.llm.provider import MockProvider
from predictops.ml.dataset import prepare


@pytest.fixture(scope="module")
def engine():
    e = PredictOpsEngine(data=prepare(), store=ExperimentStore(),
                         provider=MockProvider(), run_id="pytest-workflow",
                         verbose=False)
    try:
        e.load_bundle()
    except FileNotFoundError:
        pytest.skip("no model bundle; run run_experiments.py first")
    return e


# --- the graph model --------------------------------------------------------
def test_default_graph_is_valid_and_ordered():
    g = default_graph()
    v = validate(g["nodes"], [tuple(e) for e in g["edges"]])
    assert v["valid"], v["errors"]
    assert v["order"][0] == "predictor"
    assert v["order"][-1] == "verifier"
    assert set(v["order"]) == set(NODES)


def test_default_graph_matches_the_orchestrator_wiring():
    """If the two drift apart, the canvas is editing a fiction."""
    g = default_graph()
    assert set(map(tuple, g["edges"])) == set(DEFAULT_EDGES)
    assert set(g["nodes"]) == set(NODES)


def test_every_default_edge_is_individually_valid():
    for a, b in DEFAULT_EDGES:
        ok, why = validate_edge(a, b)
        assert ok, f"{a} -> {b}: {why}"


def test_edge_rejection_names_a_real_reason():
    ok, why = validate_edge("simulator", "confound_advocate")
    assert not ok
    assert "does not read" in why
    assert "simulation" in why      # it names what the source actually produces


def test_edges_are_rejected_by_contract_not_by_a_hardcoded_list():
    """Any edge whose payload is consumed downstream must be allowed."""
    for src, sspec in NODES.items():
        for dst, dspec in NODES.items():
            if src == dst:
                continue
            shared = set(sspec.provides) & set(dspec.consumes())
            ok, _ = validate_edge(src, dst)
            assert ok == bool(shared), f"{src} -> {dst}"


def test_self_loop_is_rejected():
    ok, why = validate_edge("predictor", "predictor")
    assert not ok and "itself" in why


def test_cycles_are_detected():
    g = default_graph()
    edges = [tuple(e) for e in g["edges"]] + [("verifier", "context")]
    v = validate(g["nodes"], edges)
    assert not v["valid"]
    assert any("cycle" in e for e in v["errors"])


def test_required_nodes_cannot_be_dropped():
    g = default_graph()
    for required in ("predictor", "investigator", "degradation_advocate",
                     "remediation"):
        nodes = [n for n in g["nodes"] if n != required]
        edges = [tuple(e) for e in g["edges"] if required not in e]
        v = validate(nodes, edges)
        assert not v["valid"], f"{required} should not be removable"
        assert any("cannot be removed" in e for e in v["errors"])


def test_optional_nodes_can_be_dropped():
    g = default_graph()
    for optional in ("context", "confound_advocate", "simulator", "verifier"):
        nodes = [n for n in g["nodes"] if n != optional]
        edges = [tuple(e) for e in g["edges"] if optional not in e]
        v = validate(nodes, edges)
        assert v["valid"], f"{optional}: {v['errors']}"
        assert any(optional.replace("_", " ") in w.lower()
                   or NODES[optional].label.lower() in w.lower()
                   for w in v["warnings"])


def test_unsatisfied_requirement_is_rejected():
    """An adjudicator with no advocate upstream cannot run."""
    v = validate(["predictor", "investigator", "degradation_advocate",
                  "remediation", "adjudicator"],
                 [("predictor", "adjudicator")])
    assert not v["valid"]
    assert any("needs" in e for e in v["errors"])


def test_node_descriptions_are_complete():
    for n in describe_nodes():
        assert n["label"] and n["question"] and n["note"]
        assert n["provides"], f"{n['name']} produces nothing"


# --- execution --------------------------------------------------------------
def test_executing_the_default_graph_runs_every_agent(engine):
    g = default_graph()
    out = execute(engine, "MOTOR-045", engine.busiest_timestamp(),
                  g["nodes"], [tuple(e) for e in g["edges"]])
    assert out["valid"]
    assert [s["agent"] for s in out["steps"]] == out["order"]
    assert len(out["steps"]) == len(NODES)
    for section in ("prediction", "investigation", "degradation_case",
                    "adjudication", "remediation", "verification"):
        assert out[section], f"{section} is empty"


def test_dropping_the_confound_advocate_changes_the_run(engine):
    """The headline ablation, executable from the canvas."""
    g = default_graph()
    nodes = [n for n in g["nodes"] if n != "confound_advocate"]
    edges = [tuple(e) for e in g["edges"] if "confound_advocate" not in e]
    out = execute(engine, "MOTOR-045", engine.busiest_timestamp(), nodes, edges)
    assert out["valid"]
    assert len(out["steps"]) == len(NODES) - 1
    assert "confound_advocate" not in [s["agent"] for s in out["steps"]]
    assert out["adjudication"], "the adjudicator must still run"
    assert out["adjudication"]["confound_score"] == 0.0
    assert any("without confound" in w for w in out["warnings"])


def test_an_invalid_graph_is_refused_before_anything_runs(engine):
    out = execute(engine, "MOTOR-045", engine.busiest_timestamp(),
                  ["predictor"], [])
    assert out["valid"] is False
    assert out["errors"]
    assert "steps" not in out


def test_dropping_verification_still_produces_a_plan(engine):
    g = default_graph()
    nodes = [n for n in g["nodes"] if n != "verifier"]
    edges = [tuple(e) for e in g["edges"] if "verifier" not in e]
    out = execute(engine, "MOTOR-045", engine.busiest_timestamp(), nodes, edges)
    assert out["valid"]
    assert out["remediation"]["plan"]
    assert out["verification"] == {}, "verification did not run, so it is empty"
