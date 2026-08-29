"""Composable workflow graphs.

The incident workflow in `orchestrator.py` is one particular wiring of the
agents. This module makes that wiring *data*, so a graph can be composed,
validated and executed — which is what the workflow canvas in the UI edits.

The important design choice is that **validation is derived from the agents'
own declared contracts**, not written separately for the editor. Each node
declares what state keys it consumes and produces, so:

* an edge A -> B is valid only if B actually consumes something A produces,
* a node may only run once everything in `requires` has been produced upstream,
* omitting an optional node is legal, and the graph downstream adapts.

That makes a rejected edge mean something ("the simulator produces nothing the
confound advocate reads") rather than being an arbitrary UI rule. It also means
adding an agent to the system automatically teaches the editor about it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assistant import AssistantAgent  # noqa: F401  (registry completeness)
from .context import ContextAgent
from .hypothesis import Adjudicator, ConfoundAdvocate, DegradationAdvocate
from .investigator import InvestigationAgent
from .predictor import PredictionAgent
from .remediation import RemediationAgent
from .simulator import SimulationAgent
from .verifier import VerificationAgent


@dataclass(frozen=True)
class NodeSpec:
    """One agent's contract, as the graph sees it."""

    name: str
    label: str
    question: str
    agent: Any
    provides: tuple[str, ...]
    requires: tuple[str, ...] = ()          # hard: must exist upstream
    optional_inputs: tuple[str, ...] = ()   # used when present
    removable: bool = True
    note: str = ""

    def consumes(self) -> tuple[str, ...]:
        return tuple(self.requires) + tuple(self.optional_inputs)


NODES: dict[str, NodeSpec] = {
    "predictor": NodeSpec(
        "predictor", "Prediction", "What is likely to happen?",
        PredictionAgent, provides=("prediction",), removable=False,
        note="Asks the model service. Everything downstream needs its output."),
    "context": NodeSpec(
        "context", "Context", "What do we already know about this machine?",
        ContextAgent, provides=("context",),
        note="Machine dossier. The confound advocate uses it to argue "
             "post-service run-in and unreliable instrumentation."),
    "investigator": NodeSpec(
        "investigator", "Investigation", "What actually changed?",
        InvestigationAgent, provides=("investigation", "evidence_builder"),
        removable=False,
        note="Builds the shared factual record every advocate argues over."),
    "degradation_advocate": NodeSpec(
        "degradation_advocate", "Cause advocate",
        "Why is a fault the best explanation?",
        DegradationAdvocate, provides=("degradation",),
        requires=("evidence_builder", "prediction"),
        optional_inputs=("context",), removable=False,
        note="Without it there is no diagnosis to act on."),
    "confound_advocate": NodeSpec(
        "confound_advocate", "Confound advocate", "Why is nothing wrong?",
        ConfoundAdvocate, provides=("confound",),
        requires=("evidence_builder", "prediction"),
        optional_inputs=("context",),
        note="Optional. Measured contribution to accuracy is +0.0000 F1; it "
             "is kept as a safety mechanism, so removing it is a fair thing "
             "to try."),
    "adjudicator": NodeSpec(
        "adjudicator", "Adjudicator", "Which reading survives?",
        Adjudicator, provides=("adjudication",),
        requires=("degradation", "prediction"),
        optional_inputs=("confound",),
        note="With no confound advocate upstream it has nothing to weigh "
             "against, and the degradation case stands unchallenged."),
    "remediation": NodeSpec(
        "remediation", "Remediation", "What can we do?",
        RemediationAgent, provides=("remediation",),
        requires=("prediction", "investigation"),
        optional_inputs=("adjudication", "context"), removable=False,
        note="Gated on the adjudication when one is present."),
    "simulator": NodeSpec(
        "simulator", "Simulation", "What happens if we do it?",
        SimulationAgent, provides=("simulation",),
        requires=("remediation", "prediction"),
        note="Rolls each action forward against a do-nothing control."),
    "verifier": NodeSpec(
        "verifier", "Verification", "Do we have enough evidence to act?",
        VerificationAgent, provides=("verification",),
        requires=("prediction", "investigation", "remediation"),
        optional_inputs=("simulation", "adjudication"),
        note="Re-derives every evidence claim from raw telemetry."),
}

#: The wiring `orchestrator.run_incident` uses.
DEFAULT_EDGES: list[tuple[str, str]] = [
    ("predictor", "degradation_advocate"),
    ("predictor", "confound_advocate"),
    ("predictor", "adjudicator"),
    ("predictor", "remediation"),
    ("predictor", "simulator"),
    ("predictor", "verifier"),
    ("context", "degradation_advocate"),
    ("context", "confound_advocate"),
    ("context", "remediation"),
    ("investigator", "degradation_advocate"),
    ("investigator", "confound_advocate"),
    ("investigator", "remediation"),
    ("investigator", "verifier"),
    ("degradation_advocate", "adjudicator"),
    ("confound_advocate", "adjudicator"),
    ("adjudicator", "remediation"),
    ("adjudicator", "verifier"),
    ("remediation", "simulator"),
    ("remediation", "verifier"),
    ("simulator", "verifier"),
]


def default_graph() -> dict:
    return {"nodes": list(NODES), "edges": [list(e) for e in DEFAULT_EDGES]}


def describe_nodes() -> list[dict]:
    return [{"name": n.name, "label": n.label, "question": n.question,
             "provides": list(n.provides), "requires": list(n.requires),
             "optional_inputs": list(n.optional_inputs),
             "removable": n.removable, "note": n.note}
            for n in NODES.values()]


# --------------------------------------------------------------------------
def validate_edge(src: str, dst: str) -> tuple[bool, str]:
    """Is this connection meaningful? Used for live feedback while wiring."""
    if src == dst:
        return False, "a node cannot feed itself"
    if src not in NODES:
        return False, f"unknown node '{src}'"
    if dst not in NODES:
        return False, f"unknown node '{dst}'"
    shared = set(NODES[src].provides) & set(NODES[dst].consumes())
    if not shared:
        return False, (f"{NODES[dst].label} does not read anything "
                       f"{NODES[src].label} produces "
                       f"(it produces {', '.join(NODES[src].provides)})")
    return True, f"carries {', '.join(sorted(shared))}"


def validate(nodes: list[str], edges: list[tuple[str, str]]) -> dict:
    """Full graph check: known nodes, acyclic, every requirement satisfied."""
    errors: list[str] = []
    warnings: list[str] = []

    unknown = [n for n in nodes if n not in NODES]
    errors += [f"unknown node '{n}'" for n in unknown]
    present = [n for n in nodes if n in NODES]

    for name, spec in NODES.items():
        if not spec.removable and name not in present:
            errors.append(f"{spec.label} cannot be removed: {spec.note}")

    kept_edges = [(a, b) for a, b in edges if a in present and b in present]
    for a, b in kept_edges:
        ok, why = validate_edge(a, b)
        if not ok:
            errors.append(f"invalid connection {a} -> {b}: {why}")

    # --- topological order (Kahn), which also detects cycles --------------
    incoming = {n: set() for n in present}
    for a, b in kept_edges:
        incoming[b].add(a)
    order, ready = [], [n for n in present if not incoming[n]]
    remaining = {n: set(v) for n, v in incoming.items()}
    # Break ties by declaration order rather than alphabetically, so a valid
    # graph reads in the canonical stage order instead of a surprising one.
    rank = {name: i for i, name in enumerate(NODES)}
    while ready:
        ready.sort(key=lambda n: rank.get(n, 99))
        n = ready.pop(0)
        order.append(n)
        for m in present:
            if n in remaining[m]:
                remaining[m].discard(n)
                if not remaining[m] and m not in order and m not in ready:
                    ready.append(m)
    if len(order) != len(present):
        cyc = sorted(set(present) - set(order))
        errors.append(f"the graph has a cycle involving {', '.join(cyc)}")
        order = []

    # --- are requirements satisfied by a CONNECTED ancestor? --------------
    # Presence is not enough. If the edge is deleted the data does not flow,
    # so the check walks the graph rather than the node list -- otherwise the
    # edges would be decorative and the canvas would be theatre.
    parents: dict[str, set[str]] = {n: set() for n in present}
    for a, b in kept_edges:
        parents[b].add(a)

    reachable: dict[str, set[str]] = {}
    for n in order:
        avail: set[str] = set()
        for pnode in parents[n]:
            avail |= set(NODES[pnode].provides) | reachable.get(pnode, set())
        reachable[n] = avail
        spec = NODES[n]
        missing = [r for r in spec.requires if r not in avail]
        if missing:
            errors.append(
                f"{spec.label} needs {', '.join(missing)} but no connected "
                "upstream node provides it")
        soft = [o for o in spec.optional_inputs if o not in avail]
        if soft:
            warnings.append(
                f"{spec.label} will run without {', '.join(soft)}")

    for name in NODES:
        if name not in present and NODES[name].removable:
            warnings.append(f"{NODES[name].label} is not in this graph")

    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "order": order}


# --------------------------------------------------------------------------
def execute(engine, machine_id: str, timestamp, nodes: list[str],
            edges: list[tuple[str, str]]) -> dict:
    """Run a validated graph and return each agent's output plus timings."""
    import time

    check = validate(nodes, edges)
    if not check["valid"]:
        return {"valid": False, **check}

    ctx = engine.ctx
    state: dict[str, Any] = {}

    # Only data from a connected ancestor is visible to a node, so deleting an
    # edge genuinely starves the target rather than merely redrawing the line.
    parents: dict[str, set[str]] = {n: set() for n in nodes if n in NODES}
    for a, b in edges:
        if a in parents and b in parents:
            parents[b].add(a)

    def upstream(node: str) -> set[str]:
        seen, stack = set(), list(parents.get(node, ()))
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(parents.get(p, ()))
        keys: set[str] = set()
        for p in seen:
            keys |= set(NODES[p].provides)
        return keys

    def visible(node: str) -> dict:
        keys = upstream(node)
        return {k: v for k, v in state.items() if k in keys}
    steps: list[dict] = []
    t0 = time.time()

    for name in check["order"]:
        spec = NODES[name]
        seen = visible(name)
        kwargs: dict[str, Any] = {}
        if name in ("predictor", "investigator"):
            kwargs.update(service=engine.service, machine_id=machine_id,
                          timestamp=timestamp)
            if name == "investigator":
                kwargs["library"] = engine.library
        elif name == "context":
            kwargs.update(machine_id=machine_id, timestamp=timestamp)
        elif name in ("degradation_advocate", "confound_advocate"):
            kwargs.update(builder=seen.get("evidence_builder"),
                          prediction=seen.get("prediction", {}),
                          context=seen.get("context", {}))
            if name == "degradation_advocate":
                kwargs["neighbours"] = seen.get(
                    "investigation", {}).get("similar_past_failures", [])
        elif name == "adjudicator":
            kwargs.update(degradation=seen.get("degradation", {}),
                          confound=seen.get("confound",
                                            {"score": 0.0,
                                             "alternative_explanations": []}),
                          prediction=seen.get("prediction", {}))
        elif name == "remediation":
            kwargs.update(prediction=seen.get("prediction", {}),
                          investigation=seen.get("investigation", {}),
                          adjudication=seen.get("adjudication"),
                          context=seen.get("context", {}))
        elif name == "simulator":
            kwargs.update(bundle=engine.bundle, machine_id=machine_id,
                          timestamp=timestamp,
                          plan=seen.get("remediation", {}).get("plan", []),
                          baseline_probability=seen.get("prediction", {})
                          .get("failure_probability", 0.0))
        elif name == "verifier":
            kwargs.update(bundle=engine.bundle,
                          prediction=seen.get("prediction", {}),
                          investigation=seen.get("investigation", {}),
                          remediation=seen.get("remediation", {}),
                          simulation=seen.get("simulation",
                                              {"arms": [], "no_action": {}}),
                          adjudication=seen.get("adjudication"))

        started = time.time()
        result = spec.agent().execute(ctx, **kwargs)
        out = result.output

        # the advocates and the adjudicator enrich the investigation record so
        # remediation and verification see the same evidence the default
        # workflow gives them
        if name == "investigator":
            state["investigation"] = out
            state["evidence_builder"] = ctx.state.get("evidence_builder")
        elif name == "degradation_advocate":
            state["degradation"] = out
            inv = dict(state.get("investigation", {}))
            b = ctx.state.get("evidence_builder")
            inv["evidence"] = b.items if b else inv.get("evidence", [])
            inv["ranked_hypotheses"] = out.get("ranked_types", [])
            inv.setdefault("likely_failure_type", out.get("failure_type"))
            inv["narrative"] = " ".join(e["claim"] + "."
                                        for e in inv.get("evidence", []))
            state["investigation"] = inv
        elif name == "adjudicator":
            state["adjudication"] = out
            inv = dict(state.get("investigation", {}))
            inv["likely_failure_type"] = out.get("failure_type")
            state["investigation"] = inv
        else:
            for key in spec.provides:
                state[key] = out

        steps.append({"agent": name, "label": spec.label,
                      "duration_s": round(time.time() - started, 3),
                      "summary": result.action,
                      "reason": result.reason})

    return {
        "valid": True, "warnings": check["warnings"], "order": check["order"],
        "machine_id": machine_id, "timestamp": str(timestamp),
        "duration_s": round(time.time() - t0, 3),
        "steps": steps,
        "prediction": state.get("prediction", {}),
        "context": state.get("context", {}),
        "investigation": state.get("investigation", {}),
        "degradation_case": state.get("degradation", {}),
        "confound_case": state.get("confound", {}),
        "adjudication": state.get("adjudication", {}),
        "remediation": state.get("remediation", {}),
        "simulation": state.get("simulation", {}),
        "verification": state.get("verification", {}),
    }
