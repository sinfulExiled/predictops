"""FastAPI service.

Thin: every endpoint delegates to the engine or reads the registry.  No
modelling logic lives here, so the dashboard and the CLI cannot diverge.

    uvicorn predictops.api.app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agents.orchestrator import PredictOpsEngine
from ..config import REPORT_DIR
from ..evaluation.scenarios import load_suite
from ..experiments.changelog import build_changelog
from ..experiments.registry import ExperimentStore
from ..llm.provider import get_provider
from ..ml.dataset import prepare
from ..simulation.interventions import CATALOGUE

STATE: dict[str, Any] = {}


SETUP_HINT = ("Generate the dataset first:  python generate_data.py  "
              "(then python run_experiments.py to train a model).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The registry and the queue never depend on generated data, so build them
    # first: the report views read committed JSON and stay usable even on a
    # fresh clone that has not generated anything yet.
    STATE["store"] = ExperimentStore()
    STATE["events"] = asyncio.Queue()
    STATE["setup_error"] = None

    try:
        data = prepare()
    except (FileNotFoundError, OSError) as exc:
        # A clone carries no dataset -- artifacts/data is regenerable and so is
        # gitignored. Starting anyway and saying what to run beats exiting with
        # a bare traceback, which is what anyone who runs uvicorn before
        # generate_data.py used to get.
        STATE["setup_error"] = f"no dataset on disk. {SETUP_HINT} ({exc})"
        print(f"[api] {STATE['setup_error']}")
    else:
        engine = PredictOpsEngine(data=data, store=STATE["store"],
                                  provider=get_provider(), verbose=False)
        try:
            engine.load_bundle()
        except Exception as exc:  # noqa: BLE001 - the API still serves data views
            print(f"[api] no model bundle yet ({exc}); "
                  f"run run_experiments.py first")
        STATE["engine"] = engine

        # Scoring the whole test period to find the busiest moment takes ~40 s
        # for a sequence model, and the fleet view is the first thing anyone
        # opens. Do it in the background so the page is warm on arrival rather
        # than hanging on a spinner. Failure here is not fatal: the request
        # path computes the same thing on demand.
        async def _warm():
            try:
                await asyncio.to_thread(engine.busiest_timestamp)
                print("[api] fleet cache warm")
            except Exception as exc:  # noqa: BLE001
                print(f"[api] fleet warm-up skipped ({exc})")

        STATE["warm_task"] = asyncio.create_task(_warm())

    yield
    task = STATE.get("warm_task")
    if task is not None:
        task.cancel()
    STATE.clear()


app = FastAPI(title="PredictOps API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def engine() -> PredictOpsEngine:
    e = STATE.get("engine")
    if e is None:
        raise HTTPException(503, STATE.get("setup_error") or "engine not ready")
    return e


def _clean(obj):
    """NaN/NumPy -> JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    return obj


# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    e = STATE.get("engine")
    return {
        "status": "ok" if e else "setup_required",
        "setup_error": STATE.get("setup_error"),
        "data_loaded": e is not None,
        "model_loaded": bool(e and e.bundle),
        "model": (
            {"kind": e.bundle.kind, "feature_set": e.bundle.feature_set,
             "threshold": round(e.bundle.threshold, 4),
             "rationale": e.bundle.selection_rationale}
            if e and e.bundle else None),
        "llm_provider": {"name": e.provider.name, "model": e.provider.model}
        if e else None,
    }


@app.get("/api/machines")
def machines(at: str | None = None):
    """Fleet snapshot. With no `at`, opens on the busiest moment in the test
    period rather than the last sample, which is usually quiet."""
    e = engine()
    df = e.fleet_scores(at=at)
    test = e.data.df[e.data.df["split"] == "test"]["timestamp"]
    return _clean({
        "timestamp": df.attrs.get("timestamp"),
        "busiest": str(e.busiest_timestamp()),
        "latest": str(test.max()),
        "earliest": str(test.min()),
        "machines": df.to_dict(orient="records"),
    })


@app.get("/api/machines/{machine_id}/telemetry")
def telemetry(machine_id: str, hours: int = Query(24, ge=1, le=168),
              until: str | None = None):
    e = engine()
    g = e.data.df[e.data.df.machine_id == machine_id].sort_values("timestamp")
    if g.empty:
        raise HTTPException(404, f"unknown machine {machine_id}")
    end = pd.Timestamp(until) if until else g["timestamp"].max()
    start = end - pd.Timedelta(hours=hours)
    w = g[(g.timestamp > start) & (g.timestamp <= end)]
    cols = ["timestamp", "temperature", "vibration", "pressure", "rpm",
            "current", "voltage", "power", "load", "ambient_temp",
            "is_downtime"]
    return _clean({
        "machine_id": machine_id,
        "series": w[[c for c in cols if c in w.columns]].to_dict(orient="records"),
    })


@app.get("/api/assistant/sources")
def assistant_sources():
    """What the assistant can actually draw on, with real counts."""
    import json as _json
    import sqlite3
    from ..config import DATA_DIR, EXPERIMENT_DIR
    from ..simulation.interventions import CATALOGUE

    e = engine()
    con = sqlite3.connect(EXPERIMENT_DIR / "experiments.db")
    n_exp, n_runs = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT run_id) FROM experiments").fetchone()
    n_steps = con.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
    n_ev = 0
    for (blob,) in con.execute(
            "SELECT output FROM trajectories WHERE agent='investigator'"):
        try:
            n_ev += len(_json.loads(blob).get("evidence", []))
        except Exception:  # noqa: BLE001
            pass
    con.close()

    def _count(path, key=None):
        p = path
        if not p.exists():
            return 0
        d = _json.loads(p.read_text())
        return len(d if key is None else d.get(key, []))

    scenarios = _count(DATA_DIR / "scenarios.json")
    sweep = _count(REPORT_DIR / "ablation_adjudication.json", "sweep")

    return {
        "sources": [
            {"key": "fleet", "label": "Fleet scores", "value":
             int(e.data.df.machine_id.nunique()), "unit": "machines tracked",
             "detail": "Risk, confidence and time-to-failure for every machine "
                       "at any moment in the record.", "href": "#/"},
            {"key": "evidence", "label": "Evidence items", "value": n_ev,
             "unit": "recorded", "detail": "Facts computed from telemetry, each "
                     "carrying the recipe that reproduces it.",
             "href": "#/investigate"},
            {"key": "experiments", "label": "Experiments", "value": n_exp,
             "unit": f"across {n_runs} run{'s' if n_runs != 1 else ''}",
             "detail": "Every model candidate with its measured outcome and "
                       "the decision taken.", "href": "#/experiments"},
            {"key": "evaluation", "label": "Evaluation", "value": scenarios,
             "unit": "fixed scenarios", "detail": "Baseline against agent on "
                     "identical cases, including the hard ones.",
             "href": "#/evaluation"},
            {"key": "ablation", "label": "Ablation", "value": sweep,
             "unit": "threshold points", "detail": "Whether the hypothesis "
                     "contest changed any decision. It did not.",
             "href": "#/lab"},
            {"key": "catalogue", "label": "Interventions", "value":
             len(CATALOGUE), "unit": "approved actions",
             "detail": "The only actions that may be proposed. Nothing outside "
                       "this list.", "href": "#/remediate"},
        ],
        "agent_steps_logged": n_steps,
    }


@app.get("/api/fleet/overview")
def fleet_overview(at: str | None = None, hours: int = Query(6, ge=1, le=48)):
    """Everything the command centre renders, computed from the scored fleet."""
    from ..fleet import overview
    e = engine()
    if e.service is None:
        raise HTTPException(503, "no model bundle; run run_experiments.py")
    return _clean(overview(e, at=at, window_hours=hours))


@app.get("/api/fleet/agents")
def fleet_agents():
    """What the agents have actually done, from the trajectory registry."""
    from ..fleet import agent_activity
    return _clean({"agents": agent_activity(engine())})


@app.get("/api/system")
def system():
    from ..fleet import system_status
    e = engine()
    if e.bundle is None:
        raise HTTPException(503, "no model bundle")
    return _clean(system_status(e))


@app.get("/api/workflow")
def workflow_spec():
    """Node contracts and the default wiring the canvas starts from."""
    from ..agents.workflow import default_graph, describe_nodes
    return {"nodes": describe_nodes(), "default": default_graph()}


class GraphRequest(BaseModel):
    nodes: list[str]
    edges: list[list[str]]
    machine_id: str | None = None
    timestamp: str | None = None


@app.post("/api/workflow/validate")
def workflow_validate(req: GraphRequest):
    from ..agents.workflow import validate
    return validate(req.nodes, [tuple(e) for e in req.edges])


@app.post("/api/workflow/run")
async def workflow_run(req: GraphRequest):
    from ..agents.workflow import execute
    e = engine()
    if e.service is None:
        raise HTTPException(503, "no model bundle; run run_experiments.py")
    mid = req.machine_id
    if not mid:
        raise HTTPException(400, "machine_id is required")
    ts = pd.Timestamp(req.timestamp) if req.timestamp else e.busiest_timestamp()

    def _run():
        return execute(e, mid, ts, req.nodes, [tuple(x) for x in req.edges])

    out = await asyncio.to_thread(_run)
    q: asyncio.Queue = STATE["events"]
    for step in e.store.trajectory(e.run_id)[-len(out.get("steps", [])):]:
        q.put_nowait({"type": "agent_step", **_clean(step)})
    return _clean(out)


class AskRequest(BaseModel):
    question: str
    allow_actions: bool = True


@app.post("/api/assistant")
async def assistant(req: AskRequest):
    """Grounded Q&A over computed artifacts. Never originates a fact."""
    e = engine()
    from ..agents.assistant import AssistantAgent

    def _run():
        return AssistantAgent().execute(
            e.ctx, question=req.question, engine=e,
            allow_actions=req.allow_actions).output

    out = await asyncio.to_thread(_run)
    q: asyncio.Queue = STATE["events"]
    for step in e.store.trajectory(e.run_id)[-1:]:
        q.put_nowait({"type": "agent_step", **_clean(step)})
    return _clean(out)


class IncidentRequest(BaseModel):
    machine_id: str
    timestamp: str | None = None
    horizon_hours: float | None = None


@app.post("/api/incidents")
async def incident(req: IncidentRequest):
    e = engine()
    if e.bundle is None:
        raise HTTPException(503, "no model bundle; run run_experiments.py")
    g = e.data.df[e.data.df.machine_id == req.machine_id]
    if g.empty:
        raise HTTPException(404, f"unknown machine {req.machine_id}")
    ts = pd.Timestamp(req.timestamp) if req.timestamp else g["timestamp"].max()

    def _run():
        return e.run_incident(req.machine_id, ts, save=True,
                              horizon_hours=req.horizon_hours)

    report = await asyncio.to_thread(_run)
    q: asyncio.Queue = STATE["events"]
    for step in report.trajectory:
        q.put_nowait({"type": "agent_step", **_clean(step)})
    return _clean(report.to_dict())


@app.get("/api/interventions")
def interventions():
    return {"catalogue": [iv.to_dict() for iv in CATALOGUE.values()]}


@app.get("/api/experiments")
def experiments(run_id: str | None = None):
    store: ExperimentStore = STATE["store"]
    run_id = run_id or store.latest_run_id()
    return _clean({
        "run_id": run_id,
        "runs": store.experiment_runs(),
        "experiments": [
            {"id": x.id, "stage": x.stage, "name": x.name, "model": x.model,
             "feature_set": x.feature_set, "hypothesis": x.hypothesis,
             "decision": x.decision, "learning": x.learning,
             "duration_s": x.duration_s, "params": x.params,
             "metrics": x.metrics}
            for x in store.all(run_id)],
    })


@app.get("/api/changelog")
def changelog(run_id: str | None = None):
    store: ExperimentStore = STATE["store"]
    return {"markdown": build_changelog(store, run_id)}


@app.get("/api/evaluation")
def evaluation():
    path = REPORT_DIR / "evaluation.json"
    if not path.exists():
        raise HTTPException(404, "run evaluate.py first")
    return json.loads(path.read_text())


@app.get("/api/ablation")
def ablation():
    """The adjudication ablation: did the hypothesis contest earn its place?"""
    path = REPORT_DIR / "ablation_adjudication.json"
    if not path.exists():
        raise HTTPException(404, "run ablate_adjudication.py first")
    return json.loads(path.read_text())


@app.get("/api/thresholds")
def thresholds():
    """The model service contract the orchestration layer routes on."""
    e = engine()
    if e.service is None:
        raise HTTPException(503, "no model bundle")
    return e.service.describe()


@app.get("/api/scenarios")
def scenarios():
    try:
        return {"scenarios": [s.to_dict() for s in load_suite()]}
    except FileNotFoundError:
        raise HTTPException(404, "no scenario suite; run evaluate.py")


@app.get("/api/trajectories")
def trajectories(run_id: str | None = None):
    store: ExperimentStore = STATE["store"]
    runs = store.runs()
    run_id = run_id or (runs[0] if runs else None)
    if run_id is None:
        return {"run_id": None, "steps": []}
    return _clean({"run_id": run_id, "runs": runs[:25],
                   "steps": store.trajectory(run_id)})


@app.websocket("/ws/agent-activity")
async def agent_activity(ws: WebSocket):
    """Streams agent steps as incidents run."""
    await ws.accept()
    q: asyncio.Queue = STATE["events"]
    try:
        await ws.send_json({"type": "connected",
                            "at": datetime.now().isoformat(timespec="seconds")})
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=20.0)
                await ws.send_json(event)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        return


# --------------------------------------------------------------------------
# Serve the built dashboard from this same process, when it has been built,
# so one `uvicorn` command gives a reviewer the whole product.
#
# Registered LAST on purpose: a Mount at "/" matches every path, so declaring
# it earlier would shadow the API routes and the websocket above it.
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _DIST.is_dir():
    # The dashboard uses a hash router (`createHashRouter`), so every client
    # route lives under "/#/..." and the server only ever sees "/". Plain
    # StaticFiles is therefore correct and complete here: a refresh or a
    # shared link on any page resolves to index.html on its own, and a
    # path-style URL like "/assistant" -- which the app never produces --
    # stays a 404 rather than silently rendering the wrong page.
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="ui")
