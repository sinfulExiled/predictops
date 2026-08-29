"""Fleet-level analytics for the command centre.

Everything the dashboard shows is computed here from the scored fleet, so the
UI renders numbers rather than inventing them. Two things this module
deliberately does *not* do, because the system cannot honestly support them:

* it does not report "live ingestion" — the dataset is a fixed synthetic
  history being replayed, and the payload says so;
* it does not report agent daemons idling in states like "analysing" — the
  agents run on demand, so the agent panel reports what the trajectory
  registry actually recorded (executions, last run, mean duration).

The scored frame is cached per bundle+split, because scoring the whole test
split takes ~0.4 s and every panel derives from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents.evidence import EvidenceBuilder

# How the fleet health score is defined, so it is never a magic number.
HEALTH_FORMULA = "100 - 100 * (2*high + watch) / (2 * live_machines)"

_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}


def _scored(engine, split: str = "test") -> pd.DataFrame:
    """Every (machine, timestamp) in the split with its risk score."""
    key = f"{split}:{engine.bundle.kind}:{engine.bundle.feature_set}"
    stamp = str(engine.bundle.threshold)
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]

    b = engine.bundle
    if b.is_sequence():
        wi = engine.data.windows(split, b.feature_set, b.scaler, b.lookback)
        from .ml.training import predict_windows
        p = predict_windows(b.torch_model, wi)
        machines = np.array([wi.machine_of(int(i)) for i in range(len(wi))])
        ts = pd.to_datetime(pd.Series(wi.timestamps))
    else:
        x, _, machines, t, _ = engine.data.tabular(split, b.feature_set)
        p = b.score_rows(x)
        ts = pd.to_datetime(pd.Series(t))

    df = pd.DataFrame({"machine_id": machines, "timestamp": ts, "p": p})
    _CACHE[key] = (stamp, df)
    return df


def _band(p: float, alert: float, watch: float) -> str:
    return "high" if p >= alert else ("watch" if p >= watch else "normal")


def _primary_factor(engine, machine_id: str, at: pd.Timestamp) -> str:
    """Which channel moved most, relative to what counts as material."""
    try:
        raw = engine.service.raw_window(engine.data.df, machine_id, at)
    except ValueError:
        return "unknown"
    items = EvidenceBuilder(raw).channel_movements()
    if not items:
        return "none"
    from .agents.evidence import MATERIAL_ABS, MATERIAL_PCT
    best, score = "none", 0.0
    for e in items:
        bar = MATERIAL_ABS.get(e["channel"], MATERIAL_PCT)
        s = abs(float(e["value"])) / bar
        if s > score:
            best, score = e["channel"], s
    return best.replace("temp_excess", "temperature").replace("_1h", "")


def overview(engine, at=None, window_hours: int = 6,
             split: str = "test") -> dict:
    svc = engine.service
    alert, watch = svc.alert_threshold, svc.investigate_threshold
    f = _scored(engine, split)

    at = pd.Timestamp(at) if at is not None else engine.busiest_timestamp(split)
    at = pd.Timestamp(at)

    now = f[f.timestamp == at]
    live_ids = set(now.machine_id)
    all_ids = set(engine.data.df.machine_id.unique())
    down = sorted(all_ids - live_ids)

    counts = {"high": 0, "watch": 0, "normal": 0, "down": len(down)}
    for p in now.p:
        counts[_band(p, alert, watch)] += 1
    n_live = max(len(now), 1)

    # --- change since the same time yesterday ---------------------------
    prev = f[f.timestamp == at - pd.Timedelta(hours=24)]
    prev_counts = {"high": 0, "watch": 0, "normal": 0}
    for p in prev.p:
        prev_counts[_band(p, alert, watch)] += 1
    deltas = {k: counts[k] - prev_counts.get(k, 0) for k in prev_counts}
    deltas["down"] = counts["down"] - len(all_ids - set(prev.machine_id)) \
        if len(prev) else 0

    health = round(100 - 100 * (2 * counts["high"] + counts["watch"])
                   / (2 * n_live))
    prev_health = (round(100 - 100 * (2 * prev_counts["high"]
                                      + prev_counts["watch"])
                         / (2 * max(len(prev), 1))) if len(prev) else health)

    # --- risk trend over the window --------------------------------------
    lo = at - pd.Timedelta(hours=window_hours)
    win = f[(f.timestamp > lo) & (f.timestamp <= at)].copy()
    win["band"] = [_band(p, alert, watch) for p in win.p]
    grid = (win.groupby(["timestamp", "band"]).size().unstack(fill_value=0)
            .reindex(columns=["high", "watch", "normal"], fill_value=0)
            .sort_index())
    trend = [{"t": str(ts), "high": int(r.high), "watch": int(r.watch),
              "normal": int(r.normal)} for ts, r in grid.iterrows()]

    # --- machines, richest first ------------------------------------------
    spark = {mid: [round(float(v), 4) for v in g.sort_values("timestamp").p][-24:]
             for mid, g in win.groupby("machine_id")}
    raw_now = engine.data.df[engine.data.df.timestamp == at].set_index("machine_id")

    machines = []
    for _, r in now.sort_values("p", ascending=False).iterrows():
        mid = r.machine_id
        band = _band(r.p, alert, watch)
        row = raw_now.loc[mid] if mid in raw_now.index else None
        eta = None
        if band in ("high", "watch") and engine.bundle.ttf_regressor is not None \
                and row is not None:
            cols = engine.bundle.ttf_regressor.columns
            eta = engine.bundle.ttf_regressor.window(
                np.asarray([[row[c] for c in cols]], dtype=np.float32))[0]
        machines.append({
            "machine_id": mid,
            "machine_type": mid.split("-")[0].title(),
            "site": int(row["site"]) if row is not None else 0,
            "failure_probability": round(float(r.p), 4),
            "status": band,
            "confidence": round(float(engine.bundle.confidence(float(r.p))), 4),
            "trend": spark.get(mid, []),
            "primary_factor": (_primary_factor(engine, mid, at)
                               if band in ("high", "watch") else None),
            "eta": eta,
            "vibration": round(float(row["vibration"]), 3) if row is not None else None,
            "temperature": round(float(row["temperature"]), 2) if row is not None else None,
            "load": round(float(row["load"]), 3) if row is not None else None,
        })
    for mid in down:
        machines.append({"machine_id": mid,
                         "machine_type": mid.split("-")[0].title(),
                         "site": 0, "failure_probability": None,
                         "status": "down", "confidence": None, "trend": [],
                         "primary_factor": None, "eta": None})

    # --- top contributing factors, across the machines at risk ------------
    at_risk = [m for m in machines if m["status"] in ("high", "watch")]
    tally: dict[str, int] = {}
    for m in at_risk:
        tally[m["primary_factor"] or "none"] = tally.get(
            m["primary_factor"] or "none", 0) + 1
    total = sum(tally.values()) or 1
    factors = [{"factor": k, "count": v, "share": round(v / total, 4)}
               for k, v in sorted(tally.items(), key=lambda kv: -kv[1])]

    # --- alerts: real band crossings inside the window --------------------
    alerts = []
    for mid, g in win.sort_values("timestamp").groupby("machine_id"):
        bands = list(g.band)
        stamps = list(g.timestamp)
        for i in range(1, len(bands)):
            if bands[i] != bands[i - 1] and bands[i] in ("high", "watch"):
                alerts.append({
                    "machine_id": mid, "at": str(stamps[i]),
                    "minutes_ago": int((at - stamps[i]).total_seconds() // 60),
                    "from": bands[i - 1], "to": bands[i],
                    "probability": round(float(g.p.iloc[i]), 4),
                })
    alerts.sort(key=lambda a: a["minutes_ago"])

    return {
        "timestamp": str(at),
        "window_hours": window_hours,
        "period": {"start": str(f.timestamp.min()), "end": str(f.timestamp.max())},
        "counts": counts,
        "deltas": deltas,
        "health": {"score": int(health), "delta": int(health - prev_health),
                   "formula": HEALTH_FORMULA},
        "thresholds": {"alert": round(alert, 4), "investigate": round(watch, 4)},
        "trend": trend,
        "top_factors": factors,
        "machines": machines,
        "alerts": alerts[:12],
        "n_at_risk": len(at_risk),
    }


def agent_activity(engine, limit_runs: int = 40) -> list[dict]:
    """What the agents have actually done, from the trajectory registry."""
    from .agents.workflow import NODES

    rows: list[dict] = []
    for run in engine.store.runs()[:limit_runs]:
        rows.extend(engine.store.trajectory(run))
    if not rows:
        return []
    df = pd.DataFrame(rows)

    known = list(NODES) + ["data_scientist", "model_researcher", "assistant"]
    out = []
    for name in known:
        g = df[df.agent == name]
        spec = NODES.get(name)
        out.append({
            "agent": name,
            "label": spec.label if spec else name.replace("_", " ").title(),
            "question": spec.question if spec else "",
            "executions": int(len(g)),
            "last_run": str(g.created_at.max()) if len(g) else None,
            "mean_duration_s": (round(float(g.duration_s.mean()), 3)
                                if len(g) else None),
            "retries": int(g.retry_count.sum()) if len(g) else 0,
        })
    return out


def system_status(engine) -> dict:
    """Honest system panel: no 'live ingestion' claim, because there is none."""
    from .agents.workflow import NODES

    b = engine.bundle
    models = {"risk": b.kind,
              "failure_type": bool(b.type_classifier),
              "time_to_failure": bool(b.ttf_regressor),
              "reliability_curve": bool(getattr(b.reliability, "edges", []))}
    df = engine.data.df
    return {
        "agents_registered": len(NODES) + 3,   # + data scientist, research, assistant
        "models_loaded": sum(1 for v in models.values() if v),
        "models_total": len(models),
        "models": models,
        "provider": {"name": engine.provider.name, "model": engine.provider.model},
        "dataset": {
            "mode": "replay of a fixed synthetic history",
            "machines": int(df.machine_id.nunique()),
            "rows": int(len(df)),
            "start": str(df.timestamp.min()), "end": str(df.timestamp.max()),
            "resolution_minutes": 10,
        },
        "simulator": "ready",
    }
