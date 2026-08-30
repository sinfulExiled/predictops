import { useEffect, useMemo, useRef, useState } from "react";
import { api, TrajectoryStep } from "../api";
import {
  ErrorNote,
  Loading,
  Panel,
  Pill,
  num,
  pageWindow,
  useAsync,
} from "../components/common";
import { Icon } from "../components/icons";

const REPO_DOCS = "https://github.com/sinfulExiled/predictops#readme";

/** One colour and glyph per agent, so a run reads as a shape before it reads
 *  as text. Every agent the registry actually records is covered; anything new
 *  falls back rather than rendering an invisible marker. */
const AGENT: Record<string, { color: string; icon: keyof typeof Icon }> = {
  data_scientist: { color: "#4a9eff", icon: "graph" },
  model_researcher: { color: "#a371f7", icon: "lab" },
  predictor: { color: "#3fb950", icon: "target" },
  context: { color: "#56d4dd", icon: "layers" },
  investigator: { color: "#d29922", icon: "search" },
  degradation_advocate: { color: "#f0883e", icon: "trend" },
  confound_advocate: { color: "#56d4dd", icon: "eye" },
  adjudicator: { color: "#a371f7", icon: "check" },
  remediation: { color: "#f0883e", icon: "wrench" },
  simulator: { color: "#56d4dd", icon: "beaker" },
  verifier: { color: "#f85149", icon: "info" },
  assistant: { color: "#a371f7", icon: "chat" },
};
const FALLBACK = { color: "#8b98a5", icon: "activity" as const };
const agentOf = (a: string) => AGENT[a] ?? FALLBACK;

/** Runs are named by what produced them; group the dropdown the same way so a
 *  real pipeline run is not lost among pytest fixtures. */
function runGroup(r: string) {
  if (r.startsWith("pipeline-")) return "Pipeline runs";
  if (r.startsWith("incident-")) return "Incidents";
  if (r.startsWith("eval-")) return "Evaluations";
  if (r.startsWith("ablation-")) return "Ablations";
  if (r.startsWith("pytest")) return "Test runs";
  return "Other";
}
const GROUP_ORDER = [
  "Pipeline runs", "Incidents", "Evaluations", "Ablations", "Other", "Test runs",
];

const clock = (iso?: string) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(+d)
    ? "—"
    : d.toLocaleTimeString([], {
        hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
      });
};

function StepCard({ s, live }: { s: TrajectoryStep & { _live?: boolean }; live?: boolean }) {
  const a = agentOf(s.agent);
  const I = Icon[a.icon];
  const [copied, setCopied] = useState(false);

  return (
    <div className="tl-row">
      <div className="tl-rail">
        <span className="tl-node" style={{ color: a.color, borderColor: a.color, background: `${a.color}1f` }}>
          <I />
        </span>
      </div>

      <div className={`tl-card${live ? " tl-live" : ""}`}>
        <div className="tl-head">
          <div className="tl-who">
            <span style={{ color: a.color, fontWeight: 600 }}>{s.agent}</span>
            <span className="tag">step {s.step}</span>
            {live && <Pill kind="pass">live</Pill>}
            {s.retry_count > 0 && <Pill kind="warn">{s.retry_count} retry</Pill>}
          </div>
          <div className="tl-meta">
            <span className="tl-time">{clock(s.created_at)}</span>
            <span className="tl-dur">
              <Icon.clock />
              {num(s.duration_s, 2)}s
            </span>
            <button
              className="icon-btn"
              title="Copy this step as JSON"
              onClick={() => {
                navigator.clipboard
                  ?.writeText(JSON.stringify(s, null, 2))
                  .then(() => {
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1400);
                  })
                  .catch(() => {});
              }}
            >
              {copied ? <Icon.check /> : <Icon.copy />}
            </button>
          </div>
        </div>

        <div className="tl-action">
          <span className="tl-label">Action:</span> {s.action}
        </div>
        {s.reason && (
          <div className="tl-line">
            <span className="tl-label plain">Reason:</span> {s.reason}
          </div>
        )}
        {s.input_summary && (
          <div className="tl-input">Input: {s.input_summary}</div>
        )}

        {s.tools_used?.length > 0 && (
          <div className="tl-tools">
            {s.tools_used.map((t) => (
              <span className="tool-chip" key={t}>
                {t}
              </span>
            ))}
          </div>
        )}

        <details className="tl-out">
          <summary>Output</summary>
          <pre>{JSON.stringify(s.output, null, 2).slice(0, 6000)}</pre>
        </details>
      </div>
    </div>
  );
}

export default function AgentActivity() {
  const [runId, setRunId] = useState<string | undefined>();
  const traj = useAsync(() => api.trajectories(runId), [runId]);
  const [live, setLive] = useState<TrajectoryStep[]>([]);
  const [wsState, setWsState] = useState("connecting");
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(20);
  const [order, setOrder] = useState<"oldest" | "newest">("oldest");
  const [showDetails, setShowDetails] = useState(false);
  const topRef = useRef<HTMLDivElement>(null);

  const activeRun = runId ?? traj.data?.run_id;

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/agent-activity`);
    ws.onopen = () => setWsState("live");
    ws.onclose = () => setWsState("disconnected");
    ws.onerror = () => setWsState("error");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "agent_step") setLive((l) => [...l, msg].slice(-200));
    };
    return () => ws.close();
  }, []);

  // Steps streamed for a different run must not be shown under this one.
  useEffect(() => setLive([]), [activeRun]);
  useEffect(() => setPage(1), [activeRun, perPage, order]);

  const base = traj.data?.steps ?? [];
  const liveForRun = live.filter((s) => !s.run_id || s.run_id === activeRun);
  // A fresh investigation opens a NEW run id, so its steps would otherwise
  // stream in and be silently filtered out while the "live" pill claimed
  // everything was working. Surface them instead.
  const otherRuns = [
    ...new Set(live.filter((s) => s.run_id && s.run_id !== activeRun).map((s) => s.run_id!)),
  ];
  const seen = new Set(base.map((s) => `${s.agent}|${s.step}`));
  const liveNew = liveForRun.filter((s) => !seen.has(`${s.agent}|${s.step}`));

  const all = useMemo(() => {
    const merged = [...base, ...liveNew.map((s) => ({ ...s, _live: true }))];
    return order === "newest" ? [...merged].reverse() : merged;
  }, [base, liveNew.length, order]);

  const total = all.length;
  const pages = Math.max(1, Math.ceil(total / perPage));
  const cur = Math.min(page, pages);
  const from = (cur - 1) * perPage;
  const slice = all.slice(from, from + perPage);

  const summary = useMemo(() => {
    const agents = new Map<string, number>();
    let secs = 0;
    let retries = 0;
    for (const s of all) {
      agents.set(s.agent, (agents.get(s.agent) ?? 0) + 1);
      secs += s.duration_s ?? 0;
      retries += s.retry_count ?? 0;
    }
    const times = all.map((s) => +new Date(s.created_at)).filter((t) => !isNaN(t));
    return {
      agents: [...agents.entries()].sort((a, b) => b[1] - a[1]),
      secs,
      retries,
      first: times.length ? new Date(Math.min(...times)) : null,
      last: times.length ? new Date(Math.max(...times)) : null,
    };
  }, [all]);

  function exportRun() {
    const blob = new Blob(
      [JSON.stringify({ run_id: activeRun, steps: all }, null, 2)],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trajectory-${activeRun ?? "run"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function goto(p: number) {
    setPage(Math.min(Math.max(1, p), pages));
    topRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  if (traj.loading) return <Loading what="agent trajectories" />;
  if (traj.error) return <ErrorNote error={traj.error} />;

  const runs = traj.data?.runs ?? [];
  const grouped = GROUP_ORDER.map((g) => [g, runs.filter((r) => runGroup(r) === g)] as const)
    .filter(([, rs]) => rs.length > 0);

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar" ref={topRef}>
        <div>
          <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
            Agent Activity
          </h2>
          <p style={{ margin: "5px 0 0", color: "var(--muted)", fontSize: 13.5, maxWidth: 820 }}>
            Every agent execution is logged: what it did, why, which tools it
            called, how long it took, and any retries. These are the
            trajectories, straight from the registry.
          </p>
        </div>
        <div className="topbar-right">
          <a className="ghost btn-link" href={REPO_DOCS} target="_blank" rel="noreferrer">
            <Icon.book />
            Docs
          </a>
          <button className="ghost accent" onClick={exportRun} disabled={!total}>
            <Icon.download />
            Export
          </button>
        </div>
      </div>

      {/* ---------------- run controls ---------------- */}
      <Panel>
        <div className="run-bar">
          <span className="small muted">Run</span>
          <select
            value={activeRun ?? ""}
            onChange={(e) => setRunId(e.target.value)}
            style={{ minWidth: 250 }}
          >
            {grouped.map(([g, rs]) => (
              <optgroup key={g} label={g}>
                {rs.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>

          <span className={`ws-pill ${wsState === "live" ? "on" : "off"}`}>
            <Icon.wave />
            websocket {wsState}
          </span>

          <span className="small muted hide-narrow">
            new steps stream in here as investigations run
          </span>

          <div className="run-bar-right">
            <div className="seg">
              <button
                className={order === "oldest" ? "on" : ""}
                onClick={() => setOrder("oldest")}
              >
                Oldest first
              </button>
              <button
                className={order === "newest" ? "on" : ""}
                onClick={() => setOrder("newest")}
              >
                Newest first
              </button>
            </div>
            <button className="ghost accent" onClick={() => setShowDetails((v) => !v)}>
              <Icon.info />
              {showDetails ? "Hide run details" : "View run details"}
            </button>
          </div>
        </div>

        {showDetails && (
          <div className="run-details">
            <div className="rd-grid">
              <div>
                <div className="label">Steps</div>
                <div className="v">{total}</div>
              </div>
              <div>
                <div className="label">Agents involved</div>
                <div className="v">{summary.agents.length}</div>
              </div>
              <div>
                <div className="label">Total agent time</div>
                <div className="v">{summary.secs.toFixed(1)}s</div>
              </div>
              <div>
                <div className="label">Retries</div>
                <div className="v">{summary.retries}</div>
              </div>
              <div>
                <div className="label">First step</div>
                <div className="v small">{clock(summary.first?.toISOString())}</div>
              </div>
              <div>
                <div className="label">Last step</div>
                <div className="v small">{clock(summary.last?.toISOString())}</div>
              </div>
            </div>
            <div className="rd-agents">
              {summary.agents.map(([a, n]) => {
                const c = agentOf(a);
                return (
                  <span key={a} className="rd-agent">
                    <i style={{ background: c.color }} />
                    {a}
                    <b>{n}</b>
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </Panel>

      {otherRuns.length > 0 && (
        <div className="banner info" style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <strong>
            Steps are streaming for {otherRuns.length === 1 ? "another run" : "other runs"}:
          </strong>
          {otherRuns.map((r) => (
            <button key={r} className="ghost" onClick={() => setRunId(r)}>
              switch to {r}
            </button>
          ))}
        </div>
      )}

      {liveNew.length > 0 && (
        <div className="banner info" style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <strong>{liveNew.length} step(s) streamed in since this run was loaded.</strong>
          <button
            className="ghost"
            onClick={() => goto(order === "oldest" ? pages : 1)}
          >
            Jump to them
          </button>
        </div>
      )}

      {total === 0 ? (
        <div className="banner info">
          No agent steps recorded for this run yet. Run an investigation from
          the Machine Investigation page.
        </div>
      ) : (
        <>
          <div className="timeline">
            {slice.map((s, i) => (
              <StepCard
                key={`${s.agent}-${s.step}-${from + i}`}
                s={s}
                live={(s as any)._live}
              />
            ))}
          </div>

          {/* ---------------- pager ---------------- */}
          <div className="pager">
            <span className="small muted">
              Showing {from + 1}–{Math.min(from + perPage, total)} of {total} steps
            </span>
            <div className="pager-btns">
              <button onClick={() => goto(1)} disabled={cur === 1} title="First">
                «
              </button>
              <button onClick={() => goto(cur - 1)} disabled={cur === 1} title="Previous">
                ‹
              </button>
              {pageWindow(cur, pages).map((p, i) =>
                p === "…" ? (
                  <span key={`g${i}`} className="pager-gap">
                    …
                  </span>
                ) : (
                  <button
                    key={p}
                    className={p === cur ? "on" : ""}
                    onClick={() => goto(p as number)}
                  >
                    {p}
                  </button>
                ),
              )}
              <button onClick={() => goto(cur + 1)} disabled={cur === pages} title="Next">
                ›
              </button>
              <button onClick={() => goto(pages)} disabled={cur === pages} title="Last">
                »
              </button>
            </div>
            <select
              value={perPage}
              onChange={(e) => setPerPage(Number(e.target.value))}
              title="Steps per page"
            >
              {[10, 20, 50, 100].map((n) => (
                <option key={n} value={n}>
                  {n} / page
                </option>
              ))}
            </select>
          </div>
        </>
      )}
    </>
  );
}

