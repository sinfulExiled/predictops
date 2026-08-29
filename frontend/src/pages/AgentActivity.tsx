import { useEffect, useRef, useState } from "react";
import { api, TrajectoryStep } from "../api";
import {
  ErrorNote,
  Loading,
  PageHead,
  Panel,
  Pill,
  num,
  useAsync,
} from "../components/common";

const AGENT_COLOR: Record<string, string> = {
  data_scientist: "#4a9eff",
  model_researcher: "#a371f7",
  predictor: "#3fb950",
  investigator: "#d29922",
  remediation: "#f0883e",
  simulator: "#56d4dd",
  verifier: "#f85149",
};

export default function AgentActivity() {
  const [runId, setRunId] = useState<string | undefined>();
  const traj = useAsync(() => api.trajectories(runId), [runId]);
  const [live, setLive] = useState<TrajectoryStep[]>([]);
  const [wsState, setWsState] = useState("connecting");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/agent-activity`);
    wsRef.current = ws;
    ws.onopen = () => setWsState("live");
    ws.onclose = () => setWsState("disconnected");
    ws.onerror = () => setWsState("error");
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "agent_step") setLive((l) => [msg, ...l].slice(0, 60));
    };
    return () => ws.close();
  }, []);

  if (traj.loading) return <Loading what="agent trajectories" />;
  if (traj.error) return <ErrorNote error={traj.error} />;

  const steps = [...live, ...(traj.data?.steps ?? [])];

  return (
    <>
      <PageHead
        title="Agent Activity"
        blurb="Every agent execution is logged: what it did, why, which tools it called, how long it took, and any retries. These are the trajectories, straight from the registry."
      />

      <Panel>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span className="small muted">Run</span>
          <select value={runId ?? traj.data?.run_id ?? ""} onChange={(e) => setRunId(e.target.value)}>
            {(traj.data?.runs ?? []).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <Pill kind={wsState === "live" ? "pass" : "warn"}>
            websocket {wsState}
          </Pill>
          <span className="small muted">
            new steps stream in here as investigations run
          </span>
        </div>
      </Panel>

      {steps.length === 0 && (
        <div className="banner info">
          No agent steps recorded for this run yet. Run an investigation from
          the Machine Investigation page.
        </div>
      )}

      {steps.map((s, i) => (
        <div className="step" key={`${s.agent}-${s.step}-${i}`}>
          <div className="head">
            <div>
              <span
                style={{
                  color: AGENT_COLOR[s.agent] ?? "var(--text)",
                  fontWeight: 600,
                }}
              >
                {s.agent}
              </span>{" "}
              <span className="tag">step {s.step}</span>{" "}
              {s.retry_count > 0 && <Pill kind="warn">{s.retry_count} retry</Pill>}
              {s.verification && <Pill kind="pass">{s.verification.slice(0, 40)}</Pill>}
            </div>
            <span className="small muted">{num(s.duration_s, 2)}s</span>
          </div>
          <div style={{ marginBottom: 4 }}>
            <strong>Action.</strong> {s.action}
          </div>
          {s.reason && (
            <div className="small" style={{ marginBottom: 4 }}>
              <strong>Reason.</strong> {s.reason}
            </div>
          )}
          {s.input_summary && (
            <div className="small muted">Input: {s.input_summary}</div>
          )}
          {s.tools_used?.length > 0 && (
            <div style={{ marginTop: 7, display: "flex", gap: 6, flexWrap: "wrap" }}>
              {s.tools_used.map((t) => (
                <span className="tag" key={t}>
                  {t}
                </span>
              ))}
            </div>
          )}
          <details style={{ marginTop: 9 }}>
            <summary className="small muted" style={{ cursor: "pointer" }}>
              output
            </summary>
            <pre>{JSON.stringify(s.output, null, 2).slice(0, 4000)}</pre>
          </details>
        </div>
      ))}
    </>
  );
}
