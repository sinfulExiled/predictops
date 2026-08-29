import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, Evidence, Hypothesis, IncidentReport } from "../api";
import Contest from "../components/Contest";
import ContestedGate from "../components/ContestedGate";
import {
  ErrorNote,
  Loading,
  PageHead,
  Panel,
  Pill,
  Stat,
  num,
  pct,
  useAsync,
} from "../components/common";

const SERIES = [
  { key: "vibration", color: "#f85149" },
  { key: "temperature", color: "#d29922" },
  { key: "current", color: "#4a9eff" },
  { key: "pressure", color: "#3fb950" },
  { key: "load", color: "#8b98a5" },
];

export default function MachineInvestigation() {
  const { machineId } = useParams();
  const fleet = useAsync(() => api.machines(), []);
  const [selected, setSelected] = useState<string>(machineId ?? "");
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!selected && fleet.data?.machines?.length) {
      setSelected(fleet.data.machines[0].machine_id);
    }
  }, [fleet.data, selected]);

  // The moment being investigated. A machine is only interesting at a
  // particular time, so this is explicit and shareable via the URL rather than
  // implicitly "now" -- and it defaults to whatever the fleet view was showing.
  const [params, setParams] = useSearchParams();
  const at = params.get("at") ?? fleet.data?.timestamp;

  const telemetry = useAsync(
    () =>
      selected
        ? api.telemetry(selected, 24, at)
        : Promise.resolve({ machine_id: "", series: [] }),
    [selected, at],
  );

  async function investigate() {
    if (!selected) return;
    setBusy(true);
    setErr("");
    setReport(null);
    try {
      setReport(await api.incident(selected, at));
    } catch (e: any) {
      setErr(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  if (fleet.loading) return <Loading what="machines" />;
  if (fleet.error) return <ErrorNote error={fleet.error} />;

  const series = (telemetry.data?.series ?? []).map((r: any) => ({
    ...r,
    t: String(r.timestamp).slice(5, 16),
  }));
  const pred = report?.prediction;
  const inv = report?.investigation;
  const ver = report?.verification;

  return (
    <>
      <PageHead
        title="Machine Investigation"
        blurb="Run the full agent workflow on one machine: score it, explain why, and check every claim against raw telemetry."
      />

      <Panel>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {(fleet.data?.machines ?? []).map((m) => (
              <option key={m.machine_id} value={m.machine_id}>
                {m.machine_id} — {m.failure_probability === null ? "down" : pct(m.failure_probability)}
              </option>
            ))}
          </select>
          <input
            type="text"
            style={{ width: 190 }}
            placeholder="YYYY-MM-DD HH:MM:SS"
            defaultValue={at}
            key={at}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                setParams({ at: (e.target as HTMLInputElement).value });
                setReport(null);
              }
            }}
          />
          <button onClick={investigate} disabled={busy || !selected}>
            {busy ? "Running agents…" : "Run investigation"}
          </button>
        </div>
        <div className="small muted" style={{ marginTop: 10 }}>
          prediction → context → facts → two advocates → adjudication →
          remediation → simulation → verification
        </div>
      </Panel>

      {err && <ErrorNote error={err} />}

      <Panel title={`Telemetry — last 24 h${selected ? ` — ${selected}` : ""}`}>
        {telemetry.loading ? (
          <Loading what="telemetry" />
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={series}>
              <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
              <XAxis dataKey="t" stroke="#8b98a5" fontSize={11} minTickGap={40} />
              <YAxis stroke="#8b98a5" fontSize={11} />
              <Tooltip
                contentStyle={{
                  background: "#151b23",
                  border: "1px solid #2a3441",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  stroke={s.color}
                  dot={false}
                  strokeWidth={1.6}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </Panel>

      {report && pred && inv && ver && (
        <>
          <div className="grid c4" style={{ marginBottom: 18 }}>
            <Stat
              label="Failure probability"
              value={pct(pred.failure_probability)}
              tone={pred.alert ? "bad" : "ok"}
              sub={`threshold ${num(pred.threshold, 3)}`}
            />
            <Stat
              label="Expected window"
              value={
                pred.prediction_window_hours?.eta_hours != null
                  ? `${num(pred.prediction_window_hours.window_low_h, 1)}–${num(
                      pred.prediction_window_hours.window_high_h,
                      1,
                    )} h`
                  : "—"
              }
              sub="model estimate"
            />
            <Stat
              label="Likely failure"
              value={
                <span style={{ fontSize: 16 }}>
                  {(inv.likely_failure_type ?? "—").replace(/_/g, " ")}
                </span>
              }
              sub={`classifier ${pct(pred.failure_type_confidence)}`}
            />
            <Stat
              label="Confidence"
              value={pct(pred.confidence)}
              sub="measured precision in this score band"
            />
          </div>

          <div
            className={`banner ${
              ver.verdict === "PASS" ? "info" : ver.verdict === "FAIL" ? "bad" : "warn"
            }`}
          >
            <strong>Verification: {ver.verdict}</strong> — {ver.headline}
            <div className="small" style={{ marginTop: 5 }}>
              {ver.action_guidance}
            </div>
          </div>

          <ContestedGate
            adjudication={report.adjudication}
            degradation={report.degradation_case}
            confound={report.confound_case}
          />

          <Contest
            degradation={report.degradation_case}
            confound={report.confound_case}
            adjudication={report.adjudication}
          />

          <div className="grid c2">
            <Panel title="Evidence">
              {(inv.evidence as Evidence[]).map((e) => (
                <div className="evidence" key={e.id}>
                  <div>
                    <span className="mono muted">[{e.id}]</span> {e.claim}
                  </div>
                  <div className="src">
                    {e.metric}({e.channel}) = {num(e.value, 2)} {e.unit} · {e.source}
                  </div>
                </div>
              ))}
              {inv.operating_context && (
                <div className="small muted" style={{ marginTop: 12 }}>
                  Operating context: load{" "}
                  {num(inv.operating_context.load_change_pct_3h, 1)}% / 3 h ·
                  ambient {num(inv.operating_context.ambient_change_c_3h, 1)} °C / 3 h
                  {inv.operating_context.load_stable
                    ? " — load is flat, so the movement is not duty-driven"
                    : " — load moved, symptoms may be duty-driven"}
                </div>
              )}
            </Panel>

            <Panel title="Hypotheses considered">
              <table>
                <thead>
                  <tr>
                    <th>Failure mode</th>
                    <th className="num">Score</th>
                    <th className="num">Signature</th>
                    <th className="num">Classifier</th>
                    <th className="num">History</th>
                  </tr>
                </thead>
                <tbody>
                  {(inv.ranked_hypotheses as Hypothesis[]).map((h) => (
                    <tr key={h.failure_type}>
                      <td>{h.failure_type.replace(/_/g, " ")}</td>
                      <td className="num">{num(h.score, 2)}</td>
                      <td className="num">{pct(h.signature_match, 0)}</td>
                      <td className="num">{pct(h.classifier_probability, 0)}</td>
                      <td className="num">{pct(h.historical_vote, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {inv.similar_past_failures?.length > 0 && (
                <div className="small muted" style={{ marginTop: 12 }}>
                  Nearest historical match:{" "}
                  {inv.similar_past_failures[0].failure_type} on{" "}
                  <span className="mono">{inv.similar_past_failures[0].machine_id}</span>
                </div>
              )}
            </Panel>
          </div>

          {report.context && (
            <Panel title="Machine dossier">
              <div className="grid c3">
                <div className="small">
                  <div className="muted">Type</div>
                  {report.context.machine_type}
                </div>
                <div className="small">
                  <div className="muted">Hours since service</div>
                  {num(report.context.operating_regime?.operating_hours_since_service, 0)} h
                  {report.context.in_run_in_period && (
                    <> <Pill kind="warn">in run-in</Pill></>
                  )}
                </div>
                <div className="small">
                  <div className="muted">Prior failures</div>
                  {report.context.prior_failures?.length ?? 0}
                  {report.context.recurring_mode &&
                    ` (recurring: ${report.context.recurring_mode.replace(/_/g, " ")})`}
                </div>
              </div>
              <ul className="small muted" style={{ marginTop: 12, paddingLeft: 18 }}>
                {(report.context.notes ?? []).map((n: string) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Panel>
          )}

          <Panel title="Verification checks">
            {ver.checks.map((c: any) => (
              <div className="check" key={c.id}>
                <Pill kind={c.status === "n/a" ? "down" : c.status}>
                  {c.status === "n/a" ? "n/a" : c.status}
                </Pill>
                <div>
                  <div>
                    <span className="mono muted">{c.id}</span> {c.check}
                  </div>
                  <div className="small muted">{c.detail}</div>
                </div>
              </div>
            ))}
          </Panel>
        </>
      )}
    </>
  );
}
