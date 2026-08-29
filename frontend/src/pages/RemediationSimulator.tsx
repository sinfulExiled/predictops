import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, IncidentReport, PlanStep, SimArm } from "../api";
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

export default function RemediationSimulator() {
  const fleet = useAsync(() => api.machines(), []);
  const [selected, setSelected] = useState("");
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [approved, setApproved] = useState<Record<string, boolean>>({});

  useEffect(() => {
    if (!selected && fleet.data?.machines?.length) {
      const risky = fleet.data.machines.find((m) => m.status === "high");
      setSelected((risky ?? fleet.data.machines[0]).machine_id);
    }
  }, [fleet.data, selected]);

  async function run() {
    setBusy(true);
    setErr("");
    setReport(null);
    setApproved({});
    try {
      setReport(await api.incident(selected, fleet.data?.timestamp));
    } catch (e: any) {
      setErr(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  if (fleet.loading) return <Loading what="machines" />;
  if (fleet.error) return <ErrorNote error={fleet.error} />;

  const sim = report?.simulation;
  const rem = report?.remediation;
  const control = sim?.no_action?.failure_probability_simulated;

  const chart =
    sim && control != null
      ? [
          { name: "No action", value: control, kind: "control" },
          ...(sim.arms as SimArm[])
            .filter((a) => a.simulated)
            .map((a) => ({
              name: a.title,
              value: a.failure_probability_simulated as number,
              kind: "action",
            })),
        ]
      : [];

  return (
    <>
      <PageHead
        title="Remediation Simulator"
        blurb="Every proposed action is rolled forward against a do-nothing control under identical assumptions, then re-scored by the trained model. Nothing here actuates a machine."
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
          <button onClick={run} disabled={busy || !selected}>
            {busy ? "Simulating…" : "Plan and simulate"}
          </button>
        </div>
      </Panel>

      {err && <ErrorNote error={err} />}

      {report && sim && rem && (
        <>
          <div className="grid c4" style={{ marginBottom: 18 }}>
            <Stat label="Risk now" value={pct(sim.probability_now)} tone="bad" />
            <Stat
              label={`No action, +${sim.horizon_hours} h`}
              value={pct(control)}
              tone="bad"
              sub="simulated control arm"
            />
            <Stat
              label="Best action"
              value={
                sim.best_by_risk
                  ? pct(
                      (sim.arms as SimArm[]).find(
                        (a) => a.intervention_id === sim.best_by_risk,
                      )?.failure_probability_simulated,
                    )
                  : "—"
              }
              tone="ok"
              sub={sim.best_by_risk ?? ""}
            />
            <Stat
              label="Best value for money"
              value={<span style={{ fontSize: 15 }}>{sim.best_by_value ?? "—"}</span>}
              sub="largest risk drop per $1k"
            />
          </div>

          <ContestedGate
            adjudication={report.adjudication}
            degradation={report.degradation_case}
            confound={report.confound_case}
          />

          <div className="banner warn">
            <strong>Simulated, not forecast.</strong> {sim.caveat}
          </div>

          <Panel title="Counterfactual risk by action">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={chart} margin={{ bottom: 60 }}>
                <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
                <XAxis
                  dataKey="name"
                  stroke="#8b98a5"
                  fontSize={11}
                  angle={-28}
                  textAnchor="end"
                  interval={0}
                  height={80}
                />
                <YAxis
                  stroke="#8b98a5"
                  fontSize={11}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <Tooltip
                  formatter={(v: any) => pct(v)}
                  contentStyle={{
                    background: "#151b23",
                    border: "1px solid #2a3441",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {chart.map((c, i) => (
                    <Cell key={i} fill={c.kind === "control" ? "#8b98a5" : "#4a9eff"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Panel>

          <Panel
            title="Proposed plan — requires human approval"
            right={
              <span className="small muted">
                {rem.approval_gate?.status ?? "none"}
              </span>
            }
          >
            {(rem.plan as PlanStep[]).map((s) => {
              const arm = (sim.arms as SimArm[]).find(
                (a) => a.intervention_id === s.intervention_id,
              );
              return (
                <div className="step" key={s.intervention_id}>
                  <div className="head">
                    <div>
                      <strong>
                        {s.order}. {s.title}
                      </strong>{" "}
                      <span className="tag">{s.intervention_id}</span>{" "}
                      <Pill kind={s.risk === "high" ? "fail" : s.risk === "medium" ? "warn" : "pass"}>
                        {s.is_diagnostic ? "diagnostic" : `${s.risk} risk`}
                      </Pill>
                    </div>
                    <div className="small muted">
                      ${s.cost_usd.toLocaleString()}
                      {s.downtime_hours > 0 && ` · ${s.downtime_hours} h downtime`}
                    </div>
                  </div>
                  <div className="small">{s.why}</div>
                  <div className="small muted" style={{ marginTop: 4 }}>
                    {s.detail}
                  </div>
                  {s.preconditions.map((p) => (
                    <div className="small muted" key={p}>
                      • requires: {p}
                    </div>
                  ))}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginTop: 10,
                    }}
                  >
                    <span className="small">
                      {arm?.simulated
                        ? `simulated risk ${pct(arm.failure_probability_simulated)} (${num(
                            (arm.delta_vs_no_action ?? 0) * 100,
                            1,
                          )} pp vs no action)`
                        : "no telemetry effect to simulate"}
                    </span>
                    {s.requires_approval && (
                      <button
                        className={approved[s.intervention_id] ? "ghost" : ""}
                        onClick={() =>
                          setApproved((a) => ({
                            ...a,
                            [s.intervention_id]: !a[s.intervention_id],
                          }))
                        }
                      >
                        {approved[s.intervention_id]
                          ? "✓ Approved (simulation only)"
                          : "Approve"}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
            <div className="small muted" style={{ marginTop: 10 }}>
              {rem.approval_gate?.statement}
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
