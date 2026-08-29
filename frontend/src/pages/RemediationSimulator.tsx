import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, IncidentReport, PlanStep, SimArm } from "../api";
import ContestedGate from "../components/ContestedGate";
import { ErrorNote, Loading, Panel, Pill, num, pct, useAsync } from "../components/common";
import { Icon, Spark } from "../components/icons";

const money = (v: number) =>
  v >= 1000 ? `$${(v / 1000).toFixed(1)}k` : `$${v.toLocaleString()}`;

const RISK_PILL: Record<string, string> = {
  low: "pass",
  medium: "warn",
  high: "fail",
};

export default function RemediationSimulator() {
  const fleet = useAsync(() => api.fleetOverview(undefined, 6), []);
  const [selected, setSelected] = useState("");
  const [horizon, setHorizon] = useState(3);
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [approved, setApproved] = useState<Record<string, boolean>>({});
  const [how, setHow] = useState(false);

  useEffect(() => {
    if (!selected && fleet.data?.machines?.length) {
      const risky = fleet.data.machines.find((m: any) => m.status === "high");
      setSelected((risky ?? fleet.data.machines[0]).machine_id);
    }
  }, [fleet.data, selected]);

  async function run() {
    setBusy(true);
    setErr("");
    setReport(null);
    setApproved({});
    try {
      setReport(await api.incident(selected, fleet.data?.timestamp, horizon));
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
  const now = sim?.probability_now ?? 0;
  const control = sim?.no_action?.failure_probability_simulated;
  const arms: SimArm[] = (sim?.arms ?? []).filter((a: SimArm) => a.simulated);
  const best = arms.length
    ? arms.reduce((a, b) =>
        (a.failure_probability_simulated ?? 1) <= (b.failure_probability_simulated ?? 1) ? a : b,
      )
    : null;
  const value = arms.find((a) => a.intervention_id === sim?.best_by_value);

  const chart =
    sim && control != null
      ? [
          { name: "No action", value: control, kind: "control" },
          ...arms.map((a) => ({
            name: a.title,
            value: a.failure_probability_simulated as number,
            kind: "action",
          })),
        ]
      : [];

  const trendOf = (m: string) =>
    fleet.data?.machines?.find((x: any) => x.machine_id === m)?.trend ?? [];

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.wrench />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Remediation Simulator
            </h2>
            <p style={{ margin: "3px 0 0", color: "var(--muted)", fontSize: 13.5, maxWidth: 760 }}>
              Every proposed action is rolled forward against a do-nothing
              control under identical assumptions, then rescored by the trained
              model. Nothing here touches a machine.
            </p>
          </div>
        </div>
        <div className="topbar-right">
          <button className="ghost" onClick={() => setHow((v) => !v)}
                  style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
            <Icon.info />
            How it works
          </button>
        </div>
      </div>

      {how && (
        <Panel title="How the simulation works">
          <ol className="small" style={{ margin: 0, paddingLeft: 20, lineHeight: 1.75 }}>
            <li>
              The machine's recent telemetry is rolled forward{" "}
              <strong>{horizon} hours</strong> by continuing each channel's
              recent slope, damped — so the underlying degradation keeps
              developing.
            </li>
            <li>
              That same rollout runs once with <strong>no action</strong>. This
              is the control arm, and it is why a drop can be attributed to the
              action rather than to the passage of time.
            </li>
            <li>
              For each proposed action the catalogue's effect model is applied
              to the rollout, ramped in over the action's settle time.
            </li>
            <li>
              Features are recomputed on the counterfactual telemetry and the{" "}
              <strong>trained model rescores it</strong>. The number is a model
              output, not a rule about how much an action ought to help.
            </li>
          </ol>
          <div className="banner warn" style={{ marginTop: 14 }}>
            The effect coefficients come from the same physical relationships as
            the data generator, so treat these as a way to <strong>rank</strong>{" "}
            actions against each other — not as a forecast. On real telemetry
            they would have to be fitted from historical work orders.
          </div>
        </Panel>
      )}

      {/* ---------------- controls ---------------- */}
      <div className="toolbar">
        <select value={selected} onChange={(e) => setSelected(e.target.value)}
                style={{ minWidth: 230 }}>
          {(fleet.data?.machines ?? []).map((m: any) => (
            <option key={m.machine_id} value={m.machine_id}>
              {m.machine_id} —{" "}
              {m.failure_probability === null ? "down" : pct(m.failure_probability)}
            </option>
          ))}
        </select>
        <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
          {[1, 2, 3, 6, 12].map((h) => (
            <option key={h} value={h}>
              Risk in {h} hour{h === 1 ? "" : "s"}
            </option>
          ))}
        </select>
        <button onClick={run} disabled={busy || !selected}>
          {busy ? "Simulating…" : "Plan and simulate"}
        </button>
        <span className="small muted">
          at {fleet.data?.timestamp} · re-runs the whole workflow
        </span>
      </div>

      {err && <ErrorNote error={err} />}

      {report && sim && rem && (
        <>
          {/* ---------------- metric cards ---------------- */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(215px, 1fr))",
              gap: 14,
              marginBottom: 18,
            }}
          >
            <div className="metric bad">
              <div className="label">Risk now</div>
              <div className="big" style={{ color: "var(--bad)" }}>{pct(now)}</div>
              <div className="delta">current failure risk</div>
              <div className="spark">
                <Spark values={trendOf(report.machine_id)} color="#f85149" />
              </div>
            </div>

            <div className="metric bad">
              <div className="label">No action · {sim.horizon_hours} h</div>
              <div className="big" style={{ color: "var(--bad)" }}>{pct(control)}</div>
              <div className="delta">simulated control arm</div>
              <div className="spark">
                <Spark values={[now, control ?? 0]} color="#f85149" />
              </div>
            </div>

            <div className="metric ok">
              <div className="label">Best action</div>
              <div className="big" style={{ color: "var(--ok)" }}>
                {best ? pct(best.failure_probability_simulated) : "—"}
              </div>
              <div className="delta">{best?.intervention_id ?? "nothing simulatable"}</div>
              <div className="spark">
                <Spark
                  values={[now, best?.failure_probability_simulated ?? now]}
                  color="#3fb950"
                />
              </div>
            </div>

            <div className="metric">
              <div className="label">Best value for money</div>
              <div className="big" style={{ fontSize: 21, lineHeight: 1.35 }}>
                {value?.title ?? "—"}
              </div>
              <div className="delta">
                {value && control != null
                  ? `${num(
                      ((control - (value.failure_probability_simulated ?? 0)) *
                        100),
                      1,
                    )} pts of risk for ${money(value.cost_usd ?? 0)}`
                  : "no costed action"}
              </div>
            </div>
          </div>

          <div className="claim" style={{ borderColor: "rgba(210,153,34,.32)",
                                          background: "linear-gradient(120deg, rgba(210,153,34,.10), var(--panel) 58%)" }}>
            <div className="claim-icon" style={{ borderColor: "rgba(210,153,34,.5)", color: "var(--warn)" }}>
              <Icon.alert />
            </div>
            <div>
              <div className="claim-title" style={{ color: "var(--warn)" }}>
                Simulated, not forecast.
              </div>
              <div className="small" style={{ marginTop: 4, color: "var(--muted)" }}>
                {sim.caveat}
              </div>
            </div>
          </div>

          <ContestedGate
            adjudication={report.adjudication}
            degradation={report.degradation_case}
            confound={report.confound_case}
          />

          {/* ---------------- chart ---------------- */}
          <Panel
            title={`Counterfactual risk by action — ${sim.horizon_hours} h ahead`}
            right={
              <span className="small muted">
                every bar is the trained model rescoring synthetic telemetry
              </span>
            }
          >
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chart} margin={{ top: 24, bottom: 4 }}>
                <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke="#8b98a5" fontSize={11}
                       angle={-15} textAnchor="end" interval={0} height={78} />
                <YAxis stroke="#8b98a5" fontSize={11} width={44}
                       domain={[0, 1]}
                       tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  formatter={(v: any) => pct(v)}
                  contentStyle={{
                    background: "#151b23",
                    border: "1px solid #2a3441",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                {/* minPointSize keeps a ~0% arm perceptible; the
                    printed label always carries the true value. */}
                <Bar dataKey="value" radius={[5, 5, 0, 0]} minPointSize={3}>
                  {chart.map((c, i) => (
                    <Cell key={i} fill={c.kind === "control" ? "#f85149" : "#4a9eff"} />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="top"
                    formatter={(v: any) => pct(v, 1)}
                    style={{ fill: "#e6edf3", fontSize: 12, fontWeight: 600 }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Panel>

          {/* ---------------- plan ---------------- */}
          <Panel
            title="Proposed plan — requires human approval"
            right={
              <Pill kind={rem.approval_gate?.required ? "warn" : "pass"}>
                {rem.approval_gate?.status ?? "none"}
              </Pill>
            }
          >
            {(rem.plan as PlanStep[]).map((s) => {
              const arm = arms.find((a) => a.intervention_id === s.intervention_id);
              const delta = arm?.delta_vs_no_action;
              return (
                <div className="plan-step" key={s.intervention_id}>
                  <div className="step-no">{s.order}</div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="step-head">
                      <strong>{s.title}</strong>
                      <span className="tag">{s.intervention_id}</span>
                      <Pill kind={s.is_diagnostic ? "reference" : RISK_PILL[s.risk]}>
                        {s.is_diagnostic ? "diagnostic" : `${s.risk} risk`}
                      </Pill>
                    </div>
                    <div className="small" style={{ marginTop: 5 }}>{s.why}</div>
                    <div className="small muted" style={{ marginTop: 6 }}>
                      {arm?.simulated ? (
                        <>
                          sim risk {pct(arm.failure_probability_simulated, 1)} (
                          {num((delta ?? 0) * 100, 1)} pts vs no action)
                        </>
                      ) : (
                        <>no telemetry change to simulate — this gathers information</>
                      )}
                      {s.downtime_hours > 0 && ` · ${s.downtime_hours} h downtime`}
                    </div>
                    {s.preconditions.map((p) => (
                      <div className="small muted" key={p}>• requires: {p}</div>
                    ))}
                  </div>
                  <div className="step-side">
                    <div className="small muted">Est. cost</div>
                    <div className="step-cost">{money(s.cost_usd)}</div>
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
                        {approved[s.intervention_id] ? "✓ Approved" : "Approve"}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}

            <div className="banner info" style={{ marginTop: 14 }}>
              <strong>Approval here is a record, not an actuation.</strong>{" "}
              {rem.approval_gate?.statement} Total if the whole plan is carried
              out: {money(rem.estimated_cost_usd ?? 0)} and{" "}
              {num(rem.estimated_downtime_hours, 1)} h of downtime.
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
