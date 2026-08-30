import { useRef } from "react";
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
import { api } from "../api";
import { AblationStudy } from "../components/Ablation";
import {
  ErrorNote,
  Loading,
  Panel,
  Pill,
  num,
  pct,
  useAsync,
} from "../components/common";
import { Icon } from "../components/icons";

/** Short axis label for a candidate, e.g. "xgboost / engineered". */
function shortName(r: any) {
  if (r.model === "threshold_baseline") return ["threshold", "rule"];
  return [r.model, r.feature_set];
}

function KpiCard({
  label,
  value,
  sub,
  icon,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  sub: string;
  icon: keyof typeof Icon;
  tone: string;
}) {
  const I = Icon[icon];
  return (
    <div className="kpi">
      <span className="kpi-icon" style={{ color: tone, background: `${tone}22` }}>
        <I />
      </span>
      <div style={{ minWidth: 0 }}>
        <div className="kpi-label">{label}</div>
        <div className="kpi-value" style={{ color: tone }}>
          {value}
        </div>
        <div className="kpi-sub">{sub}</div>
      </div>
    </div>
  );
}

/** The three bands the model service reports, drawn to true scale.
 *
 *  Deliberately proportional: with a validation-tuned alert threshold of
 *  0.987 the ALERT band really is a sliver, and that is the point -- the
 *  service only calls an alert on near-certainty. Labels move outside a band
 *  that is too narrow to hold them rather than the bar being redrawn to a
 *  flattering scale.
 */
function ServiceContract({ t }: { t: any }) {
  if (!t) return null;
  const inv = t.investigate_threshold;
  const alert = t.alert_threshold;
  const bands = [
    { key: "NORMAL", from: 0, to: inv, color: "#3fb950", note: "no action" },
    { key: "INVESTIGATE", from: inv, to: alert, color: "#d29922", note: "agents look" },
    { key: "ALERT", from: alert, to: 1, color: "#f85149", note: "plan a fix" },
  ];
  return (
    <Panel
      title="Model service contract"
      right={
        <span className="small muted">
          drawn to scale · lookback {t.lookback_steps} steps
        </span>
      }
    >
      <div className="band-track">
        {bands.map((b) => {
          const w = (b.to - b.from) * 100;
          return (
            <div
              key={b.key}
              className="band-seg"
              style={{ width: `${w}%`, background: `${b.color}59` }}
              title={`${b.key}: ${(b.from * 100).toFixed(1)}%–${(b.to * 100).toFixed(1)}%`}
            >
              {w > 14 && <span className="band-name">{b.key}</span>}
            </div>
          );
        })}
      </div>

      <div className="band-ticks">
        {(() => {
          // Threshold ticks carry the information, so they are placed first;
          // an endpoint is dropped rather than allowed to overlap one. With a
          // 0.987 alert cut the "100%" label would otherwise sit on top of it.
          const kept: { at: number; text: string; end?: "l" | "r" }[] = [
            { at: inv * 100, text: pct(inv, 1) },
            { at: alert * 100, text: pct(alert, 1) },
          ];
          for (const e of [
            { at: 0, text: "0%", end: "l" as const },
            { at: 100, text: "100%", end: "r" as const },
          ]) {
            if (kept.every((k) => Math.abs(k.at - e.at) > 6)) kept.push(e);
          }
          return kept.map((k) => (
            <span
              key={k.text}
              style={{
                left: `${k.at}%`,
                transform:
                  k.end === "l"
                    ? "none"
                    : k.end === "r"
                      ? "translateX(-100%)"
                      : "translateX(-50%)",
              }}
            >
              {k.text}
            </span>
          ));
        })()}
      </div>

      <div className="band-key">
        {bands.map((b) => (
          <div key={b.key} className="band-key-row">
            <span className="band-dot" style={{ background: b.color }} />
            <strong style={{ color: b.color }}>{b.key}</strong>
            <span className="mono small">
              {pct(b.from, 1)} – {pct(b.to, 1)}
            </span>
            <span className="small muted">{b.note}</span>
            <span className="small muted band-width">
              {((b.to - b.from) * 100).toFixed(1)} pts wide
            </span>
          </div>
        ))}
      </div>

      <p className="small muted" style={{ margin: "12px 0 0" }}>
        The model service answers the quantitative question and nothing else.
        The orchestration layer decides which workflow an answer triggers, so no
        agent ever has to ask "should we investigate?". Both cut points were
        tuned on the validation split and frozen before the test set was read.
      </p>
    </Panel>
  );
}

export default function ModelLab() {
  const exp = useAsync(() => api.experiments(), []);
  const health = useAsync(() => api.health(), []);
  const thresholds = useAsync(() => api.thresholds().catch(() => null), []);
  const ablation = useAsync(() => api.ablation().catch(() => null), []);
  const tableRef = useRef<HTMLDivElement>(null);

  if (exp.loading) return <Loading what="model results" />;
  if (exp.error) return <ErrorNote error={exp.error} />;

  const rows = exp.data?.experiments ?? [];
  const model = health.data?.model;

  const baseline = rows.find((r) => r.model === "threshold_baseline");
  const bestF1 = rows.reduce(
    (a, b) => ((b.metrics?.row?.f1 ?? 0) > (a.metrics?.row?.f1 ?? 0) ? b : a),
    rows[0],
  );
  // The deployed model is the one selection actually chose, on validation
  // PR-AUC -- not whichever candidate happens to top test F1. Derived from the
  // recorded metrics rather than read off /api/health, which reports the
  // bundle's kind but not which candidate row produced it.
  const champion = rows.reduce(
    (a, b) => ((b.metrics?.val?.pr_auc ?? 0) > (a.metrics?.val?.pr_auc ?? 0) ? b : a),
    rows[0],
  );

  const chart = rows.map((r) => {
    const [a, b] = shortName(r);
    return {
      name: a,
      sub: b,
      val: r.metrics?.val?.pr_auc ?? 0,
      f1: r.metrics?.row?.f1 ?? 0,
      decision: r.decision,
      selected: r.id === champion?.id,
    };
  });

  const b0 = baseline?.metrics?.row?.f1;
  const bf = bestF1?.metrics?.row?.f1;
  const improvement = b0 && bf ? ((bf - b0) / b0) * 100 : null;
  const minutes = rows.reduce((s, r) => s + (r.duration_s ?? 0), 0) / 60;

  // Operational counterpoint: the candidate that caught the most real failure
  // events is not necessarily the one the selection metric picked. Computed,
  // never assumed -- if they coincide the note does not render.
  const detected = (r: any) => r?.metrics?.event?.detected ?? -1;
  const bestEvents = rows.reduce(
    (a, b) => (detected(b) > detected(a) ? b : a),
    rows[0],
  );
  const disagrees =
    bestEvents && champion && bestEvents.id !== champion.id &&
    detected(bestEvents) > detected(champion);

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.lab />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Model Lab
            </h2>
            <p
              style={{
                margin: "4px 0 0",
                color: "var(--muted)",
                fontSize: 13.5,
                maxWidth: 780,
              }}
            >
              Every candidate the research agent trained, on identical data and
              an identical evaluation set. Selection is on{" "}
              <span style={{ color: "var(--accent)" }}>validation PR-AUC</span>;{" "}
              <span style={{ color: "var(--ok)" }}>test F1</span> is reported at
              the frozen threshold and never used to choose.
            </p>
          </div>
        </div>
        <div className="topbar-right">
          <div className="run-chip">
            <Icon.layers />
            <span>
              run <strong>{exp.data?.run_id ?? "—"}</strong>
            </span>
            <span className="muted">· {rows.length} candidates</span>
          </div>
          <button
            className="ghost"
            onClick={() => window.location.reload()}
            style={{ display: "inline-flex", alignItems: "center", gap: 7 }}
          >
            <Icon.refresh />
            Refresh
          </button>
        </div>
      </div>

      {/* ---------------- in production ---------------- */}
      {model && (
        <div className="prod-banner">
          <span className="prod-icon">
            <Icon.database />
          </span>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 14.5, fontWeight: 600 }}>
              In production:{" "}
              <span style={{ color: "var(--accent)" }}>
                {model.kind} on {model.feature_set} features
              </span>
            </div>
            <div
              className="small"
              style={{ color: "var(--muted)", marginTop: 5, lineHeight: 1.55 }}
            >
              {model.rationale}
            </div>
          </div>
          <div className="champion">
            <span className="champion-icon">
              <Icon.rocket />
            </span>
            <div style={{ minWidth: 0 }}>
              <div className="champion-label">Deployed model</div>
              <div className="champion-name">{champion?.name ?? model.kind}</div>
              <div className="small muted">
                chosen on val PR-AUC {num(champion?.metrics?.val?.pr_auc, 4)}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ---------------- KPIs ---------------- */}
      <div className="kpi-row">
        <KpiCard
          icon="target"
          tone="#a371f7"
          label="Baseline test F1"
          value={num(b0, 3)}
          sub="threshold rule the plant runs today"
        />
        <KpiCard
          icon="trend"
          tone="#4a9eff"
          label="Best test F1"
          value={num(bf, 3)}
          sub={bestF1?.name ?? "—"}
        />
        <KpiCard
          icon="check"
          tone="#3fb950"
          label="Improvement"
          value={improvement == null ? "—" : `+${improvement.toFixed(0)}%`}
          sub="relative to baseline, same eval rows"
        />
        <KpiCard
          icon="beaker"
          tone="#f0883e"
          label="Candidates trained"
          value={rows.length}
          sub={`${minutes.toFixed(1)} min total compute`}
        />
      </div>

      <ServiceContract t={thresholds.data} />

      {/* ---------------- bake-off ---------------- */}
      <Panel
        title="Validation PR-AUC (selection metric) vs test F1 (reported)"
        right={
          <div className="chart-legend">
            <span>
              <i style={{ background: "#4a9eff" }} /> Validation PR-AUC
            </span>
            <span>
              <i style={{ background: "#3fb950" }} /> Test F1
            </span>
            <span>
              <i style={{ background: "#3a4653" }} /> Removed
            </span>
          </div>
        }
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chart} margin={{ top: 18, bottom: 34, left: -14 }} barGap={3}>
            <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="#8b98a5"
              fontSize={11.5}
              tickLine={false}
              interval={0}
              height={46}
              tick={({ x, y, payload, index }: any) => (
                <g transform={`translate(${x},${y + 12})`}>
                  <text textAnchor="middle" fill="#c9d4e0" fontSize={11.5}>
                    {payload.value}
                  </text>
                  <text textAnchor="middle" y={14} fill="#8b98a5" fontSize={10.5}>
                    {chart[index]?.sub}
                  </text>
                </g>
              )}
            />
            <YAxis
              stroke="#8b98a5"
              fontSize={11}
              domain={[0, 0.7]}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,.035)" }}
              contentStyle={{
                background: "#151b23",
                border: "1px solid #2a3441",
                borderRadius: 6,
                fontSize: 12,
              }}
              formatter={(v: any, k: any) => [
                Number(v).toFixed(4),
                k === "val" ? "val PR-AUC" : "test F1",
              ]}
              labelFormatter={(l: any, p: any) =>
                `${l} / ${p?.[0]?.payload?.sub ?? ""}`
              }
            />
            <Bar dataKey="val" name="val" radius={[3, 3, 0, 0]} minPointSize={2}>
              {chart.map((c, i) => (
                <Cell
                  key={i}
                  fill={c.decision === "removed" ? "#3a4653" : "#4a9eff"}
                  stroke={c.selected ? "#e6edf3" : undefined}
                  strokeWidth={c.selected ? 1.5 : 0}
                />
              ))}
            </Bar>
            <Bar dataKey="f1" name="f1" radius={[3, 3, 0, 0]} minPointSize={2}>
              {chart.map((c, i) => (
                <Cell
                  key={i}
                  fill={c.decision === "removed" ? "#2f4a36" : "#3fb950"}
                  stroke={c.selected ? "#e6edf3" : undefined}
                  strokeWidth={c.selected ? 1.5 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="chart-foot">
          <span className="small muted">
            Higher is better for both. The outlined pair is the deployed model.
            Greyed candidates were run and then removed — they are shown because
            a bake-off that hides its losers is not a bake-off.
          </span>
          <button
            className="ghost"
            onClick={() =>
              tableRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
            }
            style={{ display: "inline-flex", alignItems: "center", gap: 7 }}
          >
            View all candidates
            <Icon.arrow />
          </button>
        </div>
      </Panel>

      {/* -------- the honest counterpoint, only when the data shows it ------- */}
      {disagrees && (
        <div className="claim" style={{ borderColor: "rgba(210,153,34,.4)" }}>
          <div className="claim-icon" style={{ color: "var(--warn)", background: "rgba(210,153,34,.14)" }}>
            <Icon.alert />
          </div>
          <div>
            <div className="claim-title" style={{ color: "var(--warn)" }}>
              The selection metric and the operational metric disagree.
            </div>
            <div className="small" style={{ marginTop: 5, color: "var(--muted)", lineHeight: 1.6 }}>
              <strong style={{ color: "var(--text)" }}>{bestEvents.name}</strong>{" "}
              caught {detected(bestEvents)} of{" "}
              {bestEvents.metrics.event.n_events} real failure events, against{" "}
              {detected(champion)} for the deployed{" "}
              <strong style={{ color: "var(--text)" }}>{champion.name}</strong> —
              but it scored lower on validation PR-AUC (
              {num(bestEvents.metrics?.val?.pr_auc, 4)} vs{" "}
              {num(champion.metrics?.val?.pr_auc, 4)}), so the rule fixed before
              the run did not pick it. PR-AUC scores every ten-minute row as an
              independent question; a planner is asked one question per failure.
              The model was <strong style={{ color: "var(--text)" }}>not</strong>{" "}
              swapped after the fact — changing the metric to favour a known
              outcome is the failure this bake-off exists to prevent. The fix is
              a metric declared up front, next run.
            </div>
          </div>
        </div>
      )}

      {/* ---------------- all candidates ---------------- */}
      <div ref={tableRef}>
        <Panel title="All candidates">
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Model</th>
                  <th className="num">Val PR-AUC</th>
                  <th className="num">Test F1</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                  <th className="num">Events caught</th>
                  <th className="num">Early warning</th>
                  <th className="num">False alarms/day</th>
                  <th>Decision</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const ev = r.metrics?.event ?? {};
                  const isChamp = r.id === champion?.id;
                  return (
                    <tr key={r.id} className={isChamp ? "row-champion" : undefined}>
                      <td className="muted">{r.stage}</td>
                      <td>
                        {r.name}
                        {isChamp && (
                          <span className="tag" style={{ marginLeft: 8 }}>
                            deployed
                          </span>
                        )}
                      </td>
                      <td className="num">{num(r.metrics?.val?.pr_auc, 4)}</td>
                      <td className="num">
                        <strong>{num(r.metrics?.row?.f1, 4)}</strong>
                      </td>
                      <td className="num">{num(r.metrics?.row?.precision, 3)}</td>
                      <td className="num">{num(r.metrics?.row?.recall, 3)}</td>
                      <td className="num">
                        {ev.detected != null ? `${ev.detected}/${ev.n_events}` : "—"}
                      </td>
                      <td className="num">{num(ev.mean_early_warning_h, 2)} h</td>
                      <td className="num">
                        {num(ev.false_alarms_per_machine_day, 2)}
                      </td>
                      <td>
                        <Pill kind={r.decision}>{r.decision}</Pill>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* ---------------- ablation ---------------- */}
      <div className="page-head" style={{ marginTop: 34, marginBottom: 16 }}>
        <h2 style={{ fontSize: 18 }}>Ablation study</h2>
        <p>
          The research agent's job is not to make every change look good. It is
          to find out whether a change helps. This is what it found when asked
          about the hypothesis contest.
        </p>
      </div>

      <AblationStudy data={ablation.data} />

      <Panel title="Detection rate by failure mode — best model">
        {bestF1?.metrics?.event?.detection_by_type ? (
          <table>
            <thead>
              <tr>
                <th>Failure mode</th>
                <th className="num">Events</th>
                <th className="num">Detected</th>
                <th className="num">Rate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(bestF1.metrics.event.detection_by_type).map(
                ([k, v]: any) => (
                  <tr key={k}>
                    <td>{k.replace(/_/g, " ")}</td>
                    <td className="num">{v.n}</td>
                    <td className="num">{v.detected}</td>
                    <td className="num">{pct(v.rate, 0)}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        ) : (
          <span className="muted">no event breakdown recorded</span>
        )}
      </Panel>
    </>
  );
}
