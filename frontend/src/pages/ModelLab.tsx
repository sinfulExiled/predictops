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
import { AblationStudy, ThresholdBands } from "../components/Ablation";
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

export default function ModelLab() {
  const exp = useAsync(() => api.experiments(), []);
  const health = useAsync(() => api.health(), []);
  const thresholds = useAsync(() => api.thresholds().catch(() => null), []);
  const ablation = useAsync(() => api.ablation().catch(() => null), []);

  if (exp.loading) return <Loading what="model results" />;
  if (exp.error) return <ErrorNote error={exp.error} />;

  const rows = exp.data?.experiments ?? [];
  const chart = rows.map((r) => ({
    name: r.model === "threshold_baseline" ? "threshold" : `${r.model}/${r.feature_set.slice(0, 4)}`,
    val: r.metrics?.val?.pr_auc ?? 0,
    f1: r.metrics?.row?.f1 ?? 0,
    decision: r.decision,
  }));

  const best = rows.reduce(
    (a, b) => ((b.metrics?.row?.f1 ?? 0) > (a.metrics?.row?.f1 ?? 0) ? b : a),
    rows[0],
  );
  const baseline = rows.find((r) => r.model === "threshold_baseline");
  const model = health.data?.model;

  return (
    <>
      <PageHead
        title="Model Lab"
        blurb="Every candidate the research agent trained, on identical data and an identical evaluation set. Selection is on validation PR-AUC; test F1 is reported at the frozen threshold and never used to choose."
      />

      {model && (
        <div className="banner info">
          <strong>In production: {model.kind} on {model.feature_set} features.</strong>{" "}
          {model.rationale}
        </div>
      )}

      <div className="grid c4" style={{ marginBottom: 18 }}>
        <Stat
          label="Baseline test F1"
          value={num(baseline?.metrics?.row?.f1, 3)}
          sub="threshold rule"
        />
        <Stat
          label="Best test F1"
          value={num(best?.metrics?.row?.f1, 3)}
          tone="ok"
          sub={best?.name}
        />
        <Stat
          label="Improvement"
          value={
            baseline && best
              ? `+${(
                  ((best.metrics.row.f1 - baseline.metrics.row.f1) /
                    baseline.metrics.row.f1) *
                  100
                ).toFixed(0)}%`
              : "—"
          }
          tone="ok"
          sub="relative to baseline"
        />
        <Stat
          label="Candidates trained"
          value={rows.length}
          sub={`${rows.reduce((s, r) => s + r.duration_s, 0) / 60 | 0} min compute`}
        />
      </div>

      <ThresholdBands t={thresholds.data} />

      <Panel title="Validation PR-AUC (selection metric) vs test F1 (reported)">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chart} margin={{ bottom: 50 }}>
            <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
            <XAxis
              dataKey="name"
              stroke="#8b98a5"
              fontSize={11}
              angle={-25}
              textAnchor="end"
              interval={0}
              height={70}
            />
            <YAxis stroke="#8b98a5" fontSize={11} domain={[0, 0.8]} />
            <Tooltip
              contentStyle={{
                background: "#151b23",
                border: "1px solid #2a3441",
                borderRadius: 6,
                fontSize: 12,
              }}
            />
            <Bar dataKey="val" name="val PR-AUC" radius={[3, 3, 0, 0]}>
              {chart.map((c, i) => (
                <Cell key={i} fill={c.decision === "removed" ? "#3a4653" : "#4a9eff"} />
              ))}
            </Bar>
            <Bar dataKey="f1" name="test F1" fill="#3fb950" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="All candidates">
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
              return (
                <tr key={r.id}>
                  <td className="muted">{r.stage}</td>
                  <td>{r.name}</td>
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
                  <td className="num">{num(ev.false_alarms_per_machine_day, 2)}</td>
                  <td>
                    <Pill kind={r.decision}>{r.decision}</Pill>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>

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
        {best?.metrics?.event?.detection_by_type ? (
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
              {Object.entries(best.metrics.event.detection_by_type).map(
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
