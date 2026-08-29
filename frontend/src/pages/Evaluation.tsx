import { api } from "../api";
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

const METRICS: { label: string; key: string; higherIsBetter: boolean }[] = [
  { label: "Alert accuracy (primary)", key: "alert_accuracy", higherIsBetter: true },
  { label: "F1", key: "f1", higherIsBetter: true },
  { label: "Precision", key: "precision", higherIsBetter: true },
  { label: "Recall", key: "recall", higherIsBetter: true },
  { label: "Cause accuracy", key: "cause_accuracy", higherIsBetter: true },
  { label: "Hard-case accuracy", key: "hard_case_accuracy", higherIsBetter: true },
  {
    label: "False alarms on nuisance cases",
    key: "false_alarm_rate_on_nuisance_cases",
    higherIsBetter: false,
  },
];

export default function Evaluation() {
  const ev = useAsync(() => api.evaluation(), []);

  if (ev.loading) return <Loading what="evaluation results" />;
  if (ev.error)
    return (
      <>
        <PageHead
          title="Evaluation"
          blurb="Baseline versus agent on the same fixed scenario suite."
        />
        <div className="banner warn">
          No evaluation on disk yet. Run <code>python evaluate.py</code>, then
          reload.
        </div>
      </>
    );

  const d = ev.data;
  const base = d.baseline;
  const agent = d.agent;
  const cases = d.cases?.agent ?? [];
  const baseCases: any[] = d.cases?.baseline ?? [];
  const byId = new Map(baseCases.map((c) => [c.scenario_id, c]));

  return (
    <>
      <PageHead
        title="Evaluation"
        blurb={`${d.suite.n_cases} fixed scenarios from the held-out test period — ${d.suite.n_positive} real warning windows and ${d.suite.n_negative} nuisance cases. Both systems saw exactly the same cases and the same telemetry.`}
      />

      <div className="grid c4" style={{ marginBottom: 18 }}>
        <Stat label="Scenarios" value={d.suite.n_cases} sub={`${d.suite.by_difficulty?.hard ?? 0} rated hard`} />
        <Stat
          label="Baseline accuracy"
          value={pct(base.alert_accuracy)}
          sub="threshold rule"
        />
        <Stat
          label="Agent accuracy"
          value={pct(agent.alert_accuracy)}
          tone="ok"
          sub="full workflow"
        />
        <Stat
          label="False alarms cut"
          value={`${(
            (base.false_alarm_rate_on_nuisance_cases -
              agent.false_alarm_rate_on_nuisance_cases) *
            100
          ).toFixed(0)} pp`}
          tone="ok"
          sub="on nuisance cases"
        />
      </div>

      <Panel title="Simple baseline vs agent solution — same cases, same telemetry">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th className="num">Simple baseline</th>
              <th className="num">Agent solution</th>
              <th className="num">Change</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => {
              const b = base[m.key];
              const a = agent[m.key];
              if (b == null && a == null) return null;
              const delta = a != null && b != null ? (a - b) * 100 : null;
              const good =
                delta == null
                  ? undefined
                  : m.higherIsBetter
                    ? delta > 0
                    : delta < 0;
              return (
                <tr key={m.key}>
                  <td>{m.label}</td>
                  <td className="num">{pct(b)}</td>
                  <td className="num">
                    <strong>{pct(a)}</strong>
                  </td>
                  <td
                    className="num"
                    style={{
                      color:
                        good === undefined
                          ? undefined
                          : good
                            ? "var(--ok)"
                            : "var(--bad)",
                    }}
                  >
                    {delta == null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`}
                  </td>
                </tr>
              );
            })}
            <tr>
              <td>Seconds per case</td>
              <td className="num">{num(base.median_seconds_per_case, 3)}</td>
              <td className="num">{num(agent.median_seconds_per_case, 3)}</td>
              <td className="num muted">—</td>
            </tr>
          </tbody>
        </table>
        <div className="small muted" style={{ marginTop: 12 }}>
          The baseline is given a cause-attribution rule too (the tripped
          channel mapped to the mode it most often signals), so cause accuracy
          is a real comparison rather than a category it cannot score in.
        </div>
      </Panel>

      <div className="grid c2">
        <Panel title="Capabilities the baseline does not have">
          <table>
            <tbody>
              <tr>
                <td>Verification verdicts</td>
                <td className="num">
                  {Object.entries(agent.verification_verdicts ?? {}).map(
                    ([k, v]: any) => (
                      <span key={k} style={{ marginLeft: 8 }}>
                        <Pill kind={k.startsWith("PASS") ? "pass" : "fail"}>
                          {k} {v}
                        </Pill>
                      </span>
                    ),
                  )}
                </td>
              </tr>
              <tr>
                <td>Cases cleared to act on</td>
                <td className="num">
                  {agent.cases_cleared_to_act} / {d.suite.n_cases}
                </td>
              </tr>
              <tr>
                <td>Mean simulated risk reduction</td>
                <td className="num">{num(agent.mean_simulated_risk_reduction, 3)}</td>
              </tr>
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 10 }}>
            Reported, not scored as zero — a threshold rule was never asked
            these questions.
          </div>
        </Panel>

        <Panel title="Accuracy by scenario category">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th className="num">n</th>
                <th className="num">Baseline</th>
                <th className="num">Agent</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(agent.by_category)
                .sort()
                .map(([cat, v]: any) => (
                  <tr key={cat}>
                    <td>{cat.replace(/_/g, " ")}</td>
                    <td className="num">{v.n}</td>
                    <td className="num">
                      {pct(base.by_category?.[cat]?.accuracy, 0)}
                    </td>
                    <td className="num">
                      <strong>{pct(v.accuracy, 0)}</strong>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </Panel>
      </div>

      <Panel title="Every case">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Category</th>
              <th>Difficulty</th>
              <th>Machine</th>
              <th className="num">Expected</th>
              <th className="num">Baseline</th>
              <th className="num">Agent</th>
              <th className="num">Risk</th>
              <th>Cause</th>
              <th>Verification</th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c: any) => {
              const b = byId.get(c.scenario_id);
              return (
                <tr key={c.scenario_id}>
                  <td className="mono">{c.scenario_id}</td>
                  <td className="small">{c.category.replace(/_/g, " ")}</td>
                  <td>
                    <span className="tag">{c.difficulty}</span>
                  </td>
                  <td className="mono small">{c.machine_id}</td>
                  <td className="num">{c.expected_alert ? "alert" : "quiet"}</td>
                  <td
                    className="num"
                    style={{ color: b?.correct_alert ? "var(--ok)" : "var(--bad)" }}
                  >
                    {b?.alert ? "alert" : "quiet"}
                  </td>
                  <td
                    className="num"
                    style={{ color: c.correct_alert ? "var(--ok)" : "var(--bad)" }}
                  >
                    {c.alert ? "alert" : "quiet"}
                  </td>
                  <td className="num">{pct(c.probability)}</td>
                  <td className="small">
                    {c.predicted_type ? (
                      <span
                        style={{
                          color:
                            c.correct_type === true
                              ? "var(--ok)"
                              : c.correct_type === false
                                ? "var(--bad)"
                                : undefined,
                        }}
                      >
                        {c.predicted_type.replace(/_/g, " ")}
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {c.verification && (
                      <Pill kind={c.verification.startsWith("PASS") ? "pass" : "fail"}>
                        {c.verification.replace("PASS_WITH_WARNINGS", "PASS+W")}
                      </Pill>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
