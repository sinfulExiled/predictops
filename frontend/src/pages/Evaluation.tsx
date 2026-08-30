import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import {
  ErrorNote,
  Loading,
  PageHead,
  Panel,
  Pill,
  num,
  pageWindow,
  pct,
  useAsync,
} from "../components/common";
import { Icon } from "../components/icons";

type Row = {
  label: string;
  key: string;
  icon: keyof typeof Icon;
  tone: string;
  higherIsBetter: boolean;
};

const METRICS: Row[] = [
  { label: "Alert accuracy (primary)", key: "alert_accuracy", icon: "target", tone: "#a371f7", higherIsBetter: true },
  { label: "F1", key: "f1", icon: "graph", tone: "#4a9eff", higherIsBetter: true },
  { label: "Precision", key: "precision", icon: "eye", tone: "#3fb950", higherIsBetter: true },
  { label: "Recall", key: "recall", icon: "bell", tone: "#d29922", higherIsBetter: true },
  { label: "Cause accuracy (all real failures)", key: "cause_accuracy", icon: "check", tone: "#56d4dd", higherIsBetter: true },
  { label: "Cause accuracy (of those alerted)", key: "cause_accuracy_when_alerted", icon: "check", tone: "#56d4dd", higherIsBetter: true },
  { label: "Hard-case accuracy", key: "hard_case_accuracy", icon: "activity", tone: "#f0883e", higherIsBetter: true },
  { label: "False alarms on nuisance cases", key: "false_alarm_rate_on_nuisance_cases", icon: "alert", tone: "#f85149", higherIsBetter: false },
];

/** Which quadrant of the confusion matrix a case landed in. The dot shows it
 *  at a glance, which the alert columns alone do not: "quiet" is right on a
 *  nuisance case and wrong on a real one. */
const OUTCOME = {
  TP: { color: "#3fb950", label: "caught" },
  TN: { color: "#4a9eff", label: "correctly quiet" },
  FN: { color: "#d29922", label: "missed" },
  FP: { color: "#f85149", label: "false alarm" },
} as const;

function outcomeOf(c: any): keyof typeof OUTCOME {
  if (c.expected_alert) return c.alert ? "TP" : "FN";
  return c.alert ? "FP" : "TN";
}

const PRESETS: { k: string; label: string }[] = [
  { k: "all", label: "All scenarios" },
  { k: "positive", label: "Real failures only" },
  { k: "negative", label: "Nuisance only" },
  { k: "disagree", label: "Where the two disagree" },
  { k: "agent_wrong", label: "Agent wrong" },
  { k: "baseline_wrong", label: "Baseline wrong" },
];

function Kpi({ label, value, sub, icon, tone }: {
  label: string; value: React.ReactNode; sub: string;
  icon: keyof typeof Icon; tone: string;
}) {
  const I = Icon[icon];
  return (
    <div className="kpi">
      <span className="kpi-icon" style={{ color: tone, background: `${tone}22` }}>
        <I />
      </span>
      <div style={{ minWidth: 0 }}>
        <div className="kpi-label">{label}</div>
        <div className="kpi-value" style={{ color: tone }}>{value}</div>
        <div className="kpi-sub">{sub}</div>
      </div>
    </div>
  );
}

/** TP / FP / FN / TN for one system, as recorded by the scenario runner. */
function Confusion({ title, m, tone }: { title: string; m: any; tone: string }) {
  const cells = [
    { k: "TP", v: m.true_positives, note: "caught", good: true },
    { k: "FP", v: m.false_positives, note: "false alarm", good: false },
    { k: "FN", v: m.false_negatives, note: "missed", good: false },
    { k: "TN", v: m.true_negatives, note: "correctly quiet", good: true },
  ];
  return (
    <div className="cm">
      <div className="cm-title" style={{ color: tone }}>{title}</div>
      <div className="cm-grid">
        {cells.map((c) => (
          <div key={c.k} className={`cm-cell${c.good ? " good" : " bad"}`}>
            <div className="cm-k">{c.k}</div>
            <div className="cm-v">{c.v}</div>
            <div className="cm-note">{c.note}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Evaluation() {
  const ev = useAsync(() => api.evaluation(), []);
  const [showFilters, setShowFilters] = useState(false);
  const [cat, setCat] = useState("all");
  const [diff, setDiff] = useState("all");
  const [outcome, setOutcome] = useState("all");
  const [preset, setPreset] = useState("all");
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(15);

  const d: any = ev.data;
  const cases: any[] = d?.cases?.agent ?? [];
  const baseCases: any[] = d?.cases?.baseline ?? [];
  const byId = useMemo(
    () => new Map(baseCases.map((c) => [c.scenario_id, c])),
    [baseCases],
  );

  const shown = useMemo(
    () =>
      cases.filter((c) => {
        if (cat !== "all" && c.category !== cat) return false;
        if (diff !== "all" && c.difficulty !== diff) return false;
        const b: any = byId.get(c.scenario_id);
        for (const k of [preset, outcome]) {
          if (k === "positive" && !c.expected_alert) return false;
          if (k === "negative" && c.expected_alert) return false;
          if (k === "agent_wrong" && c.correct_alert) return false;
          if (k === "baseline_wrong" && b?.correct_alert) return false;
          if (k === "disagree" && !!b?.alert === !!c.alert) return false;
          if (k === "agent_better" && !(c.correct_alert && !b?.correct_alert))
            return false;
        }
        return true;
      }),
    [cases, byId, cat, diff, outcome, preset],
  );

  const pages = Math.max(1, Math.ceil(shown.length / perPage));
  const cur = Math.min(page, pages);
  const from = (cur - 1) * perPage;
  const pageRows = shown.slice(from, from + perPage);

  // Narrowing the set can strand the reader on a page that no longer exists.
  useEffect(() => setPage(1), [cat, diff, outcome, preset, perPage]);

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

  const base = d.baseline;
  const agent = d.agent;
  const categories = [...new Set(cases.map((c) => c.category))].sort();
  const nuisance = d.suite.n_negative;
  const filtered =
    cat !== "all" || diff !== "all" || outcome !== "all" || preset !== "all";

  function exportReport() {
    const blob = new Blob([JSON.stringify(d, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "evaluation-report.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.check />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Evaluation
            </h2>
            <p style={{ margin: "4px 0 0", color: "var(--muted)", fontSize: 13.5, maxWidth: 840 }}>
              {d.suite.n_cases} fixed scenarios from the held-out test period —{" "}
              {d.suite.n_positive} real warning windows and {d.suite.n_negative}{" "}
              nuisance cases. Both systems saw exactly the same cases and the
              same telemetry.
            </p>
          </div>
        </div>
        <div className="topbar-right">
          <button
            className={`ghost${filtered ? " accent" : ""}`}
            onClick={() => setShowFilters((v) => !v)}
          >
            <Icon.search />
            Filters
          </button>
          <button className="ghost accent" onClick={exportReport}>
            <Icon.download />
            Export report
          </button>
        </div>
      </div>

      {/* ---------------- KPIs ---------------- */}
      <div className="kpi-row">
        <Kpi
          icon="layers" tone="#a371f7" label="Scenarios"
          value={d.suite.n_cases}
          sub={`${d.suite.n_positive} real / ${d.suite.n_negative} nuisance · ${d.suite.by_difficulty?.hard ?? 0} hard`}
        />
        <Kpi
          icon="target" tone="#4a9eff" label="Baseline accuracy"
          value={pct(base.alert_accuracy)} sub="threshold rule"
        />
        <Kpi
          icon="trend" tone="#3fb950" label="Agent accuracy"
          value={pct(agent.alert_accuracy)} sub="full workflow"
        />
        {/* Not "false alarms cut". The agent fires on one nuisance case the
            baseline does not, so this number went UP; showing it as a saving
            would invert the sign of the only result that went against us. */}
        <Kpi
          icon="alert" tone="#f85149" label="False alarms on nuisance"
          value={pct(agent.false_alarm_rate_on_nuisance_cases)}
          sub={`${agent.false_positives} of ${nuisance} · baseline ${base.false_positives} of ${nuisance}`}
        />
      </div>

      {/* ---------------- head-to-head ---------------- */}
      <Panel title="Simple baseline vs agent solution — same cases, same telemetry">
        <div style={{ overflowX: "auto" }}>
          <table className="cmp">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num" style={{ color: "#4a9eff" }}>Simple baseline</th>
                <th className="num" style={{ color: "#3fb950" }}>Agent solution</th>
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
                  delta == null || Math.abs(delta) < 0.05
                    ? undefined
                    : m.higherIsBetter ? delta > 0 : delta < 0;
                const I = Icon[m.icon];
                return (
                  <tr key={m.key}>
                    <td>
                      <span className="cmp-metric">
                        <span className="cmp-icon" style={{ color: m.tone, background: `${m.tone}1f` }}>
                          <I />
                        </span>
                        {m.label}
                      </span>
                    </td>
                    <td className="num">{pct(b)}</td>
                    <td className="num"><strong>{pct(a)}</strong></td>
                    <td
                      className="num"
                      style={{
                        color: good === undefined ? "var(--muted)"
                          : good ? "var(--ok)" : "var(--bad)",
                      }}
                    >
                      {delta == null ? "—" : (
                        <>
                          {good === undefined ? "" : good ? "▲ " : "▼ "}
                          {delta > 0 ? "+" : ""}{delta.toFixed(1)} pp
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
              <tr>
                <td>
                  <span className="cmp-metric">
                    <span className="cmp-icon" style={{ color: "#8b98a5", background: "#8b98a51f" }}>
                      <Icon.clock />
                    </span>
                    Seconds per case
                  </span>
                </td>
                <td className="num">{num(base.median_seconds_per_case, 3)}</td>
                <td className="num">{num(agent.median_seconds_per_case, 3)}</td>
                <td className="num muted">—</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="claim" style={{ marginTop: 16, marginBottom: 0 }}>
          <div className="claim-icon"><Icon.info /></div>
          <div>
            <div className="claim-title">Two rows here rest on a single case.</div>
            <div className="small" style={{ marginTop: 4, color: "var(--muted)", lineHeight: 1.6 }}>
              Precision and false-alarms-on-nuisance move together because the
              agent fires on <strong>{agent.false_positives}</strong> of the{" "}
              {nuisance} nuisance scenarios and the baseline on{" "}
              <strong>{base.false_positives}</strong>. Rendered as percentages
              that reads as an {Math.abs(((agent.precision - base.precision) * 100)).toFixed(1)}-point
              precision regression, which is finer than a {d.suite.n_cases}-case
              suite can resolve. The baseline achieves it by alerting on very
              little at all — see recall. The row-level result behind this, over
              tens of thousands of test rows rather than {d.suite.n_cases}, is on
              the Model Lab page.
            </div>
          </div>
        </div>

        <div className="small muted" style={{ marginTop: 14 }}>
          The baseline is given a cause-attribution rule too (the tripped
          channel mapped to the mode it most often signals), so cause accuracy
          is a real comparison rather than a category it cannot score in.
        </div>
      </Panel>

      {/* ---------------- confusion ---------------- */}
      <Panel
        title="Where the two systems actually differ"
        right={<span className="small muted">{d.suite.n_cases} cases · counted, not rated</span>}
      >
        <div className="cm-row">
          <Confusion title="Simple baseline" m={base} tone="#4a9eff" />
          <Confusion title="Agent solution" m={agent} tone="#3fb950" />
        </div>
        <div className="small muted" style={{ marginTop: 12 }}>
          The agent converts {agent.true_positives - base.true_positives} missed
          failure(s) into caught ones and adds{" "}
          {agent.false_positives - base.false_positives} false alarm. That is the
          whole difference between the two systems on this suite.
        </div>
      </Panel>

      {/* ---------------- capabilities + categories ---------------- */}
      <div className="grid c2">
        <Panel title="Capabilities the baseline does not have">
          <table>
            <tbody>
              <tr>
                <td>Verification verdicts</td>
                <td className="num">
                  {Object.entries(agent.verification_verdicts ?? {}).map(([k, v]: any) => (
                    <span key={k} style={{ marginLeft: 8 }}>
                      <Pill kind={k.startsWith("PASS") ? "pass" : "fail"}>{k} {v}</Pill>
                    </span>
                  ))}
                </td>
              </tr>
              <tr>
                <td>Cases cleared to act on</td>
                <td className="num">{agent.cases_cleared_to_act} / {d.suite.n_cases}</td>
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
              {Object.entries(agent.by_category).sort().map(([c, v]: any) => {
                const b = base.by_category?.[c]?.accuracy;
                return (
                  <tr key={c}>
                    <td>{c.replace(/_/g, " ")}</td>
                    <td className="num">{v.n}</td>
                    <td className="num">{pct(b, 0)}</td>
                    <td className="num" style={{
                      color: b != null && v.accuracy > b ? "var(--ok)"
                        : b != null && v.accuracy < b ? "var(--bad)" : undefined,
                    }}>
                      <strong>{pct(v.accuracy, 0)}</strong>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>

      {/* ---------------- every case ---------------- */}
      <Panel
        title={filtered ? `Cases — ${shown.length} of ${cases.length}` : "Every case"}
        right={
          <div style={{ display: "flex", gap: 9, alignItems: "center", flexWrap: "wrap" }}>
            {showFilters ? (
              <>
                <select value={cat} onChange={(e) => setCat(e.target.value)}>
                  <option value="all">all categories</option>
                  {categories.map((c) => (
                    <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
                  ))}
                </select>
                <select value={diff} onChange={(e) => setDiff(e.target.value)}>
                  <option value="all">all difficulties</option>
                  {["easy", "moderate", "hard"].map((x) => (
                    <option key={x} value={x}>{x}</option>
                  ))}
                </select>
                <select value={outcome} onChange={(e) => setOutcome(e.target.value)}>
                  <option value="all">all outcomes</option>
                  <option value="disagree">the two disagree</option>
                  <option value="agent_better">agent right, baseline wrong</option>
                  <option value="agent_wrong">agent wrong</option>
                  <option value="baseline_wrong">baseline wrong</option>
                </select>
                {filtered && (
                  <button className="ghost" onClick={() => { setCat("all"); setDiff("all"); setOutcome("all"); setPreset("all"); }}>
                    Clear
                  </button>
                )}
              </>
            ) : (
              <span className="small muted">Use Filters to narrow results</span>
            )}
            <select value={preset} onChange={(e) => setPreset(e.target.value)}>
              {PRESETS.map((x) => (
                <option key={x.k} value={x.k}>{x.label}</option>
              ))}
            </select>
          </div>
        }
      >
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th style={{ width: 26 }} />
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
              {pageRows.map((c: any) => {
                const b: any = byId.get(c.scenario_id);
                const o = OUTCOME[outcomeOf(c)];
                return (
                  <tr key={c.scenario_id}>
                    <td>
                      <span
                        className="dot-out"
                        style={{ background: o.color }}
                        title={`${outcomeOf(c)} — ${o.label}`}
                      />
                    </td>
                    <td className="mono">{c.scenario_id}</td>
                    <td className="small">{c.category.replace(/_/g, " ")}</td>
                    <td><span className="tag">{c.difficulty}</span></td>
                    <td className="mono small">{c.machine_id}</td>
                    <td className="num">{c.expected_alert ? "alert" : "quiet"}</td>
                    <td className="num" style={{ color: b?.correct_alert ? "var(--ok)" : "var(--bad)" }}>
                      {b?.alert ? "alert" : "quiet"}
                    </td>
                    <td className="num" style={{ color: c.correct_alert ? "var(--ok)" : "var(--bad)" }}>
                      {c.alert ? "alert" : "quiet"}
                    </td>
                    <td className="num">
                      <span className="risk-cell">
                        {pct(c.probability)}
                        <span className="risk-bar">
                          <span style={{
                            width: `${Math.min(100, (c.probability ?? 0) * 100)}%`,
                            background: c.alert ? "#f85149" : "#4a9eff",
                          }} />
                        </span>
                      </span>
                    </td>
                    <td className="small">
                      {c.predicted_type ? (
                        <span style={{
                          color: c.correct_type === true ? "var(--ok)"
                            : c.correct_type === false ? "var(--bad)" : undefined,
                        }}>
                          {c.predicted_type.replace(/_/g, " ")}
                        </span>
                      ) : "—"}
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
              {shown.length === 0 && (
                <tr>
                  <td colSpan={11} className="muted small">
                    No cases match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="case-foot">
          <div className="out-legend">
            {(Object.keys(OUTCOME) as (keyof typeof OUTCOME)[]).map((k) => (
              <span key={k}>
                <i style={{ background: OUTCOME[k].color }} />
                {k} · {OUTCOME[k].label}
              </span>
            ))}
          </div>
          <span className="small muted">
            Showing {shown.length === 0 ? 0 : from + 1}–
            {Math.min(from + perPage, shown.length)} of {shown.length} scenarios
            {shown.length !== cases.length && ` (filtered from ${cases.length})`}
          </span>
          <div className="pager-btns">
            <button onClick={() => setPage(cur - 1)} disabled={cur === 1} title="Previous">‹</button>
            {pageWindow(cur, pages).map((x, i) =>
              x === "…" ? (
                <span key={`g${i}`} className="pager-gap">…</span>
              ) : (
                <button
                  key={x}
                  className={x === cur ? "on" : ""}
                  onClick={() => setPage(x as number)}
                >
                  {x}
                </button>
              ),
            )}
            <button onClick={() => setPage(cur + 1)} disabled={cur === pages} title="Next">›</button>
          </div>
          <select value={perPage} onChange={(e) => setPerPage(Number(e.target.value))}>
            {[15, 25, 45].map((n) => (
              <option key={n} value={n}>{n} / page</option>
            ))}
          </select>
        </div>

        <div className="small muted" style={{ marginTop: 12 }}>
          Model under test: <span className="mono">{d.model_under_test?.kind}</span> on{" "}
          <span className="mono">{d.model_under_test?.feature_set}</span> features,
          threshold <span className="mono">{num(d.model_under_test?.threshold, 4)}</span>.
          Suite run {d.generated} in {d.wall_clock_s}s.
        </div>
      </Panel>
    </>
  );
}
