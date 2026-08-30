import { useMemo, useState } from "react";
import { ExperimentRow, api } from "../api";
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

const DECISION_TONE: Record<string, string> = {
  reference: "#4a9eff",
  kept: "#3fb950",
  removed: "#8b98a5",
};

/** A glyph per candidate family, so the timeline reads as a shape. */
function glyphFor(r: ExperimentRow): keyof typeof Icon {
  if (r.model === "threshold_baseline") return "target";
  if (r.model === "ensemble") return "layers";
  if (r.model === "lstm" || r.model === "tft") return "activity";
  return "trend";
}

const delta = (v: number) => `${v >= 0 ? "+" : ""}${num(v, 4)}`;
const tone = (v: number) =>
  v > 0.01 ? "var(--ok)" : v < -0.01 ? "var(--bad)" : "var(--muted)";

function MetricCell({ label, value, extra, extraTone }: {
  label: string; value: React.ReactNode; extra?: string; extraTone?: string;
}) {
  return (
    <span className="xp-metric">
      <span className="xp-metric-k">{label}</span>
      <strong>{value}</strong>
      {extra && <em style={{ color: extraTone }}>{extra}</em>}
    </span>
  );
}

function ExperimentCard({
  r, index, prev, baseline,
}: {
  r: ExperimentRow;
  index: number;
  prev?: ExperimentRow;
  baseline?: ExperimentRow;
}) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"evidence" | "operational" | "params">("evidence");

  const val = r.metrics?.val?.pr_auc ?? 0;
  const prevVal = prev?.metrics?.val?.pr_auc;
  const baseVal = baseline?.metrics?.val?.pr_auc ?? 0;
  const step = prevVal == null ? null : val - prevVal;
  const vsBase = val - baseVal;
  const ev = r.metrics?.event ?? {};
  const colour = DECISION_TONE[r.decision] ?? "var(--muted)";
  const G = Icon[glyphFor(r)];

  return (
    <div className="xp-row">
      <div className="xp-rail">
        <span
          className="xp-node"
          style={{ color: colour, borderColor: colour, background: `${colour}1f` }}
        >
          <G />
        </span>
      </div>

      <div className={`xp-card${r.decision === "removed" ? " is-removed" : ""}`}>
        <div className="xp-head">
          <div className="xp-title">
            <strong>{r.stage}</strong>
            <span className="xp-dash">—</span>
            <span>{r.name}</span>
            <span className="tag">{r.model}</span>
            <span className="tag">{r.feature_set}</span>
          </div>
          <Pill kind={r.decision}>{r.decision}</Pill>
        </div>

        <div className="xp-line">
          <span className="xp-label">HYPOTHESIS</span>
          <span className="small muted">{r.hypothesis}</span>
        </div>

        <div className="xp-metrics">
          <MetricCell label="val PR-AUC" value={num(val, 4)}
            extra={step == null ? undefined : `(${delta(step)} vs previous)`}
            extraTone={step == null ? undefined : tone(step)} />
          <MetricCell label="test F1" value={num(r.metrics?.row?.f1, 4)} />
          {/* The reference row has nothing to compare against; "+0.0000"
              would read as a measured result rather than a tautology. */}
          <MetricCell label="vs baseline"
            value={
              baseline && baseline.id === r.id
                ? <span className="muted">—</span>
                : <span style={{ color: tone(vsBase) }}>{delta(vsBase)}</span>
            }
            extra={baseline && baseline.id === r.id ? undefined : "PR-AUC"} />
          <MetricCell label="took" value={`${num(r.duration_s, 1)}s`} />
        </div>

        {r.learning && (
          <div className="xp-line">
            <span className="xp-label">EVIDENCE / DECISION</span>
            <span className="small">{r.learning}</span>
          </div>
        )}

        <button className="xp-toggle" onClick={() => setOpen((v) => !v)}>
          <span className={`chev${open ? " on" : ""}`}>›</span>
          Parameters and full metrics
        </button>

        {open && (
          <div className="xp-expand">
            <div className="tabs">
              {([
                ["evidence", "Evidence & metrics"],
                ["operational", "Operational view"],
                ["params", "Parameters"],
              ] as const).map(([k, label]) => (
                <button
                  key={k}
                  className={tab === k ? "on" : ""}
                  onClick={() => setTab(k)}
                >
                  {label}
                </button>
              ))}
            </div>

            {tab === "evidence" && (
              <div className="xp-two">
                <div className="xp-prov">
                  <div className="xp-prov-head">
                    <Icon.database />
                    Provenance
                  </div>
                  <dl>
                    <dt>Registry id</dt>
                    <dd className="mono">experiments.id = {r.id}</dd>
                    <dt>Recorded in</dt>
                    <dd className="mono">artifacts/experiments/experiments.db</dd>
                    <dt>Compared with</dt>
                    <dd>
                      {prev ? (
                        <span className="tag">{prev.name}</span>
                      ) : (
                        <span className="muted small">
                          nothing — this is the reference point
                        </span>
                      )}
                    </dd>
                    <dt>Evaluation rows</dt>
                    <dd className="mono">
                      val {r.metrics?.val?.n?.toLocaleString() ?? "—"} ·
                      test {r.metrics?.row?.n?.toLocaleString() ?? "—"}
                    </dd>
                  </dl>
                  <p className="small muted">
                    Every figure on this card was read back from the experiment
                    registry as the run executed; none is hand-entered.
                    Candidates are chosen on <strong>validation</strong> PR-AUC.
                    Test F1 is reported at the same frozen threshold and is
                    never used to choose.
                  </p>
                </div>

                <div className="xp-core">
                  <div className="xp-prov-head">
                    <Icon.graph />
                    Core metrics
                  </div>
                  <table>
                    <thead>
                      <tr>
                        <th>Split</th>
                        <th className="num">PR-AUC</th>
                        <th className="num">F1</th>
                        <th className="num">Precision</th>
                        <th className="num">Recall</th>
                        <th className="num">ROC-AUC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(["val", "row"] as const).map((k) => {
                        const m = r.metrics?.[k];
                        if (!m) return null;
                        return (
                          <tr key={k}>
                            <td>{k === "val" ? "validation" : "test"}</td>
                            <td className="num">
                              {k === "val" ? <strong>{num(m.pr_auc, 4)}</strong> : num(m.pr_auc, 4)}
                            </td>
                            <td className="num">{num(m.f1, 4)}</td>
                            <td className="num">{num(m.precision, 3)}</td>
                            <td className="num">{num(m.recall, 3)}</td>
                            <td className="num">{num(m.roc_auc, 3)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className="small muted" style={{ marginTop: 9 }}>
                    Selection metric in bold. Threshold{" "}
                    <span className="mono">{num(r.metrics?.val?.threshold, 4)}</span>,
                    tuned on validation and frozen before test was read.
                  </div>
                </div>
              </div>
            )}

            {tab === "operational" && (
              ev.n_events == null ? (
                <div className="small muted">
                  No event-level metrics recorded for this candidate.
                </div>
              ) : (
                <>
                  <div className="xp-ops">
                    <div>
                      <div className="label">Events caught</div>
                      <div className="v">{ev.detected}/{ev.n_events}</div>
                      <div className="small muted">{pct(ev.detection_rate, 0)}</div>
                    </div>
                    <div>
                      <div className="label">Mean early warning</div>
                      <div className="v">{num(ev.mean_early_warning_h, 2)} h</div>
                      <div className="small muted">
                        median {num(ev.median_early_warning_h, 2)} h
                      </div>
                    </div>
                    <div>
                      <div className="label">False alarms</div>
                      <div className="v">{num(ev.false_alarms_per_machine_day, 2)}</div>
                      <div className="small muted">per machine per day</div>
                    </div>
                  </div>
                  {ev.detection_by_type && (
                    <table style={{ marginTop: 14 }}>
                      <thead>
                        <tr>
                          <th>Failure mode</th>
                          <th className="num">Events</th>
                          <th className="num">Detected</th>
                          <th className="num">Rate</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(ev.detection_by_type).map(([k, v]: any) => (
                          <tr key={k}>
                            <td>{k.replace(/_/g, " ")}</td>
                            <td className="num">{v.n}</td>
                            <td className="num">{v.detected}</td>
                            <td className="num">{pct(v.rate, 0)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  <div className="small muted" style={{ marginTop: 10 }}>
                    Row-level PR-AUC selects the model; this is what a
                    maintenance planner actually feels.
                  </div>
                </>
              )
            )}

            {tab === "params" && (
              <div className="xp-two">
                <div>
                  <div className="xp-prov-head">
                    <Icon.beaker />
                    Parameters
                  </div>
                  {Object.keys(r.params ?? {}).length === 0 ? (
                    <div className="small muted">No parameters recorded.</div>
                  ) : (
                    <table>
                      <tbody>
                        {Object.entries(r.params).map(([k, v]) => (
                          <tr key={k}>
                            <td className="mono">{k}</td>
                            <td className="num mono">{String(v)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div>
                  <div className="xp-prov-head">
                    <Icon.info />
                    Full metrics
                  </div>
                  <pre style={{ maxHeight: 320, overflow: "auto" }}>
                    {JSON.stringify(r.metrics, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Experiments() {
  const [runId, setRunId] = useState<string | undefined>();
  const exp = useAsync(() => api.experiments(runId), [runId]);
  const log = useAsync(() => api.changelog(), []);
  const [showFilters, setShowFilters] = useState(false);
  const [decisions, setDecisions] = useState<string[]>([]);
  const [family, setFamily] = useState<string>("all");

  const all = exp.data?.experiments ?? [];
  const runs = exp.data?.runs ?? [];
  const activeRun = runId ?? exp.data?.run_id;

  // Every hook must run before the early returns below, or the hook order
  // changes between the loading render and the loaded one (React #310).
  const families = useMemo(
    () => [...new Set(all.map((r) => r.model))].sort(),
    [all],
  );

  if (exp.loading) return <Loading what="experiment timeline" />;
  if (exp.error) return <ErrorNote error={exp.error} />;

  // Filters narrow what is shown; deltas stay anchored to the full, ordered run
  // so "+0.1973 vs previous" cannot silently change meaning when a row is
  // hidden. Position in the real sequence is what the number describes.
  const shown = all.filter(
    (r) =>
      (decisions.length === 0 || decisions.includes(r.decision)) &&
      (family === "all" || r.model === family),
  );

  function exportRun() {
    const blob = new Blob(
      [JSON.stringify({ run_id: activeRun, experiments: all }, null, 2)],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `experiments-${activeRun ?? "run"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const toggle = (d: string) =>
    setDecisions((s) => (s.includes(d) ? s.filter((x) => x !== d) : [...s, d]));

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.beaker />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Experiments
            </h2>
            <p style={{ margin: "4px 0 0", color: "var(--muted)", fontSize: 13.5, maxWidth: 820 }}>
              Hypothesis → experiment → evidence → decision, for every candidate
              the research agent ran. Recorded as each experiment executed — no
              number here was written by hand, and rejected hypotheses are kept.
            </p>
          </div>
        </div>
        <div className="topbar-right">
          <button
            className={`ghost${decisions.length || family !== "all" ? " accent" : ""}`}
            onClick={() => setShowFilters((v) => !v)}
          >
            <Icon.search />
            Filters
            {(decisions.length > 0 || family !== "all") && (
              <span className="filter-count">
                {decisions.length + (family !== "all" ? 1 : 0)}
              </span>
            )}
          </button>
          <button className="ghost accent" onClick={exportRun} disabled={!all.length}>
            <Icon.download />
            Export
          </button>
        </div>
      </div>

      {showFilters && (
        <Panel>
          <div className="filter-bar">
            <span className="small muted">Decision</span>
            {["reference", "kept", "removed"].map((d) => (
              <button
                key={d}
                className={`chip${decisions.includes(d) ? " on" : ""}`}
                onClick={() => toggle(d)}
              >
                {d}
                <span className="muted"> {all.filter((r) => r.decision === d).length}</span>
              </button>
            ))}
            <span className="small muted" style={{ marginLeft: 10 }}>Model</span>
            <select value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="all">all models</option>
              {families.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            {(decisions.length > 0 || family !== "all") && (
              <button
                className="ghost"
                style={{ marginLeft: "auto" }}
                onClick={() => { setDecisions([]); setFamily("all"); }}
              >
                Clear
              </button>
            )}
          </div>
        </Panel>
      )}

      {/* ---------------- timeline ---------------- */}
      <div className="xp-timeline-head">
        <div>
          <div className="xp-timeline-title">
            Timeline — run <strong>{activeRun}</strong>
            <span className="tag">{all.length} experiments</span>
          </div>
          <div className="small muted" style={{ marginTop: 3 }}>
            Experiments in this research run, ordered by execution.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="small muted">Run</span>
          <select value={activeRun ?? ""} onChange={(e) => setRunId(e.target.value)}>
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.n_experiments})
              </option>
            ))}
          </select>
        </div>
      </div>

      {shown.length === 0 ? (
        <div className="banner info">
          No experiments match these filters. {all.length} were recorded in this
          run.
        </div>
      ) : (
        <div className="xp-timeline">
          {shown.map((r) => {
            const i = all.indexOf(r);
            return (
              <ExperimentCard
                key={r.id}
                r={r}
                index={i}
                prev={i > 0 ? all[i - 1] : undefined}
                baseline={all[0]}
              />
            );
          })}
        </div>
      )}

      <Panel title="Generated changelog">
        {log.loading ? (
          <Loading what="changelog" />
        ) : (
          <>
            <div className="small muted" style={{ marginBottom: 10 }}>
              Rendered from the registry by{" "}
              <span className="mono">experiments/changelog.py</span> — the same
              document committed to the repo.
            </div>
            <pre style={{ whiteSpace: "pre-wrap" }}>{log.data?.markdown}</pre>
          </>
        )}
      </Panel>
    </>
  );
}
