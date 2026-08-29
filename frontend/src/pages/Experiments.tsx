import { api } from "../api";
import {
  ErrorNote,
  Loading,
  PageHead,
  Panel,
  Pill,
  num,
  useAsync,
} from "../components/common";

export default function Experiments() {
  const exp = useAsync(() => api.experiments(), []);
  const log = useAsync(() => api.changelog(), []);

  if (exp.loading) return <Loading what="experiment timeline" />;
  if (exp.error) return <ErrorNote error={exp.error} />;

  const rows = exp.data?.experiments ?? [];
  const baseVal = rows[0]?.metrics?.val?.pr_auc ?? 0;

  return (
    <>
      <PageHead
        title="Experiments"
        blurb="Hypothesis → experiment → evidence → decision, for every candidate the research agent ran. Recorded as each experiment executed — no number here was written by hand, and rejected hypotheses are kept."
      />

      <Panel title={`Timeline — run ${exp.data?.run_id}`}>
        {rows.map((r, i) => {
          const val = r.metrics?.val?.pr_auc ?? 0;
          const prev = i > 0 ? rows[i - 1].metrics?.val?.pr_auc ?? 0 : val;
          const step = i === 0 ? 0 : val - prev;
          return (
            <div className="step" key={r.id}>
              <div className="head">
                <div>
                  <strong>{r.stage}</strong> — {r.name}{" "}
                  <span className="tag">{r.model}</span>{" "}
                  <span className="tag">{r.feature_set}</span>
                </div>
                <Pill kind={r.decision}>{r.decision}</Pill>
              </div>

              <div className="small muted" style={{ marginBottom: 8 }}>
                <span className="tag">HYPOTHESIS</span> {r.hypothesis}
              </div>

              <div
                style={{
                  display: "flex",
                  gap: 22,
                  flexWrap: "wrap",
                  marginBottom: 8,
                }}
              >
                <span className="small">
                  val PR-AUC <strong>{num(val, 4)}</strong>
                  {i > 0 && (
                    <span
                      style={{
                        color: step > 0.01 ? "var(--ok)" : step < -0.01 ? "var(--bad)" : "var(--muted)",
                        marginLeft: 6,
                      }}
                    >
                      ({step >= 0 ? "+" : ""}
                      {num(step, 4)} vs previous)
                    </span>
                  )}
                </span>
                <span className="small">
                  test F1 <strong>{num(r.metrics?.row?.f1, 4)}</strong>
                </span>
                <span className="small muted">
                  vs baseline {num(val - baseVal, 4)} PR-AUC
                </span>
                <span className="small muted">{num(r.duration_s, 1)}s</span>
              </div>

              {r.learning && (
                <div className="small">
                  <span className="tag">EVIDENCE → DECISION</span> {r.learning}
                </div>
              )}

              <details style={{ marginTop: 8 }}>
                <summary className="small muted" style={{ cursor: "pointer" }}>
                  parameters and full metrics
                </summary>
                <pre>{JSON.stringify({ params: r.params, metrics: r.metrics }, null, 2)}</pre>
              </details>
            </div>
          );
        })}
      </Panel>

      <Panel title="Generated changelog">
        {log.loading ? (
          <Loading what="changelog" />
        ) : (
          <pre style={{ whiteSpace: "pre-wrap" }}>{log.data?.markdown}</pre>
        )}
      </Panel>
    </>
  );
}
