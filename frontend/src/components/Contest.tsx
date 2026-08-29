import { Pill, num, pct } from "./common";

interface Props {
  degradation: any;
  confound: any;
  adjudication: any;
}

const DECISION_TONE: Record<string, string> = {
  alert: "fail",
  contested: "warn",
  overturned: "pass",
  insufficient_evidence: "warn",
  no_alert: "pass",
};

const DECISION_LABEL: Record<string, string> = {
  alert: "Alert — degradation case stands",
  contested: "Contested — check before repairing",
  overturned: "Overturned — benign explanation wins",
  insufficient_evidence: "Insufficient evidence",
  no_alert: "No alert raised",
};

/** Side-by-side view of the two competing readings and how it was settled. */
export default function Contest({ degradation, confound, adjudication }: Props) {
  if (!adjudication) return null;
  const d = adjudication.degradation_score ?? 0;
  const c = adjudication.confound_score ?? 0;
  const total = Math.max(d + c, 0.0001);

  return (
    <div className="panel">
      <h3>Competing hypotheses</h3>

      <div style={{ display: "flex", height: 28, borderRadius: 6, overflow: "hidden", marginBottom: 6 }}>
        <div
          style={{
            width: `${(d / total) * 100}%`,
            background: "rgba(248,81,73,.55)",
            display: "flex",
            alignItems: "center",
            paddingLeft: 10,
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          degradation {num(d, 2)}
        </div>
        <div
          style={{
            width: `${(c / total) * 100}%`,
            background: "rgba(63,185,80,.45)",
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            paddingRight: 10,
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          benign {num(c, 2)}
        </div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <Pill kind={DECISION_TONE[adjudication.decision] ?? "warn"}>
          {DECISION_LABEL[adjudication.decision] ?? adjudication.decision}
        </Pill>{" "}
        <span className="small muted">
          margin {adjudication.margin >= 0 ? "+" : ""}
          {num(adjudication.margin, 2)}
        </span>
        {adjudication.changed_the_model_verdict && (
          <>
            {" "}
            <Pill kind="warn">overruled the model</Pill>
          </>
        )}
      </div>

      <div className="small" style={{ marginBottom: 16 }}>
        {adjudication.rationale}
      </div>

      <div className="grid c2">
        <div className="step" style={{ borderLeft: "2px solid var(--bad)" }}>
          <div className="head">
            <strong>A fault is developing</strong>
            <span className="mono">{num(degradation?.score, 2)}</span>
          </div>
          <div className="small">{degradation?.argument ?? degradation?.conclusion}</div>
          {degradation?.factors && (
            <div className="small muted" style={{ marginTop: 8 }}>
              model {pct(degradation.factors.model_probability, 0)} · trend
              persistence {pct(degradation.factors.trend_persistence, 0)} ·
              load flat: {String(degradation.factors.load_is_flat)}
            </div>
          )}
          <div className="small muted" style={{ marginTop: 8, fontStyle: "italic" }}>
            Would change my mind: {degradation?.would_change_my_mind}
          </div>
        </div>

        <div className="step" style={{ borderLeft: "2px solid var(--ok)" }}>
          <div className="head">
            <strong>Nothing is wrong</strong>
            <span className="mono">{num(confound?.score, 2)}</span>
          </div>
          <div className="small">{confound?.argument ?? confound?.conclusion}</div>
          {(confound?.alternative_explanations ?? []).length > 0 && (
            <table style={{ marginTop: 8 }}>
              <tbody>
                {confound.alternative_explanations.map((e: any) => (
                  <tr key={e.explanation}>
                    <td className="small">{e.explanation}</td>
                    <td className="num small">{num(e.strength, 2)}</td>
                    <td className="small muted">{e.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="small muted" style={{ marginTop: 8, fontStyle: "italic" }}>
            Would change my mind: {confound?.would_change_my_mind}
          </div>
        </div>
      </div>
    </div>
  );
}
