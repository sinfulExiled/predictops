import { num } from "./common";

/**
 * When the two readings are too close to separate, the workflow must visibly
 * stop short of repair. Rendering this as an ordinary banner would hide the
 * one thing the hypothesis contest actually contributes.
 */
export default function ContestedGate({
  adjudication,
  degradation,
  confound,
}: {
  adjudication: any;
  degradation: any;
  confound: any;
}) {
  const d = adjudication?.decision;
  if (d !== "contested" && d !== "overturned" && d !== "insufficient_evidence")
    return null;

  const overturned = d === "overturned";
  const benign =
    confound?.alternative_explanations?.[0]?.explanation ??
    adjudication?.leading_benign_explanation;

  return (
    <div
      className="panel"
      style={{
        border: `1px solid ${overturned ? "var(--ok)" : "var(--warn)"}`,
        background: overturned
          ? "rgba(63,185,80,.06)"
          : "rgba(210,153,34,.06)",
      }}
    >
      <h3 style={{ color: overturned ? "var(--ok)" : "var(--warn)" }}>
        {overturned
          ? "Flag overturned — no fault indicated"
          : d === "insufficient_evidence"
            ? "Insufficient evidence"
            : "Contested diagnosis"}
      </h3>

      <div style={{ marginBottom: 14 }}>
        {overturned
          ? `The benign explanation${benign ? ` (${benign})` : ""} outweighed the degradation case. The model's flag does not stand.`
          : d === "insufficient_evidence"
            ? "Neither reading of the evidence reached the floor required to act."
            : "The system found two conflicting interpretations of the same evidence and could not separate them."}
      </div>

      <table style={{ marginBottom: 16 }}>
        <tbody>
          <tr>
            <td>Cause advocate</td>
            <td className="small muted">
              {degradation?.failure_type?.replace(/_/g, " ") ??
                degradation?.conclusion}
            </td>
            <td className="num">
              <strong>{num(adjudication.degradation_score, 2)}</strong>
            </td>
          </tr>
          <tr>
            <td>Confound advocate</td>
            <td className="small muted">{benign ?? "no benign explanation"}</td>
            <td className="num">
              <strong>{num(adjudication.confound_score, 2)}</strong>
            </td>
          </tr>
          <tr>
            <td className="muted">Margin</td>
            <td />
            <td className="num muted">
              {adjudication.margin >= 0 ? "+" : ""}
              {num(adjudication.margin, 2)}
            </td>
          </tr>
        </tbody>
      </table>

      <div
        style={{
          border: `1px dashed ${overturned ? "var(--ok)" : "var(--warn)"}`,
          borderRadius: 7,
          padding: "14px 16px",
          textAlign: "center",
          fontWeight: 600,
          letterSpacing: 0.4,
          color: overturned ? "var(--ok)" : "var(--warn)",
        }}
      >
        {overturned || d === "insufficient_evidence"
          ? "MONITORING ONLY — NO PHYSICAL WORK PROPOSED"
          : "INSPECTION REQUIRED — REPAIR NOT AUTHORISED"}
      </div>

      <div className="small muted" style={{ marginTop: 12 }}>
        Evidence is insufficient for automated remediation. The remediation
        agent is gated on this decision in code, so a case that has not survived
        challenge cannot propose a replacement or a shutdown regardless of the
        model's confidence.
      </div>
    </div>
  );
}
