import { useRef, useState } from "react";
import { api } from "../api";
import { PageHead, Panel, Pill } from "../components/common";

interface Turn {
  question: string;
  answer?: string;
  intent?: string;
  citations?: any[];
  grounded?: boolean;
  refused?: boolean;
  action?: string | null;
  narration_rejected?: string[];
  error?: string;
}

const SUGGESTIONS = [
  "Which machines are at risk?",
  "Why is MOTOR-045 flagged?",
  "Did the adjudicator actually help?",
  "Which model was selected and why?",
  "How good is it against the baseline?",
  "What can we do about bearing degradation?",
  "What are the thresholds?",
];

export default function Assistant() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setQ("");
    setTurns((t) => [...t, { question }]);
    setBusy(true);
    try {
      const r = await api.assistant(question);
      setTurns((t) => [...t.slice(0, -1), { question, ...r }]);
    } catch (e: any) {
      setTurns((t) => [
        ...t.slice(0, -1),
        { question, error: String(e.message ?? e) },
      ]);
    } finally {
      setBusy(false);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 60);
    }
  }

  return (
    <>
      <PageHead
        title="Assistant"
        blurb="Answers come from this system's own records — fleet scores, evidence items, the experiment registry, the evaluation and the ablation. Every number is cited. It refuses anything it cannot ground, and it cannot approve or carry out work."
      />

      <Panel>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            style={{ flex: 1 }}
            value={q}
            placeholder="Ask about a machine, the models, or the results…"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(q)}
          />
          <button onClick={() => ask(q)} disabled={busy || !q.trim()}>
            {busy ? "Thinking…" : "Ask"}
          </button>
        </div>
        <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginTop: 12 }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              className="ghost"
              style={{ fontSize: 12, padding: "5px 10px" }}
              onClick={() => ask(s)}
              disabled={busy}
            >
              {s}
            </button>
          ))}
        </div>
      </Panel>

      {turns.length === 0 && (
        <div className="banner info">
          This assistant retrieves; it does not recall. If the system has not
          measured something, it says so rather than producing a plausible
          answer — try asking it something outside the plant.
        </div>
      )}

      {turns.map((t, i) => (
        <div key={i} style={{ marginBottom: 18 }}>
          <div
            className="panel"
            style={{ marginBottom: 8, background: "var(--panel-2)" }}
          >
            <strong>{t.question}</strong>
          </div>

          <div className="panel">
            {t.error ? (
              <div style={{ color: "var(--bad)" }}>{t.error}</div>
            ) : !t.answer ? (
              <span className="muted">…</span>
            ) : (
              <>
                <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
                  {t.intent && <span className="tag">{t.intent}</span>}
                  {t.refused ? (
                    <Pill kind="fail">refused</Pill>
                  ) : t.grounded ? (
                    <Pill kind="pass">
                      grounded · {t.citations?.length ?? 0} citation
                      {(t.citations?.length ?? 0) === 1 ? "" : "s"}
                    </Pill>
                  ) : (
                    <Pill kind="warn">no grounding</Pill>
                  )}
                  {t.action && <Pill kind="warn">ran {t.action}</Pill>}
                </div>

                <div style={{ whiteSpace: "pre-wrap" }}>{t.answer}</div>

                {t.narration_rejected && t.narration_rejected.length > 0 && (
                  <div className="banner bad" style={{ marginTop: 12 }}>
                    The language model's rephrasing introduced{" "}
                    {t.narration_rejected.length} number(s) not in the retrieved
                    facts ({t.narration_rejected.join(", ")}), so it was
                    discarded and the computed answer is shown instead.
                  </div>
                )}

                {(t.citations?.length ?? 0) > 0 && (
                  <details style={{ marginTop: 12 }}>
                    <summary className="small muted" style={{ cursor: "pointer" }}>
                      sources
                    </summary>
                    <table style={{ marginTop: 8 }}>
                      <thead>
                        <tr>
                          <th>Source</th>
                          <th>Record</th>
                          <th>Field</th>
                          <th className="num">Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {t.citations!.map((c, j) => (
                          <tr key={j}>
                            <td className="mono small">{c.source}</td>
                            <td className="small">{c.record}</td>
                            <td className="mono small">{c.field}</td>
                            <td className="num small">
                              {c.value === null ? "—" : String(c.value)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </details>
                )}
              </>
            )}
          </div>
        </div>
      ))}
      <div ref={endRef} />
    </>
  );
}
