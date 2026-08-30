import { useRef, useState } from "react";
import { api } from "../api";
import { Panel, Pill, num, useAsync } from "../components/common";
import { Icon } from "../components/icons";

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

const SUGGESTIONS: { q: string; icon: keyof typeof Icon; tone: string }[] = [
  { q: "Which machines are at risk?", icon: "fleet", tone: "#3fb950" },
  { q: "Why is PUMP-017 flagged?", icon: "alert", tone: "#f85149" },
  { q: "Did the adjudicator actually help?", icon: "graph", tone: "#a371f7" },
  { q: "Which model was selected and why?", icon: "lab", tone: "#4a9eff" },
  { q: "How good is it against the baseline?", icon: "activity", tone: "#56d4dd" },
  { q: "What can we do about bearing degradation?", icon: "wrench", tone: "#f0883e" },
  { q: "What are the thresholds?", icon: "beaker", tone: "#d29922" },
];

const SOURCE_TONE: Record<string, string> = {
  fleet: "#3fb950",
  evidence: "#4a9eff",
  experiments: "#a371f7",
  evaluation: "#f0883e",
  ablation: "#56d4dd",
  catalogue: "#d29922",
};

const SOURCE_ICON: Record<string, keyof typeof Icon> = {
  fleet: "fleet",
  evidence: "search",
  experiments: "beaker",
  evaluation: "check",
  ablation: "lab",
  catalogue: "wrench",
};

export default function Assistant() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const src = useAsync(() => api.assistantSources().catch(() => null), []);

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setQ("");
    setTurns((t) => [...t, { question }]);
    setBusy(true);
    try {
      const r = await api.assistant(question);
      setTurns((t) => [...t.slice(0, -1), { question, ...r }]);
    } catch (e: any) {
      setTurns((t) => [...t.slice(0, -1), { question, error: String(e.message ?? e) }]);
    } finally {
      setBusy(false);
      setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 60);
    }
  }

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.chat />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Assistant
            </h2>
            <p
              style={{
                margin: "3px 0 0",
                color: "var(--muted)",
                fontSize: 13.5,
                maxWidth: 720,
              }}
            >
              Answers come from this system's own records — fleet scores,
              evidence items, the experiment registry, the evaluation and the
              ablation. Every number is cited. It refuses anything it cannot
              ground, and it cannot approve or carry out work.
            </p>
          </div>
        </div>
      </div>

      {/* ---------------- ask ---------------- */}
      <Panel>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            style={{ flex: 1 }}
            value={q}
            placeholder="Ask about a machine, the models, or the results…"
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(q)}
          />
          <button
            onClick={() => ask(q)}
            disabled={busy || !q.trim()}
            style={{ display: "inline-flex", alignItems: "center", gap: 8 }}
          >
            <Icon.arrow />
            {busy ? "Thinking…" : "Ask"}
          </button>
        </div>

        <div
          className="label"
          style={{
            fontSize: 10.5,
            letterSpacing: ".9px",
            textTransform: "uppercase",
            color: "var(--muted)",
            margin: "18px 0 10px",
            fontWeight: 600,
          }}
        >
          Suggested questions
        </div>
        <div style={{ display: "flex", gap: 9, flexWrap: "wrap" }}>
          {SUGGESTIONS.map((s) => {
            const I = Icon[s.icon];
            return (
              <button
                key={s.q}
                className="ghost suggest"
                onClick={() => ask(s.q)}
                disabled={busy}
              >
                <span style={{ color: s.tone, display: "inline-flex" }}>
                  <I />
                </span>
                {s.q}
              </button>
            );
          })}
        </div>
      </Panel>

      {/* ---------------- the standing claim ---------------- */}
      <div className="claim">
        <div className="claim-icon">
          <Icon.info />
        </div>
        <div>
          <div className="claim-title">This assistant retrieves; it does not recall.</div>
          <div className="small" style={{ marginTop: 4, color: "var(--muted)" }}>
            If the system has not measured something, it says so rather than
            producing a plausible answer — try asking it something outside the
            plant. Any rephrasing by a language model is discarded if it
            introduces a number that is not in the retrieved facts.
          </div>
        </div>
      </div>

      {/* ---------------- what it can draw on ---------------- */}
      {src.data && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(290px, 1fr))",
            gap: 14,
            marginBottom: 18,
          }}
        >
          {src.data.sources.map((s: any) => {
            const I = Icon[SOURCE_ICON[s.key] ?? "info"];
            const tone = SOURCE_TONE[s.key] ?? "var(--accent)";
            return (
              <a key={s.key} href={s.href} className="source-card">
                <div className="source-head">
                  <span className="source-icon" style={{ color: tone, background: `${tone}22` }}>
                    <I />
                  </span>
                  <span className="source-label">{s.label}</span>
                </div>
                <div className="small muted" style={{ minHeight: 34 }}>
                  {s.detail}
                </div>
                <div className="source-foot">
                  <span className="source-value" style={{ color: tone }}>
                    {s.value.toLocaleString()}
                  </span>
                  <span className="small muted">{s.unit}</span>
                  <span style={{ marginLeft: "auto", color: "var(--muted)" }}>
                    <Icon.arrow />
                  </span>
                </div>
              </a>
            );
          })}
        </div>
      )}

      {/* ---------------- conversation ---------------- */}
      {turns.map((t, i) => (
        <div key={i} style={{ marginBottom: 18 }}>
          <div className="ask-bubble">
            <strong>{t.question}</strong>
          </div>

          <div className="panel">
            {t.error ? (
              <div style={{ color: "var(--bad)" }}>{t.error}</div>
            ) : !t.answer ? (
              <span className="muted">…</span>
            ) : (
              <>
                <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
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
                      sources ({t.citations!.length})
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

      {/* ---------------- footer ---------------- */}
      <div className="grounding-foot">
        <span style={{ color: "var(--muted)", display: "inline-flex" }}>
          <Icon.check />
        </span>
        All responses are grounded in measured data. The assistant will not
        speculate or fabricate, and it cannot approve or execute work.
        {src.data && (
          <span className="muted">
            {" "}
            · {src.data.agent_steps_logged.toLocaleString()} agent steps on
            record
          </span>
        )}
      </div>
    </>
  );
}
