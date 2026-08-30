import { useEffect, useState } from "react";
import { Icon } from "../components/icons";

/* ── layout ────────────────────────────────────────────────────────────────
 * Hand-placed rather than auto-laid-out, because the point of this page is to
 * step through a narrative: the geometry has to stay put between steps so the
 * eye can follow one edge lighting up. Coordinates are in viewBox units.
 */
const VB = { w: 1330, h: 560 };

type Node = {
  id: string;
  x: number; y: number; w: number; h: number;
  label: string;
  sub?: string;
  kind?: "service" | "guard" | "llm" | "human" | "eval" | "store";
};

const H = 58;
const N: Node[] = [
  // row 1 — training time
  { id: "gen",    x: 26,   y: 62,  w: 150, h: H, label: "generator.py", sub: "physics-lite + confounders" },
  { id: "store",  x: 196,  y: 62,  w: 140, h: H, label: "artifacts/data", sub: "telemetry · failures", kind: "store" },
  { id: "pre",    x: 356,  y: 62,  w: 168, h: H, label: "preprocessing.py", sub: "leakage controls" },
  { id: "feat",   x: 544,  y: 62,  w: 142, h: H, label: "features.py", sub: "causal rolling stats" },
  { id: "models", x: 706,  y: 62,  w: 160, h: H, label: "LSTM · TFT · trees", sub: "7 candidates" },
  { id: "harn",   x: 886,  y: 62,  w: 152, h: H, label: "harness.py", sub: "canonical eval set" },
  { id: "bundle", x: 1058, y: 62,  w: 150, h: H, label: "bundle.py", sub: "frozen threshold" },

  // row 2 — serving a decision
  { id: "svc",    x: 26,   y: 212, w: 196, h: 74, label: "MODEL SERVICE", sub: "probability · band · confidence", kind: "service" },
  { id: "inv",    x: 264,  y: 216, w: 156, h: 66, label: "Investigate", sub: "predict · context · facts" },
  { id: "con",    x: 462,  y: 216, w: 176, h: 66, label: "Contest", sub: "2 advocates → adjudicator" },
  { id: "act",    x: 680,  y: 216, w: 156, h: 66, label: "Act", sub: "remediate · simulate · verify" },
  { id: "human",  x: 878,  y: 216, w: 160, h: 66, label: "Human approval", sub: "nothing is actuated", kind: "human" },
  { id: "api",    x: 1080, y: 216, w: 150, h: 66, label: "API + React UI", sub: "FastAPI · WebSocket" },

  // row 3 — supporting
  { id: "llm",    x: 26,   y: 400, w: 174, h: H, label: "llm/provider.py", sub: "narration only", kind: "llm" },
  { id: "res",    x: 224,  y: 400, w: 160, h: H, label: "Research agents", sub: "selects on validation" },
  { id: "asst",   x: 408,  y: 400, w: 150, h: H, label: "Assistant", sub: "retrieval + citations" },
  { id: "evid",   x: 582,  y: 400, w: 150, h: H, label: "Evidence", sub: "recompute recipe", kind: "guard" },
  { id: "cat",    x: 756,  y: 400, w: 150, h: H, label: "13 actions", sub: "closed catalogue", kind: "guard" },
  { id: "reg",    x: 930,  y: 400, w: 150, h: H, label: "SQLite registry", sub: "every agent step", kind: "store" },
  { id: "eval",   x: 1104, y: 400, w: 158, h: H, label: "evaluate · ablate", sub: "45 scenarios", kind: "eval" },
];

const byId = Object.fromEntries(N.map((n) => [n.id, n]));

type Edge = { id: string; from: string; to: string; route?: "h" | "vUp" | "vDown"; dashed?: boolean; label?: string };

const E: Edge[] = [
  { id: "gen-store",   from: "gen",    to: "store" },
  { id: "store-pre",   from: "store",  to: "pre" },
  { id: "pre-feat",    from: "pre",    to: "feat" },
  { id: "feat-mod",    from: "feat",   to: "models" },
  { id: "mod-harn",    from: "models", to: "harn" },
  { id: "harn-bundle", from: "harn",   to: "bundle" },
  { id: "bundle-svc",  from: "bundle", to: "svc",   route: "vDown" },
  { id: "svc-inv",     from: "svc",    to: "inv" },
  { id: "inv-con",     from: "inv",    to: "con" },
  { id: "con-act",     from: "con",    to: "act" },
  { id: "act-human",   from: "act",    to: "human" },
  { id: "human-api",   from: "human",  to: "api" },
  { id: "res-mod",     from: "res",    to: "models", route: "vUp", dashed: true, label: "selects on validation" },
  { id: "store-act",   from: "store",  to: "act",   route: "vDown", dashed: true, label: "verifier re-derives from raw telemetry" },
  { id: "evid-act",    from: "evid",   to: "act",   route: "vUp" },
  { id: "cat-act",     from: "cat",    to: "act",   route: "vUp" },
  { id: "llm-inv",     from: "llm",    to: "inv",   route: "vUp", dashed: true, label: "phrasing only" },
  { id: "svc-asst",    from: "svc",    to: "asst",  route: "vDown" },
  { id: "act-reg",     from: "act",    to: "reg",   route: "vDown" },
  { id: "eval-reg",    from: "eval",   to: "reg" },
  { id: "eval-act",    from: "eval",   to: "act",   route: "vUp", dashed: true, label: "does it earn its place?" },
  { id: "reg-api",     from: "reg",    to: "api",   route: "vUp" },
];

/* ── the narration ─────────────────────────────────────────────────────── */
type Step = { title: string; body: string; nodes: string[]; edges: string[] };

const STEPS: Step[] = [
  {
    title: "Everything starts from data you can regenerate",
    body: "A seeded generator writes telemetry for 80 machines; preprocessing applies the leakage controls — chronological splits with a one-horizon purge gap, a train-only scaler, causal features only, and windows that never span downtime.",
    nodes: ["gen", "store", "pre"],
    edges: ["gen-store", "store-pre"],
  },
  {
    title: "Seven candidates, judged on identical rows",
    body: "Features feed seven candidates. The harness scores every one of them on a canonical evaluation set, because sequence models can only score rows with a full clean lookback while trees can score everything — comparing them on different rows would rig the result.",
    nodes: ["pre", "feat", "models", "harn", "bundle", "res"],
    edges: ["pre-feat", "feat-mod", "mod-harn", "harn-bundle", "res-mod"],
  },
  {
    title: "One boundary: the Model Service",
    body: "The winning bundle is served behind ml/service.py. It answers the quantitative question — probability, band, confidence — and nothing else. No agent decides whether to investigate; the orchestrator does, from the band. That is why a tree can replace the TFT without touching a downstream agent.",
    nodes: ["bundle", "svc"],
    edges: ["bundle-svc"],
  },
  {
    title: "The incident chain",
    body: "The service hands a scored machine to the agents: investigate the facts, contest them from two opposed sides, then act. Twelve agents in total, passing structured state — no agent asks a model anything except through the service.",
    nodes: ["svc", "inv", "con", "act", "asst"],
    edges: ["svc-inv", "inv-con", "con-act", "svc-asst"],
  },
  {
    title: "Verification goes back to the raw data",
    body: "Every evidence item carries a recompute recipe — the function, the channel, the exact time range. Verification is not a language model reviewing prose: a separate agent re-derives each value from stored telemetry and diffs it.",
    nodes: ["store", "evid", "act"],
    edges: ["store-act", "evid-act"],
  },
  {
    title: "A closed action space, then a human",
    body: "Remediation selects from thirteen approved interventions by id. The language model cannot emit a physical action that is not in the catalogue, and a named human approves before anything is recorded. Nothing is ever actuated.",
    nodes: ["cat", "act", "human", "api"],
    edges: ["cat-act", "act-human", "human-api"],
  },
  {
    title: "The LLM hangs off the side",
    body: "It phrases findings. No metric, threshold or decision passes through it — which is why every number in this system reproduces on the mock provider with no API key.",
    nodes: ["llm", "inv", "con", "act"],
    edges: ["llm-inv"],
  },
  {
    title: "And the system is pointed at itself",
    body: "Every agent step is written to a SQLite registry. evaluate.py and ablate_adjudication.py replay the agents against 45 frozen scenarios and a model-only control, so 'does this agent earn its place?' is measured, not asserted. For the hypothesis contest, the measured answer was no.",
    nodes: ["act", "reg", "eval", "api"],
    edges: ["act-reg", "eval-act", "eval-reg", "reg-api"],
  },
];

/* ── geometry helpers ──────────────────────────────────────────────────── */
const cx = (n: Node) => n.x + n.w / 2;
const cy = (n: Node) => n.y + n.h / 2;

function pathFor(e: Edge): string {
  const a = byId[e.from];
  const b = byId[e.to];
  if (!a || !b) return "";
  if (e.route === "vUp" || e.route === "vDown") {
    // vertical-ish link between rows: leave from the nearer horizontal face
    const upward = cy(b) < cy(a);
    const ax = cx(a);
    const ay = upward ? a.y : a.y + a.h;
    const bx = cx(b);
    const by = upward ? b.y + b.h : b.y;
    const midY = (ay + by) / 2;
    return `M ${ax} ${ay} C ${ax} ${midY}, ${bx} ${midY}, ${bx} ${by}`;
  }
  // horizontal link
  const leftToRight = cx(b) > cx(a);
  const ax = leftToRight ? a.x + a.w : a.x;
  const bx = leftToRight ? b.x : b.x + b.w;
  const ay = cy(a);
  const by = cy(b);
  const midX = (ax + bx) / 2;
  return `M ${ax} ${ay} C ${midX} ${ay}, ${midX} ${by}, ${bx} ${by}`;
}

/** Midpoint of the drawn curve, not of the node centres. A vertical edge's
 *  centre-to-centre midpoint lands on whatever sits in the middle row -- which
 *  is how "selects on validation" ended up printed across the Contest node. */
function labelPos(e: Edge): { x: number; y: number } {
  const a = byId[e.from], b = byId[e.to];
  if (e.route === "vUp" || e.route === "vDown") {
    const upward = cy(b) < cy(a);
    const ay = upward ? a.y : a.y + a.h;
    const by = upward ? b.y + b.h : b.y;
    return { x: (cx(a) + cx(b)) / 2, y: (ay + by) / 2 + 4 };
  }
  const leftToRight = cx(b) > cx(a);
  const ax = leftToRight ? a.x + a.w : a.x;
  const bx = leftToRight ? b.x : b.x + b.w;
  return { x: (ax + bx) / 2, y: (cy(a) + cy(b)) / 2 - 8 };
}

const KIND_COLOR: Record<string, string> = {
  service: "#4a9eff",
  guard: "#d29922",
  llm: "#a371f7",
  human: "#3fb950",
  eval: "#56d4dd",
  store: "#8b98a5",
};

export default function Architecture() {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const s = STEPS[step];
  const activeNodes = new Set(showAll ? N.map((n) => n.id) : s.nodes);
  const activeEdges = new Set(showAll ? E.map((e) => e.id) : s.edges);

  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      setStep((p) => (p + 1 < STEPS.length ? p + 1 : (setPlaying(false), p)));
    }, 9000);
    return () => clearTimeout(t);
  }, [playing, step]);

  // Arrow keys drive it, so you can present without hunting for the buttons.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "ArrowRight") setStep((p) => Math.min(p + 1, STEPS.length - 1));
      if (ev.key === "ArrowLeft") setStep((p) => Math.max(p - 1, 0));
      if (ev.key === " ") { ev.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon"><Icon.layers /></div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Architecture
            </h2>
            <p style={{ margin: "4px 0 0", color: "var(--muted)", fontSize: 13.5, maxWidth: 820 }}>
              How the pieces relate, one relationship at a time. Arrow keys step
              through it; space plays.
            </p>
          </div>
        </div>
        <div className="topbar-right">
          <button
            className={`ghost${showAll ? " accent" : ""}`}
            onClick={() => { setShowAll((v) => !v); setPlaying(false); }}
          >
            <Icon.eye />
            {showAll ? "Follow the steps" : "Show everything"}
          </button>
        </div>
      </div>

      <div className="arch-wrap">
        <svg viewBox={`0 0 ${VB.w} ${VB.h}`} className="arch-svg" role="img"
             aria-label="PredictOps system architecture">
          <defs>
            <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#5b6b7d" />
            </marker>
            <marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5"
                    markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#e6edf3" />
            </marker>
          </defs>

          {/* band labels */}
          <text className="arch-band" x="26" y="42">TRAINING — offline, reproducible</text>
          <text className="arch-band" x="26" y="196">SERVING — one incident</text>
          <text className="arch-band" x="26" y="384">SUPPORTING</text>

          {/* edges first, so nodes sit on top */}
          {E.map((e) => {
            const on = activeEdges.has(e.id);
            return (
              <g key={e.id} className={`arch-edge${on ? " on" : ""}`}>
                <path
                  d={pathFor(e)}
                  fill="none"
                  strokeDasharray={e.dashed ? "6 5" : undefined}
                  markerEnd={on ? "url(#ah-on)" : "url(#ah)"}
                />
                {on && (
                  <path d={pathFor(e)} fill="none" className="arch-flow" />
                )}
              </g>
            );
          })}

          {/* nodes */}
          {N.map((n) => {
            const on = activeNodes.has(n.id);
            const tone = n.kind ? KIND_COLOR[n.kind] : "#3a4653";
            return (
              <g key={n.id} className={`arch-node${on ? " on" : ""}`}>
                <rect
                  x={n.x} y={n.y} width={n.w} height={n.h}
                  rx={n.kind === "human" ? n.h / 2 : 9}
                  style={{
                    stroke: on ? tone : "#2a3441",
                    strokeWidth: on && n.kind ? 2.2 : 1.2,
                    fill: on && n.kind ? `${tone}1f` : "#151b23",
                  }}
                />
                <text x={cx(n)} y={n.y + (n.sub ? 25 : 33)} className="arch-label"
                      style={{ fill: on && n.kind ? tone : on ? "#e6edf3" : "#7d8794" }}>
                  {n.label}
                </text>
                {n.sub && (
                  <text x={cx(n)} y={n.y + 42} className="arch-sub">{n.sub}</text>
                )}
              </g>
            );
          })}

          {/* edge captions, only for the active step */}
          {!showAll &&
            E.filter((e) => e.label && activeEdges.has(e.id)).map((e) => {
              const pos = labelPos(e);
              return (
                <text key={`l-${e.id}`} className="arch-edge-label"
                      x={pos.x} y={pos.y}>
                  {e.label}
                </text>
              );
            })}
        </svg>
      </div>

      {/* ── narration ── */}
      <div className="arch-story">
        <div className="arch-step-n">
          {showAll ? "—" : `${step + 1} / ${STEPS.length}`}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="arch-step-title">
            {showAll ? "The whole system" : s.title}
          </div>
          <p className="arch-step-body">
            {showAll
              ? "Every node and edge lit at once. Use the steps to walk it one relationship at a time."
              : s.body}
          </p>
        </div>
        <div className="arch-controls">
          <button className="ghost" onClick={() => { setStep(0); setPlaying(false); setShowAll(false); }}>
            <Icon.refresh />
          </button>
          <button className="ghost" disabled={showAll || step === 0}
                  onClick={() => { setPlaying(false); setStep((p) => p - 1); }}>‹</button>
          <button className={playing ? "" : "ghost"} disabled={showAll}
                  onClick={() => setPlaying((p) => !p)}>
            {playing ? "Pause" : "Play"}
          </button>
          <button className="ghost" disabled={showAll || step === STEPS.length - 1}
                  onClick={() => { setPlaying(false); setStep((p) => p + 1); }}>›</button>
        </div>
      </div>

      <div className="arch-dots">
        {STEPS.map((st, i) => (
          <button
            key={i}
            title={st.title}
            className={!showAll && i === step ? "on" : ""}
            onClick={() => { setShowAll(false); setPlaying(false); setStep(i); }}
          />
        ))}
      </div>

      <div className="arch-legend">
        {[
          ["service", "Model Service — the only route to a model"],
          ["guard", "Constraint — recompute recipes, closed action list"],
          ["human", "Human approval"],
          ["llm", "LLM — narration only, no metric passes through"],
          ["eval", "Measurement — does each agent earn its place"],
        ].map(([k, label]) => (
          <span key={k}>
            <i style={{ background: KIND_COLOR[k], borderColor: KIND_COLOR[k] }} />
            {label}
          </span>
        ))}
      </div>
    </>
  );
}
