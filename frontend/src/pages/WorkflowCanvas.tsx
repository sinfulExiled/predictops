import { useEffect, useMemo, useRef, useState } from "react";
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

interface NodeSpec {
  name: string;
  label: string;
  question: string;
  provides: string[];
  requires: string[];
  optional_inputs: string[];
  removable: boolean;
  note: string;
}

type Pos = Record<string, { x: number; y: number }>;

const W = 190;
const H = 62;

/** Tidy starting positions, by workflow stage. */
const LAYOUT: Pos = {
  predictor: { x: 40, y: 40 },
  context: { x: 40, y: 140 },
  investigator: { x: 40, y: 240 },
  degradation_advocate: { x: 300, y: 90 },
  confound_advocate: { x: 300, y: 210 },
  adjudicator: { x: 560, y: 150 },
  remediation: { x: 800, y: 150 },
  simulator: { x: 1040, y: 90 },
  verifier: { x: 1040, y: 220 },
};

export default function WorkflowCanvas() {
  const spec = useAsync(() => api.workflowSpec(), []);
  const [nodes, setNodes] = useState<string[]>([]);
  const [edges, setEdges] = useState<[string, string][]>([]);
  const [pos, setPos] = useState<Pos>(LAYOUT);
  const [drag, setDrag] = useState<{ id: string; dx: number; dy: number } | null>(null);
  const [wire, setWire] = useState<{ from: string; x: number; y: number } | null>(null);
  const [check, setCheck] = useState<any>(null);
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [machine, setMachine] = useState("MOTOR-045");
  const svgRef = useRef<SVGSVGElement>(null);

  const byName = useMemo(() => {
    const m: Record<string, NodeSpec> = {};
    (spec.data?.nodes ?? []).forEach((n: NodeSpec) => (m[n.name] = n));
    return m;
  }, [spec.data]);

  useEffect(() => {
    if (spec.data && nodes.length === 0) {
      setNodes(spec.data.default.nodes);
      setEdges(spec.data.default.edges.map((e: string[]) => [e[0], e[1]]));
    }
  }, [spec.data, nodes.length]);

  // revalidate whenever the graph changes
  useEffect(() => {
    if (!nodes.length) return;
    let alive = true;
    api
      .workflowValidate(nodes, edges)
      .then((c) => alive && setCheck(c))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [nodes, edges]);

  function svgPoint(e: React.MouseEvent) {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function onMove(e: React.MouseEvent) {
    if (drag) {
      const p = svgPoint(e);
      setPos((s) => ({ ...s, [drag.id]: { x: p.x - drag.dx, y: p.y - drag.dy } }));
    } else if (wire) {
      const p = svgPoint(e);
      setWire({ ...wire, x: p.x, y: p.y });
    }
  }

  function finishWire(target: string) {
    if (!wire || wire.from === target) return setWire(null);
    const exists = edges.some(([a, b]) => a === wire.from && b === target);
    if (!exists) setEdges((es) => [...es, [wire.from, target]]);
    setWire(null);
  }

  function toggleNode(name: string) {
    const s = byName[name];
    if (!s?.removable) return;
    if (nodes.includes(name)) {
      setNodes((n) => n.filter((x) => x !== name));
      setEdges((es) => es.filter(([a, b]) => a !== name && b !== name));
    } else {
      setNodes((n) => [...n, name]);
      // reconnect it the way the default graph does
      const def: [string, string][] = (spec.data?.default.edges ?? [])
        .map((e: string[]): [string, string] => [e[0], e[1]])
        .filter(([a, b]: [string, string]) => a === name || b === name);
      setEdges((es) => [
        ...es,
        ...def.filter(
          ([a, b]: [string, string]) =>
            (nodes.includes(a) || a === name) &&
            (nodes.includes(b) || b === name) &&
            !es.some(([x, y]) => x === a && y === b),
        ),
      ]);
    }
    setResult(null);
  }

  async function run() {
    setBusy(true);
    setErr("");
    setResult(null);
    try {
      setResult(await api.workflowRun(nodes, edges, machine));
    } catch (e: any) {
      setErr(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setNodes(spec.data.default.nodes);
    setEdges(spec.data.default.edges.map((e: string[]) => [e[0], e[1]]));
    setPos(LAYOUT);
    setResult(null);
  }

  if (spec.loading) return <Loading what="workflow spec" />;
  if (spec.error) return <ErrorNote error={spec.error} />;

  const all: NodeSpec[] = spec.data?.nodes ?? [];
  const width = 1260;
  const height = 340;

  return (
    <>
      <PageHead
        title="Workflow Canvas"
        blurb="Compose the incident workflow and run it. Connections are validated against the agents' own declared contracts — an edge is rejected because the target genuinely does not read what the source produces, not because of a UI rule."
      />

      <Panel>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <input
            style={{ width: 160 }}
            value={machine}
            onChange={(e) => setMachine(e.target.value.toUpperCase())}
          />
          <button onClick={run} disabled={busy || !check?.valid}>
            {busy ? "Running…" : `Run ${nodes.length} agents`}
          </button>
          <button className="ghost" onClick={reset}>
            Reset to default
          </button>
          {check && (
            <Pill kind={check.valid ? "pass" : "fail"}>
              {check.valid ? "valid graph" : `${check.errors.length} error(s)`}
            </Pill>
          )}
          <span className="small muted">
            drag nodes to move · drag from the right port to the left port of
            another node to connect · click an edge to delete
          </span>
        </div>
      </Panel>

      {err && <ErrorNote error={err} />}

      {check && !check.valid && (
        <div className="banner bad">
          <strong>This graph will not run.</strong>
          <ul style={{ margin: "8px 0 0", paddingLeft: 20 }}>
            {check.errors.map((e: string) => (
              <li key={e} className="small">
                {e}
              </li>
            ))}
          </ul>
        </div>
      )}
      {check?.valid && check.warnings?.length > 0 && (
        <div className="banner warn">
          <strong>Runs, with caveats.</strong>
          <ul style={{ margin: "8px 0 0", paddingLeft: 20 }}>
            {check.warnings.map((w: string) => (
              <li key={w} className="small">
                {w}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Panel title="Graph">
        <div style={{ overflowX: "auto" }}>
          <svg
            ref={svgRef}
            width={width}
            height={height}
            style={{ background: "var(--panel-2)", borderRadius: 8, cursor: drag ? "grabbing" : "default" }}
            onMouseMove={onMove}
            onMouseUp={() => {
              setDrag(null);
              setWire(null);
            }}
            onMouseLeave={() => {
              setDrag(null);
              setWire(null);
            }}
          >
            {/* edges */}
            {edges
              .filter(([a, b]) => nodes.includes(a) && nodes.includes(b))
              .map(([a, b], i) => {
                const pa = pos[a] ?? { x: 0, y: 0 };
                const pb = pos[b] ?? { x: 0, y: 0 };
                const x1 = pa.x + W;
                const y1 = pa.y + H / 2;
                const x2 = pb.x;
                const y2 = pb.y + H / 2;
                const mx = (x1 + x2) / 2;
                return (
                  <path
                    key={`${a}-${b}-${i}`}
                    d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                    stroke="#3a4653"
                    strokeWidth={2}
                    fill="none"
                    style={{ cursor: "pointer" }}
                    onClick={() =>
                      setEdges((es) => es.filter((_, j) => j !== i))
                    }
                  >
                    <title>
                      {a} → {b} (click to remove)
                    </title>
                  </path>
                );
              })}

            {wire && pos[wire.from] && (
              <path
                d={`M ${pos[wire.from].x + W} ${pos[wire.from].y + H / 2} L ${wire.x} ${wire.y}`}
                stroke="#4a9eff"
                strokeWidth={2}
                strokeDasharray="4 3"
                fill="none"
              />
            )}

            {/* nodes */}
            {all.map((n) => {
              const p = pos[n.name] ?? { x: 20, y: 20 };
              const on = nodes.includes(n.name);
              const ran = result?.steps?.find((s: any) => s.agent === n.name);
              return (
                <g key={n.name} transform={`translate(${p.x},${p.y})`}>
                  <rect
                    width={W}
                    height={H}
                    rx={8}
                    fill={on ? "#151b23" : "#101419"}
                    stroke={
                      ran ? "#3fb950" : on ? "#2a3441" : "#232a33"
                    }
                    strokeWidth={ran ? 2 : 1}
                    strokeDasharray={on ? undefined : "4 3"}
                    style={{ cursor: "grab" }}
                    onMouseDown={(e) => {
                      const q = svgPoint(e as any);
                      setDrag({ id: n.name, dx: q.x - p.x, dy: q.y - p.y });
                    }}
                  />
                  <text
                    x={12}
                    y={24}
                    fontSize={13}
                    fontWeight={600}
                    fill={on ? "#e6edf3" : "#5b6675"}
                    style={{ pointerEvents: "none" }}
                  >
                    {n.label}
                  </text>
                  <text
                    x={12}
                    y={42}
                    fontSize={10.5}
                    fill="#8b98a5"
                    style={{ pointerEvents: "none" }}
                  >
                    {n.question.length > 30
                      ? n.question.slice(0, 29) + "…"
                      : n.question}
                  </text>
                  {ran && (
                    <text x={W - 12} y={24} fontSize={10.5} fill="#3fb950"
                          textAnchor="end" style={{ pointerEvents: "none" }}>
                      {num(ran.duration_s, 2)}s
                    </text>
                  )}
                  {/* input port */}
                  <circle
                    cx={0}
                    cy={H / 2}
                    r={6}
                    fill={wire ? "#4a9eff" : "#2a3441"}
                    stroke="#0d1117"
                    style={{ cursor: "crosshair" }}
                    onMouseUp={() => finishWire(n.name)}
                  />
                  {/* output port */}
                  <circle
                    cx={W}
                    cy={H / 2}
                    r={6}
                    fill="#2a3441"
                    stroke="#0d1117"
                    style={{ cursor: "crosshair" }}
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      const q = svgPoint(e as any);
                      setWire({ from: n.name, x: q.x, y: q.y });
                    }}
                  />
                  {/* enable / disable */}
                  <g
                    style={{ cursor: n.removable ? "pointer" : "not-allowed" }}
                    onClick={() => toggleNode(n.name)}
                  >
                    <rect x={W - 26} y={H - 22} width={18} height={14} rx={3}
                          fill={on ? "rgba(63,185,80,.2)" : "rgba(139,152,165,.15)"} />
                    <text x={W - 17} y={H - 11} fontSize={9}
                          fill={on ? "#3fb950" : "#8b98a5"} textAnchor="middle">
                      {n.removable ? (on ? "on" : "off") : "req"}
                    </text>
                  </g>
                </g>
              );
            })}
          </svg>
        </div>
      </Panel>

      {check?.order?.length > 0 && (
        <Panel title="Execution order">
          <div className="small mono">{check.order.join("  →  ")}</div>
        </Panel>
      )}

      {result?.valid && (
        <Panel title={`Result — ${result.machine_id} at ${result.timestamp}`}>
          <div className="grid c4" style={{ marginBottom: 16 }}>
            <div className="stat">
              <div className="label">Agents run</div>
              <div className="value">{result.steps.length}</div>
              <div className="sub">{num(result.duration_s, 2)}s total</div>
            </div>
            <div className="stat">
              <div className="label">Adjudication</div>
              <div className="value" style={{ fontSize: 17 }}>
                {result.adjudication?.decision ?? "—"}
              </div>
              <div className="sub">
                deg {num(result.adjudication?.degradation_score, 2)} vs benign{" "}
                {num(result.adjudication?.confound_score, 2)}
              </div>
            </div>
            <div className="stat">
              <div className="label">Plan</div>
              <div className="value" style={{ fontSize: 17 }}>
                {result.remediation?.plan?.length ?? 0} action(s)
              </div>
              <div className="sub">{result.remediation?.mode}</div>
            </div>
            <div className="stat">
              <div className="label">Verification</div>
              <div className="value" style={{ fontSize: 17 }}>
                {result.verification?.verdict ?? "not run"}
              </div>
              <div className="sub">
                {result.verification?.checks?.length ?? 0} checks
              </div>
            </div>
          </div>

          <table>
            <thead>
              <tr>
                <th>Agent</th>
                <th>What it did</th>
                <th className="num">Time</th>
              </tr>
            </thead>
            <tbody>
              {result.steps.map((s: any) => (
                <tr key={s.agent}>
                  <td>{s.label}</td>
                  <td className="small">
                    {s.summary}
                    {s.reason && (
                      <div className="muted">{s.reason}</div>
                    )}
                  </td>
                  <td className="num">{num(s.duration_s, 2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="small muted" style={{ marginTop: 12 }}>
            Plan actions:{" "}
            {(result.remediation?.plan ?? [])
              .map((p: any) => p.intervention_id)
              .join(", ") || "none"}
          </div>
        </Panel>
      )}

      <Panel title="Node contracts">
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Question it owns</th>
              <th>Requires</th>
              <th>Produces</th>
              <th>Removable</th>
            </tr>
          </thead>
          <tbody>
            {all.map((n) => (
              <tr key={n.name}>
                <td>{n.label}</td>
                <td className="small muted">{n.question}</td>
                <td className="mono small">
                  {n.requires.join(", ") || "—"}
                  {n.optional_inputs.length > 0 && (
                    <span className="muted"> (+{n.optional_inputs.join(", ")})</span>
                  )}
                </td>
                <td className="mono small">{n.provides.join(", ")}</td>
                <td>
                  {n.removable ? (
                    <Pill kind="warn">optional</Pill>
                  ) : (
                    <Pill kind="fail">required</Pill>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
