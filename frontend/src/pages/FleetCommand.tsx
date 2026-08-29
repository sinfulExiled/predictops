import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import {
  ErrorNote,
  Loading,
  Panel,
  Pill,
  num,
  pct,
  useAsync,
} from "../components/common";
import { Donut, Gauge, Icon, Spark } from "../components/icons";

const BAND_COLOR: Record<string, string> = {
  high: "#f85149",
  watch: "#d29922",
  normal: "#3fb950",
  down: "#8b98a5",
};

const FACTOR_COLOR = ["#f85149", "#d29922", "#4a9eff", "#a371f7", "#56d4dd", "#8b98a5"];

function Delta({ v, invert = false }: { v: number; invert?: boolean }) {
  if (!v) return <span className="delta">no change</span>;
  const worse = invert ? v < 0 : v > 0;
  return (
    <span className="delta" style={{ color: worse ? "var(--bad)" : "var(--ok)" }}>
      {v > 0 ? "▲" : "▼"} {Math.abs(v)} since yesterday
    </span>
  );
}

export default function FleetCommand() {
  const nav = useNavigate();
  const [at, setAt] = useState<string | undefined>(undefined);
  const [hours, setHours] = useState(6);
  const [view, setView] = useState<"all" | "high" | "watch">("all");
  const [q, setQ] = useState("");

  const ov = useAsync(() => api.fleetOverview(at, hours), [at, hours]);
  const ag = useAsync(() => api.fleetAgents().catch(() => null), []);

  const d = ov.data;
  const machines = useMemo(() => {
    let rows = d?.machines ?? [];
    if (view === "high") rows = rows.filter((m: any) => m.status === "high");
    if (view === "watch") rows = rows.filter((m: any) => m.status === "watch");
    if (q.trim()) {
      const s = q.toLowerCase();
      rows = rows.filter(
        (m: any) =>
          m.machine_id.toLowerCase().includes(s) ||
          m.machine_type.toLowerCase().includes(s),
      );
    }
    return rows;
  }, [d, view, q]);

  if (ov.loading && !d) return <Loading what="fleet status" />;
  if (ov.error) return <ErrorNote error={ov.error} />;
  if (!d) return null;

  const atRisk = (d.machines ?? []).filter(
    (m: any) => m.status === "high" || m.status === "watch",
  );
  const trend = (d.trend ?? []).map((r: any) => ({
    ...r,
    t: String(r.t).slice(11, 16),
  }));
  const sparkOf = (band: string) => (d.trend ?? []).map((r: any) => r[band] ?? 0);

  return (
    <>
      {/* ---------------- header ---------------- */}
      <div className="topbar">
        <div className="topbar-title">
          <div className="topbar-icon">
            <Icon.fleet />
          </div>
          <div>
            <h2 style={{ margin: 0, fontSize: 21, letterSpacing: "-0.3px" }}>
              Fleet Command Center
            </h2>
            <p style={{ margin: "3px 0 0", color: "var(--muted)", fontSize: 13.5 }}>
              Every machine scored from telemetry available at the snapshot
              moment only
            </p>
          </div>
        </div>

        <div className="topbar-right">
          <div className="clock">
            <div className="t">{String(d.timestamp).slice(11, 19)}</div>
            <div className="d">{String(d.timestamp).slice(0, 10)} · snapshot</div>
          </div>
          <button
            className="ghost"
            title={`${d.alerts?.length ?? 0} band changes in the last ${d.window_hours} h`}
            style={{ position: "relative", padding: "8px 11px" }}
          >
            <Icon.bell />
            {(d.alerts?.length ?? 0) > 0 && (
              <span
                style={{
                  position: "absolute",
                  top: -5,
                  right: -5,
                  background: "var(--bad)",
                  color: "#fff",
                  fontSize: 10,
                  minWidth: 17,
                  height: 17,
                  borderRadius: 9,
                  display: "grid",
                  placeItems: "center",
                  fontWeight: 700,
                }}
              >
                {d.alerts.length}
              </span>
            )}
          </button>
          <a
            href="/api/fleet/overview"
            target="_blank"
            rel="noreferrer"
            style={{
              border: "1px solid var(--line)",
              borderRadius: 6,
              padding: "8px 15px",
              fontSize: 13,
              display: "inline-flex",
              alignItems: "center",
              gap: 7,
              color: "var(--text)",
            }}
          >
            <Icon.download />
            Export
          </a>
        </div>
      </div>

      {/* ---------------- metric cards ---------------- */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          gap: 14,
          marginBottom: 18,
        }}
      >
        {[
          { k: "high", label: "High risk", tone: "bad", inv: false },
          { k: "watch", label: "Watch", tone: "warn", inv: false },
          { k: "normal", label: "Normal", tone: "ok", inv: true },
          { k: "down", label: "Down", tone: "", inv: false },
        ].map((c) => (
          <div key={c.k} className={`metric ${c.tone}`}>
            <div className="label">{c.label}</div>
            <div
              className="big"
              style={{ color: c.tone ? BAND_COLOR[c.k] : "var(--text)" }}
            >
              {d.counts[c.k]}
            </div>
            <Delta v={d.deltas?.[c.k] ?? 0} invert={c.inv} />
            <div className="spark">
              {c.k !== "down" && <Spark values={sparkOf(c.k)} color={BAND_COLOR[c.k]} />}
            </div>
          </div>
        ))}

        <div className="metric">
          <div className="label">Fleet health score</div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div>
              <div className="big">
                {d.health.score}
                <span style={{ fontSize: 15, color: "var(--muted)" }}>/100</span>
              </div>
              <Delta v={d.health.delta} invert />
            </div>
            <div style={{ marginLeft: "auto" }}>
              <Gauge value={d.health.score} />
            </div>
          </div>
          <div
            className="small muted"
            style={{ marginTop: 4, fontSize: 10.5 }}
            title={`Defined as ${d.health.formula}`}
          >
            weighted by band ⓘ
          </div>
        </div>
      </div>

      {/* ---------------- toolbar ---------------- */}
      <div className="toolbar">
        <span className="small muted">Snapshot</span>
        <input
          style={{ width: 185 }}
          defaultValue={d.timestamp}
          key={d.timestamp}
          onKeyDown={(e) =>
            e.key === "Enter" && setAt((e.target as HTMLInputElement).value)
          }
        />
        <button className="ghost" onClick={() => setAt(undefined)}>
          Busiest moment
        </button>
        <select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
          {[3, 6, 12, 24].map((h) => (
            <option key={h} value={h}>
              Last {h} hours
            </option>
          ))}
        </select>
        <div className="seg">
          {(["all", "high", "watch"] as const).map((v) => (
            <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>
              {v === "all" ? "All machines" : v === "high" ? "High risk" : "Watch list"}
            </button>
          ))}
        </div>
        <input
          className="search"
          placeholder="Search machine ID or type…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </div>

      {/* ---------------- middle row ---------------- */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(310px, 1fr))",
          gap: 16,
          marginBottom: 18,
        }}
      >
        <Panel title={`Sites — ${d.machines.length} machines`}>
          <div className="site-grid">
            {[0, 1, 2, 3].map((site) => {
              const rows = d.machines.filter((m: any) => m.site === site);
              const high = rows.filter((m: any) => m.status === "high").length;
              return (
                <div className="site-tile" key={site}>
                  <div className="sn">Site {site + 1}</div>
                  <div style={{ fontSize: 19, fontWeight: 600, marginTop: 3 }}>
                    {rows.length}
                  </div>
                  {high > 0 && (
                    <div style={{ fontSize: 11, color: "var(--bad)" }}>
                      {high} at risk
                    </div>
                  )}
                  <div className="machine-dots">
                    {rows.slice(0, 24).map((m: any) => (
                      <span
                        key={m.machine_id}
                        title={`${m.machine_id} — ${
                          m.failure_probability === null
                            ? "down"
                            : pct(m.failure_probability)
                        }`}
                        style={{ background: BAND_COLOR[m.status] }}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="small muted" style={{ marginTop: 12 }}>
            Grouped by the site each machine belongs to in the dataset.
          </div>
        </Panel>

        <Panel title={`Machines at risk — last ${d.window_hours} hours`}>
          {/* Only the bands that need attention. Stacking `normal` on top
              swamped them: 74 against 5 made the interesting series a sliver
              one pixel high. */}
          <ResponsiveContainer width="100%" height={214}>
            <AreaChart data={trend}>
              <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
              <XAxis dataKey="t" stroke="#8b98a5" fontSize={11} minTickGap={26} />
              <YAxis stroke="#8b98a5" fontSize={11} allowDecimals={false} width={28} />
              <Tooltip
                contentStyle={{
                  background: "#151b23",
                  border: "1px solid #2a3441",
                  borderRadius: 6,
                  fontSize: 12,
                }}
              />
              <Area type="monotone" dataKey="watch" stackId="1" stroke="#d29922"
                    fill="rgba(210,153,34,.28)" name="watch" />
              <Area type="monotone" dataKey="high" stackId="1" stroke="#f85149"
                    fill="rgba(248,81,73,.34)" name="high risk" />
            </AreaChart>
          </ResponsiveContainer>
          <div className="small muted" style={{ marginTop: 8 }}>
            {d.counts.normal} machines are running normally and are not plotted.
          </div>
        </Panel>

        <Panel title="Top contributing factors">
          {atRisk.length === 0 ? (
            <div className="muted small">No machine is at risk at this moment.</div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <Donut
                slices={(d.top_factors ?? []).map((f: any, i: number) => ({
                  label: f.factor,
                  value: f.count,
                  color: FACTOR_COLOR[i % FACTOR_COLOR.length],
                }))}
              />
              <div className="legend" style={{ flex: 1 }}>
                {(d.top_factors ?? []).map((f: any, i: number) => (
                  <div className="legend-row" key={f.factor}>
                    <span
                      className="sw"
                      style={{ background: FACTOR_COLOR[i % FACTOR_COLOR.length] }}
                    />
                    {f.factor.replace(/_/g, " ")}
                    <span className="v">{pct(f.share, 0)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="small muted" style={{ marginTop: 12 }}>
            The channel that moved most on each at-risk machine, relative to what
            counts as material for that channel.
          </div>
        </Panel>
      </div>

      {/* ---------------- table + alerts ---------------- */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 2.2fr) minmax(250px, 1fr)",
          gap: 16,
          marginBottom: 18,
        }}
      >
        <Panel
          title={`Machines (${machines.length})`}
          right={<span className="small muted">click a row to investigate</span>}
        >
          <div style={{ maxHeight: 430, overflowY: "auto" }}>
            <table className="machines-table">
              <thead>
                <tr>
                  <th>Machine</th>
                  <th>Type</th>
                  <th className="num">Risk</th>
                  <th>Trend</th>
                  <th className="num">Conf.</th>
                  <th>Primary factor</th>
                  <th className="num">Time to failure</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {machines.map((m: any) => (
                  <tr
                    key={m.machine_id}
                    className="clickable"
                    onClick={() =>
                      nav(
                        `/investigate/${m.machine_id}?at=${encodeURIComponent(
                          d.timestamp,
                        )}`,
                      )
                    }
                  >
                    <td className="mono">{m.machine_id}</td>
                    <td className="muted">{m.machine_type}</td>
                    <td className="num" style={{ minWidth: 126 }}>
                      <div className="risk-cell">
                        <span style={{ width: 40, textAlign: "right" }}>
                          {m.failure_probability === null
                            ? "—"
                            : pct(m.failure_probability, 0)}
                        </span>
                        <div className="risk-bar">
                          <div
                            style={{
                              width: `${(m.failure_probability ?? 0) * 100}%`,
                              background: BAND_COLOR[m.status],
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td>
                      <Spark values={m.trend} color={BAND_COLOR[m.status]}
                             width={56} height={20} />
                    </td>
                    <td className="num muted">{pct(m.confidence, 0)}</td>
                    <td className="small">
                      {m.primary_factor ? m.primary_factor.replace(/_/g, " ") : "—"}
                    </td>
                    <td className="num small">
                      {m.eta
                        ? `${num(m.eta.window_low_h, 1)}–${num(m.eta.window_high_h, 1)} h`
                        : "—"}
                    </td>
                    <td>
                      <Pill kind={m.status}>{m.status}</Pill>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title={`Recent band changes (${d.alerts?.length ?? 0})`}>
          {(d.alerts ?? []).length === 0 ? (
            <div className="muted small">
              No machine changed band in the last {d.window_hours} hours.
            </div>
          ) : (
            (d.alerts ?? []).map((a: any, i: number) => (
              <div className="feed-item" key={`${a.machine_id}-${i}`}>
                <div className={`feed-icon ${a.to}`}>
                  <Icon.alert />
                </div>
                <div className="feed-body">
                  <div className="m" style={{ color: BAND_COLOR[a.to] }}>
                    {a.machine_id}
                  </div>
                  <div className="s">
                    {a.from} → {a.to} at {pct(a.probability, 0)}
                  </div>
                </div>
                <div className="feed-time">
                  {a.minutes_ago === 0 ? "now" : `${a.minutes_ago}m ago`}
                </div>
              </div>
            ))
          )}
          <div className="small muted" style={{ marginTop: 10 }}>
            Real band crossings computed from the scored history, not a
            notification log.
          </div>
        </Panel>
      </div>

      {/* ---------------- agent strip ---------------- */}
      <Panel
        title="Agent activity"
        right={
          <span className="small muted">
            recorded executions, from the trajectory registry
          </span>
        }
      >
        <div className="agent-strip">
          {(ag.data?.agents ?? []).map((a: any) => (
            <div className="agent-card" key={a.agent}>
              <div className="n">
                <span
                  className="dot"
                  style={{
                    background: a.executions ? "var(--ok)" : "var(--muted)",
                    marginRight: 0,
                  }}
                />
                {a.label}
              </div>
              <div className="m">
                {a.executions
                  ? `${a.executions.toLocaleString()} runs · ${num(a.mean_duration_s, 2)}s avg`
                  : "no runs recorded"}
              </div>
            </div>
          ))}
        </div>
        <div className="small muted" style={{ marginTop: 10 }}>
          Agents run on demand rather than as daemons, so this reports what they
          have actually executed rather than a live activity state.
        </div>
      </Panel>
    </>
  );
}
