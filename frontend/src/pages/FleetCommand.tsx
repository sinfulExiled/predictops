import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, MachineRow } from "../api";
import {
  ErrorNote,
  Loading,
  PageHead,
  Panel,
  Pill,
  RiskBar,
  Stat,
  pct,
  useAsync,
} from "../components/common";

export default function FleetCommand() {
  const nav = useNavigate();
  const [at, setAt] = useState<string | undefined>(undefined);
  const health = useAsync(() => api.health(), []);
  const fleet = useAsync(() => api.machines(at), [at]);

  if (fleet.loading) return <Loading what="fleet status" />;
  if (fleet.error) return <ErrorNote error={fleet.error} />;

  const rows: MachineRow[] = fleet.data?.machines ?? [];
  const live = rows.filter((r) => r.status !== "down");
  const high = live.filter((r) => r.status === "high");
  const watch = live.filter((r) => r.status === "watch");
  const down = rows.filter((r) => r.status === "down");
  const model = health.data?.model;

  return (
    <>
      <PageHead
        title="Fleet Command Center"
        blurb={`Every machine scored at ${fleet.data?.timestamp}. Risk is the probability of a failure starting within the next 6 hours, from telemetry available at that moment only.`}
      />

      <Panel>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <span className="small muted">Snapshot</span>
          <button
            className={at === undefined ? "" : "ghost"}
            onClick={() => setAt(undefined)}
          >
            Busiest moment
          </button>
          <button
            className={at === fleet.data?.latest ? "" : "ghost"}
            onClick={() => setAt(fleet.data?.latest)}
          >
            Latest sample
          </button>
          <input
            type="text"
            style={{ width: 190 }}
            placeholder="YYYY-MM-DD HH:MM:SS"
            defaultValue={fleet.data?.timestamp}
            onKeyDown={(e) => {
              if (e.key === "Enter") setAt((e.target as HTMLInputElement).value);
            }}
          />
          <span className="small muted">
            any timestamp in {String(fleet.data?.earliest).slice(0, 10)} to{" "}
            {String(fleet.data?.latest).slice(0, 10)} (press Enter)
          </span>
        </div>
      </Panel>

      {model && (
        <div className="banner info">
          Scoring with <strong>{model.kind}</strong> on {model.feature_set}{" "}
          features, alert threshold {model.threshold}.{" "}
          <span className="muted">{model.rationale}</span>
        </div>
      )}

      <div className="grid c4" style={{ marginBottom: 18 }}>
        <Stat
          label="High risk"
          value={high.length}
          tone={high.length ? "bad" : "ok"}
          sub="above alert threshold"
        />
        <Stat label="Watch" value={watch.length} tone="warn" sub="elevated, below threshold" />
        <Stat label="Normal" value={live.length - high.length - watch.length} tone="ok" sub="running nominally" />
        <Stat label="Down" value={down.length} sub="stopped — not scored" />
      </div>

      <Panel
        title={`Machines by risk (${rows.length})`}
        right={<span className="small muted">click a row to investigate</span>}
      >
        <table>
          <thead>
            <tr>
              <th>Machine</th>
              <th>Type</th>
              <th className="num">Risk</th>
              <th style={{ width: 150 }}></th>
              <th className="num">Confidence</th>
              <th>Status</th>
              <th className="num">Vibration</th>
              <th className="num">Temp</th>
              <th className="num">Load</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.machine_id}
                className="clickable"
                onClick={() =>
                  nav(
                    `/investigate/${r.machine_id}?at=${encodeURIComponent(
                      fleet.data?.timestamp ?? "",
                    )}`,
                  )
                }
              >
                <td className="mono">{r.machine_id}</td>
                <td className="muted">{r.machine_type}</td>
                <td className="num">
                  {r.failure_probability === null ? "—" : pct(r.failure_probability)}
                </td>
                <td>
                  {r.failure_probability !== null && (
                    <RiskBar value={r.failure_probability} />
                  )}
                </td>
                <td className="num muted">{pct(r.confidence)}</td>
                <td>
                  <Pill kind={r.status}>{r.status}</Pill>
                </td>
                <td className="num">{r.vibration ?? "—"}</td>
                <td className="num">{r.temperature ?? "—"}</td>
                <td className="num">{r.load ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </>
  );
}
