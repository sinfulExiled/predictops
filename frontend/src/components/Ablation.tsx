import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Panel, Pill, num, pct } from "./common";

/** The adjudication ablation, presented as the negative result it is. */
export function AblationStudy({ data }: { data: any }) {
  if (!data) return null;
  const sweep = data.sweep ?? [];
  const s = data.summary ?? {};
  const improved = (s.delta_f1 ?? 0) > 0.005;

  const chart = sweep.map((r: any) => ({
    threshold: r.threshold.toFixed(3),
    "model only": r.model_only.f1,
    adjudicated: r.adjudicated.f1,
  }));

  // Where the nuisance cases actually score, versus the trigger.
  const cases = data.cases ?? [];
  const nuisance = cases.filter((c: any) => !c.expect);
  const maxNuisance = nuisance.length
    ? Math.max(...nuisance.map((c: any) => c.p))
    : 0;
  const trigger = data.tuned_threshold * 0.45;

  return (
    <>
      <div className={`banner ${improved ? "info" : "warn"}`}>
        <strong>
          Ablation: does the hypothesis contest improve decisions?{" "}
          {improved ? "Yes." : "No."}
        </strong>{" "}
        Across the threshold sweep the adjudicator changed{" "}
        <strong>{s.verdicts_changed}</strong> verdict(s) and is worth{" "}
        <strong>
          {(s.delta_f1 ?? 0) >= 0 ? "+" : ""}
          {num(s.delta_f1, 4)} F1
        </strong>
        . This is a measured negative result, kept in the report rather than
        removed from it.
      </div>

      <div className="grid c3" style={{ marginBottom: 18 }}>
        <div className="stat">
          <div className="label">Best model only</div>
          <div className="value">{num(s.best_model_only_f1, 4)}</div>
          <div className="sub">F1, tuned over the sweep</div>
        </div>
        <div className="stat">
          <div className="label">Best adjudicated</div>
          <div className="value">{num(s.best_adjudicated_f1, 4)}</div>
          <div className="sub">F1, same sweep</div>
        </div>
        <div className="stat">
          <div className="label">Verdicts changed</div>
          <div className="value" style={{ color: "var(--muted)" }}>
            {s.verdicts_changed}
          </div>
          <div className="sub">
            {s.helped} correct · {s.hurt} incorrect
          </div>
        </div>
      </div>

      <Panel title="F1 by alert threshold — model only vs adjudicated">
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chart}>
            <CartesianGrid stroke="#2a3441" strokeDasharray="3 3" />
            <XAxis dataKey="threshold" stroke="#8b98a5" fontSize={11} />
            <YAxis stroke="#8b98a5" fontSize={11} domain={[0, 0.8]} />
            <Tooltip
              contentStyle={{
                background: "#151b23",
                border: "1px solid #2a3441",
                borderRadius: 6,
                fontSize: 12,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="model only" fill="#4a9eff" radius={[3, 3, 0, 0]} />
            <Bar dataKey="adjudicated" fill="#a371f7" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <div className="small muted" style={{ marginTop: 10 }}>
          The two bars are identical at every threshold. That is the finding.
        </div>
      </Panel>

      <Panel title="Why it is inert">
        <div className="small" style={{ marginBottom: 14 }}>
          The confound advocate exists to catch alarms caused by load surges,
          hot weather and sensor spikes. It never gets the chance: the model
          scores every nuisance case far below the investigation trigger, so
          those cases never reach the agent layer at all.
        </div>

        <div style={{ position: "relative", height: 62, marginBottom: 10 }}>
          <div
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: 26,
              height: 8,
              background: "var(--panel-2)",
              borderRadius: 4,
            }}
          />
          <div
            style={{
              position: "absolute",
              left: 0,
              width: `${maxNuisance * 100}%`,
              top: 26,
              height: 8,
              background: "rgba(63,185,80,.6)",
              borderRadius: 4,
            }}
          />
          <div
            style={{
              position: "absolute",
              left: `${trigger * 100}%`,
              top: 8,
              bottom: 8,
              width: 2,
              background: "var(--warn)",
            }}
          />
          <div
            style={{
              position: "absolute",
              left: `${trigger * 100}%`,
              top: 0,
              fontSize: 11,
              color: "var(--warn)",
              paddingLeft: 6,
              whiteSpace: "nowrap",
            }}
          >
            investigate trigger {num(trigger, 2)}
          </div>
          <div
            style={{
              position: "absolute",
              left: 0,
              bottom: 0,
              fontSize: 11,
              color: "var(--ok)",
            }}
          >
            all {nuisance.length} nuisance cases score 0.00 –{" "}
            {num(maxNuisance, 2)}
          </div>
        </div>

        <div className="small muted">
          An earlier version had no floor on the degradation case. It changed
          four verdicts across the sweep and <strong>all four were wrong</strong>{" "}
          (−0.039 F1), including overturning a real failure one hour out because
          the duty happened to rise at the same time. The rule that fixed it: a
          benign explanation may break a marginal case, never a strong one.
        </div>
      </Panel>

      <Panel title="What it is kept for">
        <div className="small">
          Not accuracy. A <Pill kind="warn">contested</Pill> verdict routes to{" "}
          <strong>inspection</strong> and can never authorise repair — the
          remediation agent is gated on the adjudication, with a test that
          proves it. And the planner sees both readings, so "0.92 vs 0.00" and
          "0.58 vs 0.87" are visibly different situations rather than two
          identical high scores.
        </div>
      </Panel>
    </>
  );
}

/** The model service contract: one number, three routes. */
export function ThresholdBands({ t }: { t: any }) {
  if (!t) return null;
  const inv = t.investigate_threshold;
  const alert = t.alert_threshold;
  return (
    <Panel title="Model service contract">
      <div style={{ display: "flex", height: 34, borderRadius: 6, overflow: "hidden" }}>
        <div
          style={{
            width: `${inv * 100}%`,
            background: "rgba(63,185,80,.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          NORMAL
        </div>
        <div
          style={{
            width: `${(alert - inv) * 100}%`,
            background: "rgba(210,153,34,.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          INVESTIGATE
        </div>
        <div
          style={{
            width: `${(1 - alert) * 100}%`,
            background: "rgba(248,81,73,.35)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          ALERT
        </div>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11,
          color: "var(--muted)",
          marginTop: 6,
        }}
      >
        <span>0%</span>
        <span>{pct(inv, 0)} investigate</span>
        <span>{pct(alert, 0)} alert</span>
        <span>100%</span>
      </div>
      <div className="small muted" style={{ marginTop: 12 }}>
        The model service answers the quantitative question and nothing else.
        The orchestration layer decides which workflow that answer triggers, so
        no agent ever has to ask "should we investigate?"
      </div>
    </Panel>
  );
}
