export interface MachineRow {
  machine_id: string;
  failure_probability: number | null;
  confidence?: number;
  alert?: boolean;
  status: string;
  machine_type?: string;
  vibration?: number;
  temperature?: number;
  load?: number;
}

export interface Evidence {
  id: string;
  claim: string;
  channel: string;
  metric: string;
  value: number;
  unit: string;
  direction: string;
  source: string;
}

export interface Hypothesis {
  failure_type: string;
  score: number;
  classifier_probability: number;
  signature_match: number;
  historical_vote: number;
  matched_channels: string[];
  evidence_ids: string[];
}

export interface PlanStep {
  order: number;
  intervention_id: string;
  title: string;
  detail: string;
  why: string;
  risk: string;
  cost_usd: number;
  downtime_hours: number;
  requires_approval: boolean;
  is_diagnostic: boolean;
  preconditions: string[];
}

export interface SimArm {
  intervention_id: string;
  title: string;
  simulated?: boolean;
  is_simulated: boolean;
  failure_probability_simulated: number | null;
  delta_vs_no_action: number | null;
  reason_not_simulated?: string;
  cost_usd?: number;
  risk_reduction_per_1k_usd?: number;
}

export interface Check {
  id: string;
  check: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}

export interface IncidentReport {
  run_id: string;
  machine_id: string;
  timestamp: string;
  duration_s: number;
  prediction: any;
  context: any;
  investigation: any;
  degradation_case: any;
  confound_case: any;
  adjudication: any;
  remediation: any;
  simulation: any;
  verification: any;
  trajectory: TrajectoryStep[];
}

export interface TrajectoryStep {
  agent: string;
  step: number;
  action: string;
  reason: string;
  tools_used: string[];
  input_summary: string;
  output: any;
  verification: string;
  retry_count: number;
  duration_s: number;
  created_at: string;
}

export interface ExperimentRow {
  id: number;
  stage: string;
  name: string;
  model: string;
  feature_set: string;
  hypothesis: string;
  decision: string;
  learning: string;
  duration_s: number;
  params: Record<string, unknown>;
  metrics: any;
}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json() as Promise<T>;
}

export const api = {
  health: () => get<any>("/api/health"),
  machines: (at?: string) =>
    get<{
      timestamp: string;
      busiest: string;
      latest: string;
      earliest: string;
      machines: MachineRow[];
    }>(`/api/machines${at ? `?at=${encodeURIComponent(at)}` : ""}`),
  telemetry: (id: string, hours = 24, until?: string) =>
    get<{ machine_id: string; series: any[] }>(
      `/api/machines/${id}/telemetry?hours=${hours}` +
        (until ? `&until=${encodeURIComponent(until)}` : ""),
    ),
  incident: async (machine_id: string, timestamp?: string) => {
    const r = await fetch("/api/incidents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ machine_id, timestamp }),
    });
    if (!r.ok) throw new Error(await r.text());
    return (await r.json()) as IncidentReport;
  },
  assistant: async (question: string, allow_actions = true) => {
    const r = await fetch("/api/assistant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, allow_actions }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  workflowSpec: () => get<any>("/api/workflow"),
  workflowValidate: async (nodes: string[], edges: [string, string][]) => {
    const r = await fetch("/api/workflow/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes, edges }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  workflowRun: async (
    nodes: string[],
    edges: [string, string][],
    machine_id: string,
    timestamp?: string,
  ) => {
    const r = await fetch("/api/workflow/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nodes, edges, machine_id, timestamp }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  experiments: () =>
    get<{ run_id: string; experiments: ExperimentRow[] }>("/api/experiments"),
  changelog: () => get<{ markdown: string }>("/api/changelog"),
  evaluation: () => get<any>("/api/evaluation"),
  ablation: () => get<any>("/api/ablation"),
  thresholds: () =>
    get<{
      kind: string;
      feature_set: string;
      alert_threshold: number;
      investigate_threshold: number;
      lookback_steps: number;
      selection_rationale: string;
    }>("/api/thresholds"),
  scenarios: () => get<{ scenarios: any[] }>("/api/scenarios"),
  interventions: () => get<{ catalogue: any[] }>("/api/interventions"),
  trajectories: (runId?: string) =>
    get<{ run_id: string; runs: string[]; steps: TrajectoryStep[] }>(
      `/api/trajectories${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`,
    ),
};
