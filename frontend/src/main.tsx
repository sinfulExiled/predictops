import React from "react";
import ReactDOM from "react-dom/client";
import {
  createHashRouter,
  NavLink,
  Outlet,
  RouterProvider,
} from "react-router-dom";
import "./styles.css";

import { api } from "./api";
import { useAsync } from "./components/common";
import { Icon } from "./components/icons";

import Assistant from "./pages/Assistant";
import FleetCommand from "./pages/FleetCommand";
import MachineInvestigation from "./pages/MachineInvestigation";
import RemediationSimulator from "./pages/RemediationSimulator";
import AgentActivity from "./pages/AgentActivity";
import ModelLab from "./pages/ModelLab";
import Experiments from "./pages/Experiments";
import Evaluation from "./pages/Evaluation";
import WorkflowCanvas from "./pages/WorkflowCanvas";

const NAV = [
  { to: "/", label: "Fleet Command Center", icon: Icon.fleet, end: true },
  { to: "/assistant", label: "Assistant", icon: Icon.chat },
  { to: "/investigate", label: "Machine Investigation", icon: Icon.search },
  { to: "/remediate", label: "Remediation Simulator", icon: Icon.wrench },
  { to: "/workflow", label: "Workflow Canvas", icon: Icon.graph },
  { to: "/activity", label: "Agent Activity", icon: Icon.activity },
  { to: "/lab", label: "Model Lab", icon: Icon.lab },
  { to: "/experiments", label: "Experiments", icon: Icon.beaker },
  { to: "/evaluation", label: "Evaluation", icon: Icon.check },
];

function SystemStatus() {
  const sys = useAsync(() => api.system().catch(() => null), []);
  const s = sys.data;
  return (
    <div className="status-panel">
      <h4>System status</h4>
      {!s ? (
        <div className="small muted">no model bundle loaded</div>
      ) : (
        <>
          <div className="status-row" style={{ marginBottom: 6 }}>
            <span>
              <span className="dot ok" />
              Operational
            </span>
          </div>
          <div className="status-row">
            <span className="k">Agents registered</span>
            <span>{s.agents_registered}</span>
          </div>
          <div className="status-row">
            <span className="k">Models loaded</span>
            <span>
              {s.models_loaded}/{s.models_total}
            </span>
          </div>
          <div className="status-row">
            <span className="k">Risk model</span>
            <span>{s.models?.risk}</span>
          </div>
          <div className="status-row">
            <span className="k">LLM provider</span>
            <span>{s.provider?.name}</span>
          </div>
          <div className="status-row">
            <span className="k">Data</span>
            <span title={s.dataset?.mode}>replay</span>
          </div>
          <div className="small muted" style={{ marginTop: 9, lineHeight: 1.45 }}>
            {s.dataset?.machines} machines at {s.dataset?.resolution_minutes} min
            resolution. Fixed synthetic history, not a live feed.
          </div>
        </>
      )}
    </div>
  );
}

function Shell() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#0d1117"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 12h3l3-8 4 16 3-8h5" />
            </svg>
          </div>
          <div>
            <h1>PredictOps</h1>
            <span>Predictive Failure Intelligence</span>
          </div>
        </div>

        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}>
              <n.icon />
              {n.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <SystemStatus />

        <div className="user-chip">
          <div className="avatar">RE</div>
          <div style={{ lineHeight: 1.3 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600 }}>
              Reliability Engineer
            </div>
            <div className="small muted">decision support only</div>
          </div>
        </div>

        <div className="sidebar-foot">
          Recommendations are simulated and require human approval before any
          action.
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}

const router = createHashRouter([
  {
    path: "/",
    element: <Shell />,
    children: [
      { index: true, element: <FleetCommand /> },
      { path: "assistant", element: <Assistant /> },
      { path: "investigate", element: <MachineInvestigation /> },
      { path: "investigate/:machineId", element: <MachineInvestigation /> },
      { path: "remediate", element: <RemediationSimulator /> },
      { path: "workflow", element: <WorkflowCanvas /> },
      { path: "activity", element: <AgentActivity /> },
      { path: "lab", element: <ModelLab /> },
      { path: "experiments", element: <Experiments /> },
      { path: "evaluation", element: <Evaluation /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
