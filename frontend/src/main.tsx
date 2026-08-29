import React from "react";
import ReactDOM from "react-dom/client";
import {
  createHashRouter,
  NavLink,
  Outlet,
  RouterProvider,
} from "react-router-dom";
import "./styles.css";

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
  { to: "/", label: "Fleet Command Center", end: true },
  { to: "/assistant", label: "Assistant" },
  { to: "/investigate", label: "Machine Investigation" },
  { to: "/remediate", label: "Remediation Simulator" },
  { to: "/workflow", label: "Workflow Canvas" },
  { to: "/activity", label: "Agent Activity" },
  { to: "/lab", label: "Model Lab" },
  { to: "/experiments", label: "Experiments" },
  { to: "/evaluation", label: "Evaluation" },
];

function Shell() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>PredictOps</h1>
          <span>Predictive Failure Intelligence</span>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          Decision support only. Recommendations are simulated and require
          human approval before any action.
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
