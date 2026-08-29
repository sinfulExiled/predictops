import React from "react";

export function PageHead({ title, blurb }: { title: string; blurb: string }) {
  return (
    <div className="page-head">
      <h2>{title}</h2>
      <p>{blurb}</p>
    </div>
  );
}

export function Panel({
  title,
  children,
  right,
}: {
  title?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="panel">
      {title && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <h3 style={{ marginBottom: 14 }}>{title}</h3>
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  tone?: "ok" | "warn" | "bad";
}) {
  const color =
    tone === "bad"
      ? "var(--bad)"
      : tone === "warn"
        ? "var(--warn)"
        : tone === "ok"
          ? "var(--ok)"
          : undefined;
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value" style={{ color }}>
        {value}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Pill({ kind, children }: { kind: string; children: React.ReactNode }) {
  return <span className={`pill ${kind}`}>{children}</span>;
}

export function RiskBar({ value }: { value: number }) {
  const color =
    value >= 0.6 ? "var(--bad)" : value >= 0.3 ? "var(--warn)" : "var(--ok)";
  return (
    <div className="bar">
      <div style={{ width: `${Math.min(100, value * 100)}%`, background: color }} />
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <div className="spinner">Loading {what}…</div>;
}

export function ErrorNote({ error }: { error: string }) {
  return (
    <div className="banner bad">
      <strong>Could not load.</strong> {error}
      <div className="small muted" style={{ marginTop: 6 }}>
        Is the API running? <code>uvicorn predictops.api.app:app --port 8000</code>
      </div>
    </div>
  );
}

export function pct(v: number | null | undefined, nd = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(nd)}%`;
}

export function num(v: number | null | undefined, nd = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(nd);
}

/** Small hook: fetch once, expose {data, error, loading}. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = React.useState<T | null>(null);
  const [error, setError] = React.useState("");
  const [loading, setLoading] = React.useState(true);
  React.useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    fn()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message ?? e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}
