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

/** Format like Python's `format(x, ".Nf")`, which rounds halves to even.
 *
 *  `toFixed` rounds halves away from zero, so a value sitting exactly on the
 *  boundary renders differently in the browser than in the CLI that produced
 *  it: the suite's F1 of 0.5625 printed as 56.2% from `evaluate.py` and the
 *  README, and 56.3% here. Same number, two conventions, and a reader
 *  comparing the two surfaces cannot tell which is wrong. The reports are the
 *  reproducible artifact, so the UI follows them.
 *
 *  The tie test reads the double's decimal expansion rather than rescaling by
 *  a power of ten. Rescaling rounds a second time: 4.55 is really
 *  4.54999...82 and must round DOWN, but 4.55 * 10 lands exactly on 45.5 and
 *  would round up, putting the nuisance rate at 4.6% against the report's 4.5%.
 */
function fixed(v: number, nd: number): string {
  if (!Number.isFinite(v)) return String(v);
  const neg = v < 0;
  const a = Math.abs(v);
  const exact = a.toFixed(20);
  const dot = exact.indexOf(".");
  const beyond = exact.slice(dot + 1).slice(nd);
  let out: string;
  if (/^50*$/.test(beyond)) {
    // Exact tie: keep the truncated value if its last kept digit is even,
    // otherwise step one unit up in the last kept place.
    const keptStr = nd > 0 ? exact.slice(0, dot + 1 + nd) : exact.slice(0, dot);
    const lastDigit = Number(keptStr[keptStr.length - 1]);
    const kept = Number(keptStr);
    out = (lastDigit % 2 === 0 ? kept : kept + 10 ** -nd).toFixed(nd);
  } else {
    out = a.toFixed(nd);
  }
  return neg && Number(out) !== 0 ? `-${out}` : out;
}

export function pct(v: number | null | undefined, nd = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${fixed(v * 100, nd)}%`;
}

export function num(v: number | null | undefined, nd = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return fixed(v, nd);
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
