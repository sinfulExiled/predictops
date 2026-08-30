/** Inline icons — no icon dependency, so the bundle stays self-contained. */

const S = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

function Svg({ children, size = 16 }: { children: React.ReactNode; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...S}>
      {children}
    </svg>
  );
}

export const Icon = {
  refresh: () => (
    <Svg>
      <path d="M21 12a9 9 0 1 1-2.6-6.4" />
      <path d="M21 3v6h-6" />
    </Svg>
  ),
  database: () => (
    <Svg>
      <ellipse cx="12" cy="5.5" rx="8" ry="3" />
      <path d="M4 5.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
      <path d="M4 11.5v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </Svg>
  ),
  rocket: () => (
    <Svg>
      <path d="M12 2.5c3 2 4.7 5.3 4.7 9L12 16l-4.7-4.5c0-3.7 1.7-7 4.7-9Z" />
      <path d="M7.3 11.5 5 13l1.5 4L9 15.4M16.7 11.5 19 13l-1.5 4L15 15.4" />
      <path d="M12 8.5h.01" />
    </Svg>
  ),
  target: () => (
    <Svg>
      <circle cx="12" cy="12" r="8.5" />
      <circle cx="12" cy="12" r="4.5" />
      <path d="M12 12h.01" />
    </Svg>
  ),
  trend: () => (
    <Svg>
      <path d="M3 17l6-6 4 4 8-8" />
      <path d="M14 7h7v7" />
    </Svg>
  ),
  layers: () => (
    <Svg>
      <path d="M12 3 3 8l9 5 9-5-9-5Z" />
      <path d="M3 13l9 5 9-5M3 18l9 5 9-5" />
    </Svg>
  ),
  fleet: () => (
    <Svg>
      <path d="M3 21h18M5 21V8l5-3v16M14 21V11l5-2v12" />
      <path d="M8 11h.01M8 14h.01M17 14h.01" />
    </Svg>
  ),
  chat: () => (
    <Svg>
      <path d="M21 11.5a8.4 8.4 0 0 1-9 8.4 8.4 8.4 0 0 1-3.8-.9L3 20l1.3-3.9A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5Z" />
    </Svg>
  ),
  search: () => (
    <Svg>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </Svg>
  ),
  wrench: () => (
    <Svg>
      <path d="M14.7 6.3a4 4 0 0 0 5 5l-9.4 9.4a2.1 2.1 0 0 1-3-3Z" />
      <path d="m18 3-2.5 2.5" />
    </Svg>
  ),
  graph: () => (
    <Svg>
      <circle cx="6" cy="6" r="2.5" />
      <circle cx="18" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="M8 7.5 10.5 16M16 7.5 13.5 16" />
    </Svg>
  ),
  activity: () => (
    <Svg>
      <path d="M3 12h4l3-8 4 16 3-8h4" />
    </Svg>
  ),
  lab: () => (
    <Svg>
      <path d="M9 3v6l-5 9a2 2 0 0 0 1.7 3h12.6a2 2 0 0 0 1.7-3l-5-9V3" />
      <path d="M8 3h8M7.5 15h9" />
    </Svg>
  ),
  beaker: () => (
    <Svg>
      <path d="M4 5h16M7 5v5l-3 8a1.8 1.8 0 0 0 1.6 2.6h12.8A1.8 1.8 0 0 0 20 18l-3-8V5" />
    </Svg>
  ),
  check: () => (
    <Svg>
      <path d="M20 6 9 17l-5-5" />
    </Svg>
  ),
  bell: () => (
    <Svg>
      <path d="M18 8a6 6 0 1 0-12 0c0 6-2 7-2 7h16s-2-1-2-7" />
      <path d="M10.5 20a2 2 0 0 0 3 0" />
    </Svg>
  ),
  expand: () => (
    <Svg>
      <path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5" />
    </Svg>
  ),
  download: () => (
    <Svg>
      <path d="M12 3v12M7 11l5 5 5-5M4 20h16" />
    </Svg>
  ),
  alert: () => (
    <Svg size={13}>
      <path d="M12 4 2.5 20h19L12 4Z" />
      <path d="M12 10v4M12 17h.01" />
    </Svg>
  ),
  info: () => (
    <Svg size={13}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </Svg>
  ),
  eye: () => (
    <Svg size={14}>
      <path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6Z" />
      <circle cx="12" cy="12" r="2.5" />
    </Svg>
  ),
  arrow: () => (
    <Svg size={14}>
      <path d="M5 12h13M13 6l6 6-6 6" />
    </Svg>
  ),
};

/** A tiny inline sparkline. */
export function Spark({
  values,
  color = "#8b98a5",
  width = 74,
  height = 26,
}: {
  values: number[];
  color?: string;
  width?: number;
  height?: number;
}) {
  if (!values || values.length < 2) return <svg width={width} height={height} />;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * (width - 2) + 1;
      const y = height - 2 - ((v - lo) / span) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

/** Donut gauge for the fleet health score. */
export function Gauge({ value, size = 62 }: { value: number; size?: number }) {
  const r = size / 2 - 5;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value)) / 100;
  const color = value >= 80 ? "#3fb950" : value >= 55 ? "#d29922" : "#f85149";
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#2a3441" strokeWidth={5} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={5}
        strokeLinecap="round"
        strokeDasharray={`${c * pct} ${c}`}
      />
    </svg>
  );
}

/** Donut for the contributing-factor breakdown. */
export function Donut({
  slices,
  size = 128,
}: {
  slices: { label: string; value: number; color: string }[];
  size?: number;
}) {
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  let offset = 0;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      {slices.map((s) => {
        const len = (s.value / total) * c;
        const el = (
          <circle
            key={s.label}
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={s.color}
            strokeWidth={14}
            strokeDasharray={`${len} ${c - len}`}
            strokeDashoffset={-offset}
          />
        );
        offset += len;
        return el;
      })}
    </svg>
  );
}
