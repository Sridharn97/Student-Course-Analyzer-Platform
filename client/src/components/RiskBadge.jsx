const LEVEL_CLASS = {
  Low: "badge-low",
  Medium: "badge-medium",
  High: "badge-high",
  Critical: "badge-critical",
  success: "badge-success",
  info: "badge-info",
  warning: "badge-warning",
};

const DOTS = {
  Low: "#34d399",
  Medium: "#fbbf24",
  High: "#fb923c",
  Critical: "#fb7185",
};

export default function RiskBadge({ level }) {
  const cls = LEVEL_CLASS[level] || "badge-info";
  const dot = DOTS[level];
  return (
    <span className={`badge ${cls}`}>
      {dot && <span style={{ width: 6, height: 6, borderRadius: "50%", background: dot, display: "inline-block" }} />}
      {level}
    </span>
  );
}
