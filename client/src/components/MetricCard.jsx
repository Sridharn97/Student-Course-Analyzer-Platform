export default function MetricCard({ label, value, icon: Icon, color = "indigo" }) {
  return (
    <div className={`metric-card metric-${color}`}>
      <div className="metric-heading">
        <div className="metric-label">{label}</div>
        {Icon && <div className="metric-icon"><Icon size={19} /></div>}
      </div>
      <div className="metric-value">{value ?? "—"}</div>
      <div className="metric-baseline"><span /> Current dataset</div>
    </div>
  );
}
