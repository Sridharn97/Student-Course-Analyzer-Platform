export default function MetricCard({ label, value, icon: Icon, color = "indigo" }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value ?? "—"}</div>
      {Icon && (
        <div className="metric-icon">
          <Icon size={48} />
        </div>
      )}
    </div>
  );
}
