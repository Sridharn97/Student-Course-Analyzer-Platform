import { BarChart3 } from "lucide-react";

export default function ChartCard({ title, children, style }) {
  return (
    <div className="card chart-card" style={style}>
      {title && (
        <div className="chart-card-header">
          <div className="card-title"><span className="chart-title-icon"><BarChart3 size={15} /></span>{title}</div>
          <span className="chart-card-caption">Current data</span>
        </div>
      )}
      {children}
    </div>
  );
}
