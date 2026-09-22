export default function ChartCard({ title, children, style }) {
  return (
    <div className="card card-padded" style={style}>
      {title && <div className="card-title" style={{ marginBottom: "1rem" }}>{title}</div>}
      {children}
    </div>
  );
}
