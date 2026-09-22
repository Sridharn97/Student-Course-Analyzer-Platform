import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, RefreshCw, ShieldAlert, Users,
  BarChart3, List,
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar,
  PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine,
} from "recharts";
import { getAtRisk } from "../api";
import MetricCard from "../components/MetricCard";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";
import RiskBadge from "../components/RiskBadge";

const RISK_COLORS = {
  Low: "#34d399",
  Medium: "#fbbf24",
  High: "#fb923c",
  Critical: "#fb7185",
};

export default function AtRiskPage({ sessionId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(50);
  const [committed, setCommitted] = useState(50);
  const [view, setView] = useState("table");

  const loadData = useCallback(async (t) => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getAtRisk(sessionId, t);
      setData(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load at-risk data.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => { loadData(committed); }, [loadData, committed]);

  if (!sessionId) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#fbbf24" />
        <p>No active session. Please upload a dataset first.</p>
        <Link to="/" className="btn btn-primary">Upload Dataset</Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <p>Computing risk scores for all students…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#f43f5e" />
        <p>{error}</p>
        <button onClick={() => loadData(committed)} className="btn btn-ghost">
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    );
  }

  const {
    stats = {},
    at_risk_students = [],
    all_students = [],
    risk_histogram = [],
    risk_level_counts = [],
    at_risk_level_counts = [],
    common_risk_factors = [],
  } = data || {};

  const atRiskCols = [
    { key: "Student_ID", label: "ID" },
    { key: "Student_Name", label: "Name" },
    {
      key: "Risk_Score",
      label: "Risk Score",
      render: (v) => (
        <span style={{ fontWeight: 700, color: v >= 70 ? "#fb7185" : v >= 50 ? "#fb923c" : "#fbbf24" }}>
          {v != null ? `${v}%` : "—"}
        </span>
      ),
    },
    {
      key: "Risk_Level",
      label: "Level",
      render: (v) => <RiskBadge level={v} />,
    },
    {
      key: "Avg_Progress",
      label: "Avg Progress",
      render: (v) => `${v != null ? Number(v).toFixed(1) : "—"}%`,
    },
    {
      key: "Avg_Score",
      label: "Avg Score",
      render: (v) => `${v != null ? Number(v).toFixed(1) : "—"}%`,
    },
    { key: "Sentiment", label: "Sentiment" },
    { key: "Risk_Reasons", label: "Risk Factors" },
  ];

  const allStudentCols = [
    { key: "Student_ID", label: "ID" },
    { key: "Student_Name", label: "Name" },
    {
      key: "Risk_Score",
      label: "Risk Score",
      render: (v) => (
        <span style={{ fontWeight: 700, color: v >= 70 ? "#fb7185" : v >= 50 ? "#fb923c" : v >= 30 ? "#fbbf24" : "#34d399" }}>
          {v != null ? `${v}%` : "—"}
        </span>
      ),
    },
    {
      key: "Risk_Level",
      label: "Level",
      render: (v) => <RiskBadge level={v} />,
    },
    {
      key: "Avg_Progress",
      label: "Progress",
      render: (v) => `${v != null ? Number(v).toFixed(1) : "—"}%`,
    },
    {
      key: "Avg_Score",
      label: "Score",
      render: (v) => `${v != null ? Number(v).toFixed(1) : "—"}%`,
    },
  ];

  const factorCols = [
    { key: "factor", label: "Risk Factor" },
    { key: "count", label: "Affected Students" },
  ];

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">At-Risk Student Detection</h1>
        <p className="page-subtitle">
          Identify students who may need academic intervention using composite risk scores.
        </p>
      </div>

      {/* Threshold slider */}
      <div className="card card-padded" style={{ marginBottom: "1.5rem" }}>
        <div className="slider-wrapper">
          <div className="slider-header">
            <label className="form-label" style={{ margin: 0 }}>Risk Threshold</label>
            <span className="slider-value">{threshold}%</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            onMouseUp={(e) => setCommitted(Number(e.target.value))}
            onTouchEnd={(e) => setCommitted(Number(e.target.value))}
            style={{
              background: `linear-gradient(to right, #6366f1 ${threshold}%, rgba(255,255,255,0.1) ${threshold}%)`,
            }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-muted)" }}>
            <span>0% (All Students)</span>
            <span>Students with Risk ≥ {committed}% shown below</span>
            <span>100% (Most Critical)</span>
          </div>
        </div>
      </div>

      {/* KPI Metrics */}
      <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))" }}>
        <MetricCard label="Total Students" value={stats.total?.toLocaleString()} icon={Users} />
        <MetricCard label="At-Risk Count" value={stats.at_risk_count?.toLocaleString()} icon={ShieldAlert} />
        <MetricCard label="At-Risk %" value={stats.at_risk_pct != null ? `${stats.at_risk_pct}%` : null} icon={BarChart3} />
        <MetricCard label="Critical Cases" value={stats.critical_count?.toLocaleString()} icon={AlertTriangle} />
        <MetricCard label="Avg Risk Score" value={stats.avg_risk_score != null ? `${stats.avg_risk_score}%` : null} />
        <MetricCard label="Max Risk Score" value={stats.max_risk_score != null ? `${stats.max_risk_score}%` : null} />
      </div>

      <div className="tabs">
        <button className={`tab-btn${view === "table" ? " active" : ""}`} onClick={() => setView("table")}>
          At-Risk Students
        </button>
        <button className={`tab-btn${view === "all" ? " active" : ""}`} onClick={() => setView("all")}>
          All Students
        </button>
        <button className={`tab-btn${view === "charts" ? " active" : ""}`} onClick={() => setView("charts")}>
          Risk Charts
        </button>
      </div>

      {view === "table" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {at_risk_students.length === 0 ? (
            <div className="insight-card insight-success">
              <ShieldAlert size={16} />
              <span>No students exceed the {committed}% risk threshold. Lower the threshold to show more students.</span>
            </div>
          ) : (
            <div className="card card-padded">
              <h3 className="section-title">
                <ShieldAlert size={16} style={{ color: "#f43f5e" }} />
                {at_risk_students.length} Students At Risk (≥{committed}% threshold)
              </h3>
              <DataTable columns={atRiskCols} data={at_risk_students} pageSize={15} />
            </div>
          )}

          {common_risk_factors.length > 0 && (
            <div className="card card-padded">
              <h3 className="section-title">Common Risk Factors</h3>
              <DataTable columns={factorCols} data={common_risk_factors} pageSize={10} />
            </div>
          )}
        </div>
      )}

      {view === "all" && (
        <div className="card card-padded">
          <h3 className="section-title">
            <List size={16} /> All Students — Risk Overview
          </h3>
          <DataTable columns={allStudentCols} data={all_students} pageSize={20} />
        </div>
      )}

      {view === "charts" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="grid-2">
            <ChartCard title="Risk Score Distribution">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={risk_histogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="range" tick={{ fill: "#94a3b8", fontSize: 9 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }}
                    />
                    <ReferenceLine x={`${committed}`} stroke="#f43f5e" strokeDasharray="4 4" label={{ value: `Threshold ${committed}%`, fill: "#f43f5e", fontSize: 10 }} />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Students" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard title="Risk Level Breakdown (All Students)">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={risk_level_counts}
                      dataKey="count"
                      nameKey="risk_level"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      innerRadius={50}
                      paddingAngle={3}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {risk_level_counts.map((entry, idx) => (
                        <Cell key={idx} fill={RISK_COLORS[entry.risk_level] || "#6366f1"} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>

          {at_risk_level_counts.length > 0 && (
            <ChartCard title={`Risk Level Breakdown (At-Risk — ≥${committed}%)`}>
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={at_risk_level_counts}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="risk_level" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }} />
                    <Bar dataKey="count" name="Students" radius={[6, 6, 0, 0]}>
                      {at_risk_level_counts.map((entry, idx) => (
                        <rect key={idx} fill={RISK_COLORS[entry.risk_level] || "#6366f1"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          )}
        </div>
      )}
    </div>
  );
}
