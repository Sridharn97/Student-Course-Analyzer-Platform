import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Users,
  BookOpen,
  GraduationCap,
  Award,
  Star,
  Activity,
  Download,
  AlertTriangle,
  CheckCircle,
  Info,
  RefreshCw,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { getDashboard, getExportUrl } from "../api";
import MetricCard from "../components/MetricCard";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";

const PIE_COLORS = ["#10b981", "#06b6d4", "#f43f5e", "#f59e0b"];

export default function DashboardPage({ sessionId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("overview");

  const loadData = async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getDashboard(sessionId);
      setData(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load dashboard data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [sessionId]);

  if (!sessionId) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#fbbf24" />
        <p>No active dataset session found.</p>
        <Link to="/" className="btn btn-primary">
          Upload Dataset First
        </Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <p>Analyzing course data & generating metrics…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#f43f5e" />
        <p>{error}</p>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button onClick={loadData} className="btn btn-ghost">
            <RefreshCw size={14} /> Retry
          </button>
          <Link to="/" className="btn btn-primary">
            Upload New File
          </Link>
        </div>
      </div>
    );
  }

  const {
    metrics = {},
    platform_stats = [],
    score_histogram = [],
    progress_histogram = [],
    sentiment_data = [],
    sentiment_by_platform = [],
    top_students = [],
    course_performance = [],
    performance_levels = [],
    insights = [],
  } = data || {};

  const topStudentCols = [
    { key: "Student_ID", label: "Student ID" },
    { key: "Student_Name", label: "Name" },
    {
      key: "Score",
      label: "Score",
      render: (val) => (
        <span style={{ fontWeight: 700, color: "#34d399" }}>
          {val != null ? `${Number(val).toFixed(1)}%` : "—"}
        </span>
      ),
    },
    {
      key: "Progress_Percent",
      label: "Progress",
      render: (val) => `${val != null ? Number(val).toFixed(1) : "—"}%`,
    },
  ];

  const courseCols = [
    { key: "Course_ID", label: "Course ID" },
    { key: "Course_Name", label: "Course Name" },
    { key: "students_enrolled", label: "Enrollments" },
    {
      key: "avg_score",
      label: "Avg Score",
      render: (v) => `${Number(v || 0).toFixed(1)}%`,
    },
    {
      key: "std_dev",
      label: "Std Dev",
      render: (v) => Number(v || 0).toFixed(2),
    },
    {
      key: "avg_progress",
      label: "Avg Progress",
      render: (v) => `${Number(v || 0).toFixed(1)}%`,
    },
  ];

  const platformCols = [
    { key: "Platform", label: "Platform" },
    { key: "total_students", label: "Students" },
    {
      key: "avg_score",
      label: "Avg Score",
      render: (v) => `${Number(v || 0).toFixed(1)}%`,
    },
    {
      key: "median_score",
      label: "Median Score",
      render: (v) => `${Number(v || 0).toFixed(1)}%`,
    },
    {
      key: "std_dev",
      label: "Std Dev",
      render: (v) => Number(v || 0).toFixed(2),
    },
    {
      key: "avg_rating",
      label: "Avg Rating",
      render: (v) => `${Number(v || 0).toFixed(2)} ★`,
    },
  ];

  return (
    <div className="page-enter">
      {/* Page Header */}
      <div
        className="page-header"
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          flexWrap: "wrap",
          gap: "1rem",
        }}
      >
        <div>
          <h1 className="page-title">Executive Analytics Dashboard</h1>
          <p className="page-subtitle">
            Comprehensive overview of academic achievements, platform performance, and student progress.
          </p>
        </div>
        <a
          href={getExportUrl(sessionId)}
          className="btn btn-primary"
          target="_blank"
          rel="noreferrer"
          download
        >
          <Download size={16} /> Export Clean Dataset
        </a>
      </div>

      {/* KPI Cards Grid */}
      <div className="metrics-grid">
        <MetricCard
          label="Total Students"
          value={metrics.total_students?.toLocaleString()}
          icon={Users}
        />
        <MetricCard
          label="Total Courses"
          value={metrics.total_courses?.toLocaleString()}
          icon={BookOpen}
        />
        <MetricCard
          label="Total Enrollments"
          value={metrics.total_enrollments?.toLocaleString()}
          icon={GraduationCap}
        />
        <MetricCard
          label="Average Score"
          value={metrics.avg_score != null ? `${metrics.avg_score}%` : null}
          icon={Award}
        />
        <MetricCard
          label="Average Rating"
          value={metrics.avg_rating != null ? `${metrics.avg_rating} / 5` : null}
          icon={Star}
        />
        <MetricCard
          label="Average Progress"
          value={metrics.avg_progress != null ? `${metrics.avg_progress}%` : null}
          icon={Activity}
        />
      </div>

      {/* Automated Insights */}
      {insights.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "1.5rem" }}>
          {insights.map((item, idx) => (
            <div key={idx} className={`insight-card insight-${item.type}`}>
              {item.type === "success" && <CheckCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />}
              {item.type === "warning" && <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />}
              {item.type === "info" && <Info size={16} style={{ flexShrink: 0, marginTop: 2 }} />}
              <span>{item.text}</span>
            </div>
          ))}
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="tabs">
        <button
          className={`tab-btn${tab === "overview" ? " active" : ""}`}
          onClick={() => setTab("overview")}
        >
          Overview & Distributions
        </button>
        <button
          className={`tab-btn${tab === "sentiment" ? " active" : ""}`}
          onClick={() => setTab("sentiment")}
        >
          Sentiment & Feedback
        </button>
        <button
          className={`tab-btn${tab === "performance" ? " active" : ""}`}
          onClick={() => setTab("performance")}
        >
          Courses & High Achievers
        </button>
      </div>

      {/* TAB 1: Overview & Distributions */}
      {tab === "overview" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="grid-2">
            <ChartCard title="Score Distribution Histogram">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={score_histogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="range" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1b2e",
                        border: "1px solid rgba(99,102,241,0.2)",
                        borderRadius: 8,
                        color: "#f0f4ff",
                      }}
                    />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="Students" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard title="Progress Distribution (%)">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={progress_histogram}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="range" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1b2e",
                        border: "1px solid rgba(6,182,212,0.2)",
                        borderRadius: 8,
                        color: "#f0f4ff",
                      }}
                    />
                    <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Students" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>

          <ChartCard title="Platform Comparison (Score vs Progress)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={platform_stats}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="Platform" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#0d1b2e",
                      border: "1px solid rgba(99,102,241,0.2)",
                      borderRadius: 8,
                      color: "#f0f4ff",
                    }}
                  />
                  <Legend />
                  <Bar dataKey="avg_score" fill="#6366f1" name="Avg Score (%)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="avg_progress" fill="#22d3ee" name="Avg Progress (%)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <div className="card card-padded">
            <h3 className="section-title">Platform Statistics Breakdown</h3>
            <DataTable columns={platformCols} data={platform_stats} pageSize={10} />
          </div>
        </div>
      )}

      {/* TAB 2: Sentiment & Feedback */}
      {tab === "sentiment" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="grid-2">
            <ChartCard title="Overall Sentiment Breakdown">
              <div className="chart-container" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={sentiment_data}
                      dataKey="count"
                      nameKey="sentiment"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      innerRadius={55}
                      paddingAngle={4}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      {sentiment_data.map((entry, index) => {
                        const s = String(entry.sentiment || "").toLowerCase();
                        const color = s.includes("pos")
                          ? "#10b981"
                          : s.includes("neg")
                          ? "#f43f5e"
                          : "#06b6d4";
                        return <Cell key={`cell-${index}`} fill={color} />;
                      })}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: "#0d1b2e",
                        border: "1px solid rgba(99,102,241,0.2)",
                        borderRadius: 8,
                        color: "#f0f4ff",
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>

            <ChartCard title="Sentiment Distribution Across Platforms">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={sentiment_by_platform}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="Platform" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1b2e",
                        border: "1px solid rgba(99,102,241,0.2)",
                        borderRadius: 8,
                        color: "#f0f4ff",
                      }}
                    />
                    <Legend />
                    <Bar dataKey="Positive" stackId="a" fill="#10b981" />
                    <Bar dataKey="Neutral" stackId="a" fill="#06b6d4" />
                    <Bar dataKey="Negative" stackId="a" fill="#f43f5e" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>
        </div>
      )}

      {/* TAB 3: Courses & High Achievers */}
      {tab === "performance" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div className="grid-2">
            <div className="card card-padded">
              <h3 className="section-title">Top 10 High Achievers</h3>
              <DataTable columns={topStudentCols} data={top_students} pageSize={10} />
            </div>

            <ChartCard title="Average Progress by Performance Tier">
              <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performance_levels}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="level" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: "#0d1b2e",
                        border: "1px solid rgba(99,102,241,0.2)",
                        borderRadius: 8,
                        color: "#f0f4ff",
                      }}
                    />
                    <Bar dataKey="avg_progress" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Avg Progress (%)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </ChartCard>
          </div>

          <div className="card card-padded">
            <h3 className="section-title">Course Performance Directory</h3>
            <DataTable columns={courseCols} data={course_performance} pageSize={10} />
          </div>
        </div>
      )}
    </div>
  );
}
