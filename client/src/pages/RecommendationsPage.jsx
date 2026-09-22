import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, RefreshCw, BookOpen,
  Star, TrendingUp, Award,
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import { getStudents, getRecommendations } from "../api";
import MetricCard from "../components/MetricCard";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";

export default function RecommendationsPage({ sessionId }) {
  const [students, setStudents] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [recData, setRecData] = useState(null);
  const [loadingStudents, setLoadingStudents] = useState(true);
  const [loadingRecs, setLoadingRecs] = useState(false);
  const [error, setError] = useState(null);

  // Load students list
  useEffect(() => {
    if (!sessionId) return;
    setLoadingStudents(true);
    getStudents(sessionId)
      .then(res => {
        setStudents(res.data.students || []);
        if (res.data.students?.length > 0) {
          setSelectedId(String(res.data.students[0].Student_ID));
        }
      })
      .catch(err => setError(err?.response?.data?.detail || "Failed to load students."))
      .finally(() => setLoadingStudents(false));
  }, [sessionId]);

  // Load recommendations when student is selected
  const loadRecs = useCallback(async (id) => {
    if (!id || !sessionId) return;
    setLoadingRecs(true);
    setError(null);
    setRecData(null);
    try {
      const res = await getRecommendations(sessionId, id);
      setRecData(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load recommendations.");
    } finally {
      setLoadingRecs(false);
    }
  }, [sessionId]);

  useEffect(() => {
    if (selectedId) loadRecs(selectedId);
  }, [selectedId, loadRecs]);

  if (!sessionId) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#fbbf24" />
        <p>No active session. Please upload a dataset first.</p>
        <Link to="/" className="btn btn-primary">Upload Dataset</Link>
      </div>
    );
  }

  if (loadingStudents) {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <p>Loading student roster…</p>
      </div>
    );
  }

  const recCols = [
    { key: "course_id", label: "Course ID" },
    { key: "course_name", label: "Course Name" },
    {
      key: "recommendation_score",
      label: "Match Score",
      render: (v) => (
        <span style={{ fontWeight: 700, color: "#818cf8" }}>{v != null ? `${v}%` : "—"}</span>
      ),
    },
    {
      key: "avg_course_score",
      label: "Avg Score",
      render: (v) => `${Number(v || 0).toFixed(1)}%`,
    },
    {
      key: "avg_course_rating",
      label: "Rating",
      render: (v) => `${Number(v || 0).toFixed(2)} ★`,
    },
    { key: "enrollments", label: "Enrollments" },
  ];

  const enrolledCols = [
    { key: "course_id", label: "Course ID" },
    { key: "course_name", label: "Course Name" },
    {
      key: "score",
      label: "Score",
      render: (v) => (
        <span style={{ fontWeight: 700, color: "#34d399" }}>
          {v != null ? `${Number(v).toFixed(1)}%` : "—"}
        </span>
      ),
    },
  ];

  const top5Recs = (recData?.recommendations || []).slice(0, 8);

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">Course Recommendations</h1>
        <p className="page-subtitle">
          AI-powered personalized course suggestions based on student performance profiles.
        </p>
      </div>

      {/* Student Selector */}
      <div className="card card-padded" style={{ marginBottom: "1.5rem" }}>
        <label className="form-label" htmlFor="student-select">Select Student</label>
        <select
          id="student-select"
          className="form-select"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
        >
          {students.map((s) => (
            <option key={s.Student_ID} value={String(s.Student_ID)}>
              {s.Student_Name ? `${s.Student_Name} (ID: ${s.Student_ID})` : `Student ${s.Student_ID}`}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div className="insight-card insight-warning" style={{ marginBottom: "1rem" }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {loadingRecs && (
        <div className="loading-center" style={{ padding: "3rem" }}>
          <div className="spinner" />
          <p>Generating personalized recommendations…</p>
        </div>
      )}

      {!loadingRecs && recData && (
        <>
          {/* Student Overview */}
          <div className="metrics-grid" style={{ marginBottom: "1.5rem" }}>
            <MetricCard
              label="Student"
              value={recData.student_name}
              icon={BookOpen}
            />
            <MetricCard
              label="Avg Score"
              value={recData.avg_score != null ? `${recData.avg_score}%` : null}
              icon={Award}
            />
            <MetricCard
              label="Avg Rating Given"
              value={recData.avg_rating != null ? `${recData.avg_rating} / 5` : null}
              icon={Star}
            />
            <MetricCard
              label="Recommendations"
              value={recData.recommendations?.length?.toLocaleString()}
              icon={TrendingUp}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <div className="grid-2">
              {/* Enrolled Courses */}
              {recData.enrolled_courses?.length > 0 && (
                <div className="card card-padded">
                  <h3 className="section-title">
                    <BookOpen size={15} /> Currently Enrolled
                  </h3>
                  <DataTable columns={enrolledCols} data={recData.enrolled_courses} pageSize={10} />
                </div>
              )}

              {/* Top Recommendations Chart */}
              {top5Recs.length > 0 && (
                <ChartCard title="Top Recommendations — Match Score">
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={top5Recs} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                        <XAxis
                          type="number"
                          domain={[0, 100]}
                          tick={{ fill: "#94a3b8", fontSize: 10 }}
                          unit="%"
                        />
                        <YAxis
                          type="category"
                          dataKey="course_name"
                          tick={{ fill: "#94a3b8", fontSize: 10 }}
                          width={120}
                        />
                        <Tooltip
                          contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }}
                          formatter={(val) => [`${val}%`, "Match Score"]}
                        />
                        <Bar
                          dataKey="recommendation_score"
                          fill="#6366f1"
                          radius={[0, 6, 6, 0]}
                          name="Match Score"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </ChartCard>
              )}
            </div>

            {/* Full Recommendations Table */}
            <div className="card card-padded">
              <h3 className="section-title">
                <TrendingUp size={15} /> All Recommended Courses ({recData.recommendations?.length})
              </h3>
              <DataTable columns={recCols} data={recData.recommendations || []} pageSize={15} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
