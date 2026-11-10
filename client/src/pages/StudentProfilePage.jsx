import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  Award,
  BarChart3,
  BookOpen,
  CheckCircle2,
  CircleDot,
  FileText,
  HeartPulse,
  RefreshCw,
  TrendingUp,
  UserRound,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getRecommendations, getStudentProfile } from "../api";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";
import MetricCard from "../components/MetricCard";
import RiskBadge from "../components/RiskBadge";

export default function StudentProfilePage({ sessionId }) {
  const { studentId } = useParams();
  const [profile, setProfile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadProfile = useCallback(async () => {
    if (!sessionId || !studentId) return;
    setLoading(true);
    setError(null);
    try {
      const [profileResponse, recommendationsResponse] = await Promise.all([
        getStudentProfile(sessionId, studentId),
        getRecommendations(sessionId, studentId),
      ]);
      setProfile(profileResponse.data);
      setRecommendations(recommendationsResponse.data.recommendations || []);
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || "Failed to load the student profile.");
    } finally {
      setLoading(false);
    }
  }, [sessionId, studentId]);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

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
        <p>Assembling the student profile...</p>
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#f43f5e" />
        <p>{error || "Student profile not found."}</p>
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button onClick={loadProfile} className="btn btn-ghost"><RefreshCw size={14} /> Retry</button>
          <Link to="/at-risk" className="btn btn-primary">Back to students</Link>
        </div>
      </div>
    );
  }

  const { metrics, risk, courses = [], history = [] } = profile;
  const completedCourses = courses.filter((course) => Number(course.progress || 0) >= 100);
  const ongoingCourses = courses.filter((course) => Number(course.progress || 0) < 100);
  const averageGrade = metrics.avg_score >= 90 ? "A" : metrics.avg_score >= 80 ? "B" : metrics.avg_score >= 70 ? "C" : metrics.avg_score >= 60 ? "D" : "Needs support";
  const nextActions = [];
  if (Number(metrics.avg_progress || 0) < 70) {
    nextActions.push({ icon: TrendingUp, title: "Increase course progress", text: "Complete the next pending module and keep your weekly learning pace steady." });
  }
  if (Number(metrics.avg_score || 0) < 70) {
    nextActions.push({ icon: BookOpen, title: "Review lower-scoring subjects", text: "Revisit lessons and practice activities in courses where your score is below 70%." });
  }
  if (risk.level === "High" || risk.level === "Critical") {
    nextActions.push({ icon: HeartPulse, title: "Prioritize support", text: "Focus on the risk factors below and consider asking a tutor or instructor for help." });
  }
  if (nextActions.length === 0) {
    nextActions.push({ icon: CheckCircle2, title: "Keep your momentum", text: "Your current indicators are healthy. Continue progressing and explore a recommended next course." });
  }
  if (recommendations.length > 0 && nextActions.length < 3) {
    nextActions.push({ icon: TrendingUp, title: "Explore your next course", text: `${recommendations[0].course_name} is your strongest current course match.` });
  }
  const chartData = history.map((entry) => ({
    ...entry,
    label: entry.course_name || `Course ${entry.course_id}`,
  }));
  const courseColumns = [
    { key: "course_id", label: "Course ID" },
    { key: "course_name", label: "Course" },
    { key: "score", label: "Score", render: (value) => `${Number(value || 0).toFixed(1)}%` },
    { key: "progress", label: "Progress", render: (value) => `${Number(value || 0).toFixed(1)}%` },
  ];
  const recommendationColumns = [
    { key: "course_id", label: "Course ID" },
    { key: "course_name", label: "Course" },
    { key: "recommendation_score", label: "Match", render: (value) => `${Number(value || 0).toFixed(1)}%` },
    { key: "avg_course_score", label: "Avg Score", render: (value) => `${Number(value || 0).toFixed(1)}%` },
  ];

  return (
    <div className="page-enter student-profile-page">
      <div className="profile-back-row">
        <Link to="/at-risk" className="back-link"><ArrowLeft size={15} /> Back to student list</Link>
        <div className="profile-actions">
          <Link to="/reports" className="btn btn-ghost"><FileText size={15} /> Generate report</Link>
          <Link to="/recommendations" className="btn btn-primary"><TrendingUp size={15} /> Course matching</Link>
        </div>
      </div>

      <div className="profile-hero">
        <div className="profile-identity">
          <div className="profile-avatar"><UserRound size={25} /></div>
          <div>
            <span className="dashboard-eyebrow"><i /> Student 360 profile</span>
            <h1 className="page-title">{profile.student_name}</h1>
            <p className="page-subtitle">Student ID: {profile.student_id} · {metrics.enrollments} enrollment records</p>
          </div>
        </div>
        <div className="profile-risk-summary">
          <span className="profile-risk-label">Current risk</span>
          <strong>{risk.score.toFixed(1)}%</strong>
          <RiskBadge level={risk.level} />
        </div>
      </div>

      <div className="metrics-grid personal-metrics-grid">
        <MetricCard label="Overall Performance" value={`${metrics.avg_score}%`} icon={BarChart3} color="indigo" />
        <MetricCard label="Average Grade" value={averageGrade} icon={Award} color="violet" />
        <MetricCard label="Course Progress" value={`${metrics.avg_progress}%`} icon={TrendingUp} color="cyan" />
        <MetricCard label="Completed Courses" value={completedCourses.length} icon={CheckCircle2} color="emerald" />
        <MetricCard label="Ongoing Courses" value={ongoingCourses.length} icon={BookOpen} color="amber" />
        <MetricCard label="Current Risk" value={risk.level} icon={HeartPulse} color={risk.level === "Low" ? "emerald" : "rose"} />
      </div>

      <div className="profile-grid">
        <ChartCard title="Performance history">
          <div className="chart-container profile-chart">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 12, left: -18, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(23,35,58,0.08)" />
                  <XAxis dataKey="attempt" tick={{ fill: "#60708a", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#60708a", fontSize: 10 }} />
                  <Tooltip labelFormatter={(value) => `Record ${value}`} formatter={(value, name) => [`${value}%`, name === "score" ? "Score" : "Progress"]} />
                  <Line type="monotone" dataKey="score" stroke="#6657e8" strokeWidth={3} dot={{ r: 3 }} name="score" />
                  <Line type="monotone" dataKey="progress" stroke="#0ea99c" strokeWidth={3} dot={{ r: 3 }} name="progress" />
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="table-empty">No performance records are available.</p>}
          </div>
        </ChartCard>

        <div className="card card-padded profile-risk-card">
          <div className="profile-section-heading"><ShieldIcon /><h3 className="section-title">Risk signals</h3></div>
          <div className="profile-risk-score"><strong>{risk.score.toFixed(1)}%</strong><span>Composite risk score</span></div>
          <RiskBadge level={risk.level} />
          <p className="profile-risk-reasons">{risk.reasons}</p>
          <Link to="/at-risk" className="profile-inline-link">Review intervention queue <TrendingUp size={14} /></Link>
        </div>
      </div>

      <section className="card card-padded next-actions-card" aria-labelledby="next-actions-title">
        <div className="profile-section-heading"><CircleDot size={16} /><h3 id="next-actions-title" className="section-title">Recommended next actions</h3></div>
        <div className="next-actions-list">
          {nextActions.slice(0, 3).map(({ icon: Icon, title, text }) => (
            <div className="next-action" key={title}>
              <div className="next-action-icon"><Icon size={17} /></div>
              <div><strong>{title}</strong><p>{text}</p></div>
            </div>
          ))}
        </div>
      </section>

      <div className="card card-padded">
        <div className="profile-section-heading"><BookOpen size={16} /><h3 className="section-title">Enrolled course performance</h3></div>
        <DataTable columns={courseColumns} data={courses} pageSize={10} />
      </div>

      <div className="card card-padded">
        <div className="profile-section-heading"><TrendingUp size={16} /><h3 className="section-title">Recommended next courses</h3></div>
        <DataTable columns={recommendationColumns} data={recommendations.slice(0, 8)} pageSize={8} />
      </div>
    </div>
  );
}

function ShieldIcon() {
  return <HeartPulse size={16} color="#dc4c64" />;
}
