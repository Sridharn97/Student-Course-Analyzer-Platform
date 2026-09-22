import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, GraduationCap, BarChart3, Brain, AlertTriangle, CheckCircle, Sparkles } from "lucide-react";
import { uploadFile, loadSample } from "../api";

const FEATURES = [
  { icon: BarChart3, title: "Real-time Analytics", desc: "Interactive charts for platform, sentiment and performance data" },
  { icon: Brain, title: "ML Model Comparison", desc: "Compare Random Forest, Gradient Boosting, XGBoost & Ridge models" },
  { icon: AlertTriangle, title: "At-Risk Detection", desc: "Identify students needing intervention with adjustable thresholds" },
  { icon: GraduationCap, title: "Course Recommendations", desc: "Personalized suggestions based on student performance profiles" },
];

export default function UploadPage({ onSession }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const inputRef = useRef();
  const navigate = useNavigate();

  const handleFile = async (file) => {
    if (!file) return;
    if (!file.name.endsWith(".xlsx")) {
      setError("Please upload an Excel (.xlsx) file.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const { data } = await uploadFile(file);
      setSuccess(data);
      onSession(data.session_id);
      setTimeout(() => navigate("/dashboard"), 1200);
    } catch (e) {
      setError(e?.response?.data?.detail || "Upload failed. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setError(null);
    setLoading(true);
    try {
      const { data } = await loadSample();
      setSuccess(data);
      onSession(data.session_id);
      setTimeout(() => navigate("/dashboard"), 1000);
    } catch (e) {
      setError(e?.response?.data?.detail || "Failed to load demo dataset. Make sure the backend server is running.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="page-enter" style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "3rem 1.5rem" }}>
      {/* Hero */}
      <div style={{ textAlign: "center", marginBottom: "3rem", maxWidth: 640 }}>
        <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 72, height: 72, borderRadius: 20, background: "linear-gradient(135deg, #6366f1, #06b6d4)", boxShadow: "0 8px 32px rgba(99,102,241,0.4)", marginBottom: "1.5rem" }}>
          <GraduationCap size={36} color="#fff" />
        </div>
        <h1 style={{ fontSize: "2.5rem", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.2, background: "linear-gradient(135deg, #f0f4ff 30%, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Student Course Analyzer
        </h1>
        <p style={{ marginTop: "0.75rem", color: "var(--text-secondary)", fontSize: "1.05rem" }}>
          Upload your dataset to unlock AI-powered analytics, ML predictions, and student insights.
        </p>
      </div>

      {/* Upload card */}
      <div className="card" style={{ width: "100%", maxWidth: 560, padding: "2.5rem" }}>
        <div
          className={`upload-zone${dragging ? " drag-over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept=".xlsx" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
          <div className="upload-icon">
            {loading ? <div className="spinner" /> : <Upload size={32} color="#6366f1" />}
          </div>
          <div className="upload-title">
            {loading ? "Processing your data…" : dragging ? "Drop it here!" : "Drop your Excel file here"}
          </div>
          <p className="upload-subtitle" style={{ marginTop: "0.4rem" }}>
            {loading ? "Running data pipelines…" : "or click to browse · .xlsx format required"}
          </p>
          <p style={{ marginTop: "1rem", fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Expected sheets: Courses, Students, Enrollments, Feedback, Platform_Performance
          </p>
        </div>

        <div style={{ marginTop: "1.25rem", textAlign: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", margin: "0.75rem 0", color: "var(--text-muted)", fontSize: "0.75rem" }}>
            <span style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
            <span>OR QUICK START</span>
            <span style={{ flex: 1, height: 1, background: "var(--border-subtle)" }} />
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            style={{ width: "100%", justifyContent: "center", gap: "0.5rem" }}
            onClick={handleLoadSample}
            disabled={loading}
          >
            <Sparkles size={16} color="#818cf8" />
            Load Sample Dataset (Course Analysis Prediction)
          </button>
        </div>

        {error && (
          <div className="insight-card insight-warning" style={{ marginTop: "1rem" }}>
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
            {error}
          </div>
        )}

        {success && (
          <div className="insight-card insight-success" style={{ marginTop: "1rem" }}>
            <CheckCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
            <span>
              Loaded <strong>{success.rows.toLocaleString()}</strong> rows ·{" "}
              <strong>{success.students}</strong> students ·{" "}
              <strong>{success.courses}</strong> courses — redirecting…
            </span>
          </div>
        )}
      </div>

      {/* Feature grid */}
      <div className="grid-2" style={{ maxWidth: 700, width: "100%", marginTop: "2.5rem" }}>
        {FEATURES.map(({ icon: Icon, title, desc }) => (
          <div key={title} className="card" style={{ padding: "1.25rem 1.5rem", display: "flex", gap: "0.875rem", alignItems: "flex-start" }}>
            <div style={{ flexShrink: 0, width: 38, height: 38, borderRadius: 10, background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Icon size={18} color="#818cf8" />
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--text-primary)", marginBottom: "0.2rem" }}>{title}</div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", lineHeight: 1.5 }}>{desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
