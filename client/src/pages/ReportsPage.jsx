import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, FileText, Download, Copy,
  CheckCircle, RefreshCw,
} from "lucide-react";
import { getStudents, getCourses, getReport, getExportUrl } from "../api";

function renderMarkdown(text) {
  if (!text) return null;
  return text.split("\n").map((line, i) => {
    if (line.startsWith("# ")) {
      return <h2 key={i} className="report-h1">{line.slice(2)}</h2>;
    }
    if (line.startsWith("## ")) {
      return <h3 key={i} className="report-h2">{line.slice(3)}</h3>;
    }
    if (line.startsWith("- ")) {
      return (
        <div key={i} className="report-bullet">
          <span className="report-bullet-dot">•</span>
          <span dangerouslySetInnerHTML={{ __html: line.slice(2).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
        </div>
      );
    }
    if (line.startsWith("**") && line.endsWith("**")) {
      return <strong key={i} style={{ color: "var(--text-primary)" }}>{line.slice(2, -2)}</strong>;
    }
    if (line === "") return <div key={i} style={{ height: "0.5rem" }} />;
    return (
      <p key={i} style={{ margin: 0 }}
        dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }}
      />
    );
  });
}

export default function ReportsPage({ sessionId }) {
  const [students, setStudents] = useState([]);
  const [courses, setCourses] = useState([]);
  const [reportType, setReportType] = useState("student");
  const [selectedId, setSelectedId] = useState("");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMeta, setLoadingMeta] = useState(true);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoadingMeta(true);
    Promise.all([getStudents(sessionId), getCourses(sessionId)])
      .then(([sRes, cRes]) => {
        const s = sRes.data.students || [];
        const c = cRes.data.courses || [];
        setStudents(s);
        setCourses(c);
        if (s.length > 0 && reportType === "student") setSelectedId(String(s[0].Student_ID));
        if (c.length > 0 && reportType === "course") setSelectedId(String(c[0].Course_ID));
      })
      .catch(() => setError("Failed to load students/courses."))
      .finally(() => setLoadingMeta(false));
  }, [sessionId]);

  // Update selectedId when reportType changes
  useEffect(() => {
    if (reportType === "student" && students.length > 0) {
      setSelectedId(String(students[0].Student_ID));
    } else if (reportType === "course" && courses.length > 0) {
      setSelectedId(String(courses[0].Course_ID));
    }
    setReport(null);
    setError(null);
  }, [reportType]);

  const generateReport = async () => {
    if (!selectedId) return;
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const res = await getReport(sessionId, reportType, selectedId);
      setReport(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to generate report.");
    } finally {
      setLoading(false);
    }
  };

  const copyReport = () => {
    if (!report?.report) return;
    navigator.clipboard.writeText(report.report).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const downloadReport = () => {
    if (!report?.report) return;
    const blob = new Blob([report.report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${reportType}-report-${selectedId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!sessionId) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#fbbf24" />
        <p>No active session. Please upload a dataset first.</p>
        <Link to="/" className="btn btn-primary">Upload Dataset</Link>
      </div>
    );
  }

  if (loadingMeta) {
    return (
      <div className="loading-center">
        <div className="spinner" />
        <p>Loading report configuration…</p>
      </div>
    );
  }

  const options = reportType === "student"
    ? students.map(s => ({ value: String(s.Student_ID), label: s.Student_Name ? `${s.Student_Name} (${s.Student_ID})` : `Student ${s.Student_ID}` }))
    : courses.map(c => ({ value: String(c.Course_ID), label: c.Course_Name ? `${c.Course_Name} (${c.Course_ID})` : `Course ${c.Course_ID}` }));

  return (
    <div className="page-enter">
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 className="page-title">Report Generator</h1>
          <p className="page-subtitle">
            Generate detailed academic performance reports for students or courses.
          </p>
        </div>
        <a href={getExportUrl(sessionId)} className="btn btn-ghost" target="_blank" rel="noreferrer" download>
          <Download size={15} /> Export Full Dataset (CSV)
        </a>
      </div>

      {/* Config Card */}
      <div className="card card-padded" style={{ marginBottom: "1.5rem" }}>
        <div className="grid-2" style={{ gap: "1rem" }}>
          <div>
            <label className="form-label">Report Type</label>
            <div className="tabs" style={{ margin: 0 }}>
              <button
                className={`tab-btn${reportType === "student" ? " active" : ""}`}
                onClick={() => setReportType("student")}
              >
                Student Report
              </button>
              <button
                className={`tab-btn${reportType === "course" ? " active" : ""}`}
                onClick={() => setReportType("course")}
              >
                Course Report
              </button>
            </div>
          </div>
          <div>
            <label className="form-label" htmlFor="report-target">
              {reportType === "student" ? "Select Student" : "Select Course"}
            </label>
            <select
              id="report-target"
              className="form-select"
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
            >
              {options.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ marginTop: "1.25rem" }}>
          <button
            className="btn btn-primary"
            onClick={generateReport}
            disabled={loading || !selectedId}
          >
            {loading ? (
              <><div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Generating…</>
            ) : (
              <><FileText size={16} /> Generate Report</>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="insight-card insight-warning" style={{ marginBottom: "1rem" }}>
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Report Output */}
      {report && (
        <div className="card" style={{ overflow: "hidden" }}>
          {/* Report Toolbar */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "1rem 1.5rem",
            borderBottom: "1px solid var(--border-subtle)",
            flexWrap: "wrap",
            gap: "0.75rem",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <FileText size={16} color="#818cf8" />
              <span style={{ fontWeight: 600, fontSize: "0.9rem" }}>
                {reportType === "student" ? "Student" : "Course"} Performance Report
              </span>
            </div>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button className="btn btn-ghost" onClick={copyReport} style={{ padding: "0.4rem 0.875rem", fontSize: "0.8rem" }}>
                {copied ? <CheckCircle size={14} color="#34d399" /> : <Copy size={14} />}
                {copied ? "Copied!" : "Copy"}
              </button>
              <button className="btn btn-ghost" onClick={downloadReport} style={{ padding: "0.4rem 0.875rem", fontSize: "0.8rem" }}>
                <Download size={14} /> Download .txt
              </button>
            </div>
          </div>

          {/* Summary metrics */}
          {report.summary && (
            <div style={{ padding: "1rem 1.5rem", borderBottom: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                {(report.summary.Metric || []).map((metric, i) => (
                  <div key={i} style={{
                    background: "rgba(99,102,241,0.06)",
                    border: "1px solid rgba(99,102,241,0.15)",
                    borderRadius: 8,
                    padding: "0.6rem 1rem",
                    minWidth: 120,
                  }}>
                    <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.2rem" }}>
                      {metric}
                    </div>
                    <div style={{ fontWeight: 700, color: "var(--indigo-light)", fontSize: "1rem" }}>
                      {report.summary.Value?.[i] ?? "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Report body */}
          <div style={{ padding: "1.5rem", lineHeight: 1.8, fontSize: "0.9rem", color: "var(--text-secondary)", maxHeight: "60vh", overflowY: "auto" }}>
            {renderMarkdown(report.report)}
          </div>
        </div>
      )}
    </div>
  );
}
