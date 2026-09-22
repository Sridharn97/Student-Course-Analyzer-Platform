import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle, RefreshCw, Trophy, Brain,
  TrendingUp, ChevronDown, ChevronUp,
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from "recharts";
import { getMLModels } from "../api";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";

const MODEL_COLORS = {
  "Random Forest": "#6366f1",
  "Gradient Boosting": "#06b6d4",
  "Ridge Regression": "#10b981",
  "XGBoost": "#f59e0b",
};

export default function MLModelsPage({ sessionId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("comparison");
  const [selectedModel, setSelectedModel] = useState("");

  const loadData = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await getMLModels(sessionId);
      setData(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to load ML model data.");
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => { loadData(); }, [loadData]);

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
        <p>Training ML models (Random Forest, Gradient Boosting, Ridge, XGBoost)…</p>
        <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>This may take 15–30 seconds</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="loading-center">
        <AlertTriangle size={36} color="#f43f5e" />
        <p>{error}</p>
        <button onClick={loadData} className="btn btn-ghost">
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    );
  }

  const { results = [], best_model, cv_results = [], feature_importance = [], features = [] } = data || {};

  // Sync selectedModel to best_model once loaded
  if (best_model && !selectedModel) setSelectedModel(best_model);

  // Build per-model feature importance for the selected model
  const modelNames = [...new Set(feature_importance.map(d => d.model))];

  const fiForModel = feature_importance
    .filter(d => d.model === selectedModel)
    .sort((a, b) => b.importance - a.importance);

  const r2ChartData = results.map(r => ({
    name: r.model,
    r2: +(r.r2 * 100).toFixed(2),
    fill: MODEL_COLORS[r.model] || "#6366f1",
  }));

  const metricsCols = [
    { key: "model", label: "Model" },
    { key: "r2", label: "R² Score", render: v => <strong style={{ color: "#818cf8" }}>{v}</strong> },
    { key: "rmse", label: "RMSE" },
    { key: "mae", label: "MAE" },
    { key: "mse", label: "MSE" },
    { key: "mape", label: "MAPE (%)" },
    {
      key: "model",
      label: "Best?",
      render: (v) =>
        v === best_model ? (
          <span className="badge badge-success">★ Best</span>
        ) : null,
    },
  ];

  const cvCols = [
    { key: "model", label: "Model" },
    { key: "mean_cv_r2", label: "Mean CV R²", render: v => <strong style={{ color: "#34d399" }}>{v}</strong> },
    { key: "std_cv_r2", label: "Std Dev" },
    { key: "min_cv_r2", label: "Min" },
    { key: "max_cv_r2", label: "Max" },
  ];

  return (
    <div className="page-enter">
      <div className="page-header">
        <h1 className="page-title">ML Model Comparison</h1>
        <p className="page-subtitle">
          Train and evaluate predictive models for student performance. Compare accuracy, stability, and feature drivers.
        </p>
      </div>

      {/* Best model banner */}
      {best_model && (
        <div className="insight-card insight-success" style={{ marginBottom: "1.5rem" }}>
          <Trophy size={18} style={{ flexShrink: 0 }} />
          <span>
            <strong>{best_model}</strong> achieved the highest R² score and is the recommended model for this dataset.
            {data?.xgboost_available === false && (
              <span style={{ marginLeft: "0.5rem", opacity: 0.7 }}>(XGBoost not available — install xgboost for more options)</span>
            )}
          </span>
        </div>
      )}

      <div className="tabs">
        <button className={`tab-btn${tab === "comparison" ? " active" : ""}`} onClick={() => setTab("comparison")}>
          Model Comparison
        </button>
        <button className={`tab-btn${tab === "cv" ? " active" : ""}`} onClick={() => setTab("cv")}>
          Cross-Validation
        </button>
        <button className={`tab-btn${tab === "importance" ? " active" : ""}`} onClick={() => setTab("importance")}>
          Feature Importance
        </button>
      </div>

      {tab === "comparison" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <ChartCard title="R² Score Comparison (higher = better)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={r2ChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} unit="%" />
                  <Tooltip
                    contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }}
                    formatter={(val) => [`${val}%`, "R² Score"]}
                  />
                  {r2ChartData.map((entry) => null)}
                  <Bar dataKey="r2" name="R² Score" radius={[6, 6, 0, 0]}>
                    {r2ChartData.map((entry, idx) => (
                      <rect key={idx} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <div className="card card-padded">
            <h3 className="section-title">Detailed Metrics Table</h3>
            <DataTable columns={metricsCols} data={results} pageSize={10} />
          </div>
        </div>
      )}

      {tab === "cv" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <ChartCard title="Cross-Validation R² Scores (5-fold)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={cv_results}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="model" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }}
                  />
                  <Legend />
                  <Bar dataKey="mean_cv_r2" fill="#6366f1" name="Mean CV R²" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="std_cv_r2" fill="#f59e0b" name="Std Dev" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>

          <div className="card card-padded">
            <h3 className="section-title">Cross-Validation Results</h3>
            <DataTable columns={cvCols} data={cv_results} pageSize={10} />
          </div>
        </div>
      )}

      {tab === "importance" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.5rem" }}>
            {modelNames.map(m => (
              <button
                key={m}
                className={`tab-btn${selectedModel === m ? " active" : ""}`}
                onClick={() => setSelectedModel(m)}
              >
                {m}
              </button>
            ))}
          </div>

          <ChartCard title={`Feature Importance — ${selectedModel}`}>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fiForModel} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis type="number" domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis type="category" dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 11 }} width={140} />
                  <Tooltip
                    contentStyle={{ background: "#0d1b2e", border: "1px solid rgba(99,102,241,0.2)", borderRadius: 8, color: "#f0f4ff" }}
                    formatter={(val) => [val.toFixed(4), "Importance"]}
                  />
                  <Bar dataKey="importance" fill={MODEL_COLORS[selectedModel] || "#6366f1"} radius={[0, 4, 4, 0]} name="Importance" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>
      )}
    </div>
  );
}
