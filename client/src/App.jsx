import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation, Link } from "react-router-dom";
import { Menu, Upload } from "lucide-react";
import Sidebar from "./components/Sidebar";
import UploadPage from "./pages/UploadPage";
import DashboardPage from "./pages/DashboardPage";
import MLModelsPage from "./pages/MLModelsPage";
import AtRiskPage from "./pages/AtRiskPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import ReportsPage from "./pages/ReportsPage";

const pageDetails = {
  "/dashboard": ["Overview", "Student performance at a glance"],
  "/ml-models": ["Model lab", "Predictive performance"],
  "/at-risk": ["Early intervention", "Student risk signals"],
  "/recommendations": ["Course matching", "Personalized pathways"],
  "/reports": ["Reports", "Shareable academic summaries"],
};

function AppShell({ sessionId }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const [section, context] = pageDetails[location.pathname] || ["Workspace", "Student intelligence platform"];

  return (
    <div className="app-layout">
      <Sidebar sessionId={sessionId} isOpen={menuOpen} onClose={() => setMenuOpen(false)} />
      <button
        className={`nav-scrim${menuOpen ? " visible" : ""}`}
        type="button"
        aria-label="Close navigation"
        onClick={() => setMenuOpen(false)}
      />
      <div className="app-content">
        <header className="topbar">
          <div className="topbar-context">
            <button className="menu-toggle" type="button" aria-label="Open navigation" onClick={() => setMenuOpen(true)}>
              <Menu size={20} />
            </button>
            <div>
              <span className="topbar-kicker">Student Course Analyzer</span>
              <div className="topbar-title">{section}</div>
            </div>
          </div>
          <div className="topbar-actions">
            <span className={`session-status${sessionId ? " is-ready" : ""}`}>
              <i /> {sessionId ? "Dataset ready" : "No dataset"}
            </span>
            <Link to="/" className="topbar-upload"><Upload size={15} /> <span>New dataset</span></Link>
          </div>
        </header>
        <main className="app-main" aria-label={context}>
          <Routes>
            <Route path="/dashboard" element={<DashboardPage sessionId={sessionId} />} />
            <Route path="/ml-models" element={<MLModelsPage sessionId={sessionId} />} />
            <Route path="/at-risk" element={<AtRiskPage sessionId={sessionId} />} />
            <Route path="/recommendations" element={<RecommendationsPage sessionId={sessionId} />} />
            <Route path="/reports" element={<ReportsPage sessionId={sessionId} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

function App() {
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem("sessionId") || null);

  const handleSession = (id) => {
    sessionStorage.setItem("sessionId", id);
    setSessionId(id);
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<UploadPage onSession={handleSession} />} />
        <Route path="/*" element={<AppShell sessionId={sessionId} />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
