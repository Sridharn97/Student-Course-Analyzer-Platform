import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import UploadPage from "./pages/UploadPage";
import DashboardPage from "./pages/DashboardPage";
import MLModelsPage from "./pages/MLModelsPage";
import AtRiskPage from "./pages/AtRiskPage";
import RecommendationsPage from "./pages/RecommendationsPage";
import ReportsPage from "./pages/ReportsPage";

function App() {
  const [sessionId, setSessionId] = useState(() => {
    // Persist session across page reloads
    return sessionStorage.getItem("sessionId") || null;
  });

  const handleSession = (id) => {
    sessionStorage.setItem("sessionId", id);
    setSessionId(id);
  };

  return (
    <BrowserRouter>
      <div className="app-layout">
        <Routes>
          {/* Upload page has no sidebar */}
          <Route
            path="/"
            element={<UploadPage onSession={handleSession} />}
          />

          {/* Dashboard pages share sidebar layout */}
          <Route
            path="/*"
            element={
              <>
                <Sidebar sessionId={sessionId} />
                <main className="app-main">
                  <Routes>
                    <Route
                      path="/dashboard"
                      element={<DashboardPage sessionId={sessionId} />}
                    />
                    <Route
                      path="/ml-models"
                      element={<MLModelsPage sessionId={sessionId} />}
                    />
                    <Route
                      path="/at-risk"
                      element={<AtRiskPage sessionId={sessionId} />}
                    />
                    <Route
                      path="/recommendations"
                      element={<RecommendationsPage sessionId={sessionId} />}
                    />
                    <Route
                      path="/reports"
                      element={<ReportsPage sessionId={sessionId} />}
                    />
                    <Route path="*" element={<Navigate to="/" replace />} />
                  </Routes>
                </main>
              </>
            }
          />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
