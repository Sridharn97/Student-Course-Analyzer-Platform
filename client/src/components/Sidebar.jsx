import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Brain,
  AlertTriangle,
  BookOpen,
  FileText,
  GraduationCap,
} from "lucide-react";

const navItems = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/ml-models", icon: Brain, label: "ML Models" },
  { to: "/at-risk", icon: AlertTriangle, label: "At-Risk Detection" },
  { to: "/recommendations", icon: BookOpen, label: "Course Recs" },
  { to: "/reports", icon: FileText, label: "Reports" },
];

export default function Sidebar({ sessionId }) {
  const location = useLocation();
  const hasSession = Boolean(sessionId);

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <GraduationCap size={20} color="#fff" strokeWidth={2.5} />
        </div>
        <div className="sidebar-logo-text">
          Student Course<br />Analyzer
        </div>
      </div>

      <span className="sidebar-section-label">Navigation</span>

      <nav className="sidebar-nav">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={hasSession ? to : "/"}
            className={({ isActive }) =>
              `sidebar-link${isActive && hasSession ? " active" : ""}${!hasSession ? " disabled" : ""}`
            }
            style={!hasSession ? { opacity: 0.4, pointerEvents: "none" } : {}}
          >
            <Icon size={18} className="sidebar-link-icon" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        {hasSession ? (
          <span style={{ color: "#34d399", fontSize: "0.75rem" }}>✓ Data loaded</span>
        ) : (
          <span>Upload a file to start</span>
        )}
      </div>
    </aside>
  );
}
