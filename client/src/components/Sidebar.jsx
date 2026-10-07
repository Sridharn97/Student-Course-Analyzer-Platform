import { NavLink, Link } from "react-router-dom";
import {
  LayoutDashboard,
  Brain,
  AlertTriangle,
  BookOpen,
  FileText,
  GraduationCap,
  X,
  Upload,
} from "lucide-react";

const navItems = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/ml-models", icon: Brain, label: "Model lab" },
  { to: "/at-risk", icon: AlertTriangle, label: "Early intervention" },
  { to: "/recommendations", icon: BookOpen, label: "Course matching" },
  { to: "/reports", icon: FileText, label: "Reports" },
];

export default function Sidebar({ sessionId, isOpen, onClose }) {
  const hasSession = Boolean(sessionId);

  return (
    <aside className={`sidebar${isOpen ? " is-open" : ""}`}>
      <div className="sidebar-logo">
        <Link to="/" className="sidebar-logo-icon" aria-label="Go to dataset upload">
          <GraduationCap size={20} color="#fff" strokeWidth={2.5} />
        </Link>
        <div className="sidebar-logo-text">
          <span>Student</span> Course Analyzer
          <small>Academic intelligence</small>
        </div>
        <button className="sidebar-close" type="button" onClick={onClose} aria-label="Close navigation"><X size={20} /></button>
      </div>

      <div className="sidebar-workspace">
        <span className="sidebar-workspace-dot" />
        <div>
          <strong>{hasSession ? "Active workspace" : "Get started"}</strong>
          <span>{hasSession ? "Analytics session" : "Upload a workbook"}</span>
        </div>
      </div>

      <span className="sidebar-section-label">Explore</span>
      <nav className="sidebar-nav" aria-label="Primary navigation">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={hasSession ? to : "/"}
            className={({ isActive }) => `sidebar-link${isActive && hasSession ? " active" : ""}${!hasSession ? " disabled" : ""}`}
            style={!hasSession ? { opacity: 0.4, pointerEvents: "none" } : {}}
            onClick={onClose}
          >
            <Icon size={18} className="sidebar-link-icon" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        {hasSession ? (
          <span className="sidebar-ready"><i /> Data loaded</span>
        ) : (
          <Link to="/" className="sidebar-upload" onClick={onClose}><Upload size={14} /> Upload a file to start</Link>
        )}
      </div>
    </aside>
  );
}
