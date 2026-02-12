"""Streamlit view renderers for the Student Course Analyzer."""

from .main_dashboard import render_main_dashboard
from .dashboard_analytics import render_dashboard_analytics
from .ml_models import render_ml_models
from .at_risk import render_at_risk
from .course_recommendations import render_course_recommendations
from .report_generator import render_report_generator

__all__ = [
    "render_main_dashboard",
    "render_dashboard_analytics",
    "render_ml_models",
    "render_at_risk",
    "render_course_recommendations",
    "render_report_generator",
]
