"""Student Course Analyzer and Prediction Platform - main Streamlit app."""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from config import PAGE_OPTIONS, XGBOOST_AVAILABLE
from data_loader import load_and_process_data, ensure_df_filled, prepare_ml_data

st.set_page_config(page_title="Student Course Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 1.5rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; }
    .main-caption { color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }
    div[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    .stRadio > div { gap: 0.5rem; }
    .block-container { padding-top: 1.5rem; max-width: 1400px; }
    hr { margin: 1.5rem 0; border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Student Course Analyzer and Prediction Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="main-caption">A data science–based dashboard to evaluate and predict student performance.</p>', unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", PAGE_OPTIONS, label_visibility="collapsed")

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost not installed. Install with: pip install xgboost")

uploaded_file = st.file_uploader("Upload your Excel file (Course_Analysis_Prediction_Advanced.xlsx)", type=["xlsx"])

if uploaded_file:
    st.success("File uploaded successfully.")
    df, courses, students, enrollments, feedback, platform = load_and_process_data(uploaded_file)
    ensure_df_filled(df)

    base_features = ['Progress_Percent', 'Credits', 'Course_Rating', 'Completion_Status']
    available_features = [f for f in base_features if f in df.columns]
    X_train = X_test = y_train = y_test = None
    X_train_scaled = X_test_scaled = scaler = None
    if len(available_features) > 0:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, available_features = prepare_ml_data(df, available_features)

    if page == "Main Dashboard":
        from views.main_dashboard import render_main_dashboard
        from views.dashboard_analytics import render_dashboard_analytics
        render_main_dashboard(df)
        render_dashboard_analytics(df)

    elif page == "ML Models Comparison":
        from views.ml_models import render_ml_models
        if X_train is not None:
            render_ml_models(df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, available_features)
        else:
            st.warning("Not enough features available for prediction models.")

    elif page == "At-Risk Detection":
        from views.at_risk import render_at_risk
        render_at_risk(df)

    elif page == "Course Recommendations":
        from views.course_recommendations import render_course_recommendations
        render_course_recommendations(df)

    elif page == "Report Generator":
        from views.report_generator import render_report_generator
        render_report_generator(df)

else:
    st.info("Please upload your Excel file to begin analysis.")
