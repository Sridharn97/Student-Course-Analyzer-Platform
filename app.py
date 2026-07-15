"""Student Course Analyzer and Prediction Platform - main Streamlit app."""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from config import PAGE_OPTIONS, XGBOOST_AVAILABLE
from data_loader import load_and_process_data, ensure_df_filled, prepare_ml_data

st.set_page_config(page_title="Student Course Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Fonts & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .main-caption { 
        color: #64748b; 
        font-size: 1rem; 
        font-weight: 500;
        margin-bottom: 2rem; 
    }
    
    /* Layout */
    .block-container { 
        padding-top: 2rem; 
        padding-bottom: 3rem;
        max-width: 1400px; 
    }
    
    /* Cards and Metrics */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    div[data-testid="stSidebar"] > div:first-child { 
        padding-top: 2rem; 
    }
    
    /* Radio Buttons (Sidebar Navigation) */
    .stRadio > div { gap: 0.5rem; }
    
    /* General elements */
    hr { 
        margin: 2rem 0; 
        border-color: #E2E8F0; 
    }
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
