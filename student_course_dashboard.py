import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.sidebar.warning("⚠️ XGBoost not installed. Install with: pip install xgboost")

try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False

st.set_page_config(page_title="Student Course Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 Student Course Analyzer and Prediction Platform")
st.caption("A data science–based dashboard to evaluate and predict student performance.")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Main Dashboard", "🤖 ML Models Comparison", 
     "⚠️ At-Risk Detection", "🎯 Course Recommendations", "📄 Report Generator"]
)

uploaded_file = st.file_uploader("📂 Upload your Excel file (Course_Analysis_Prediction_Advanced.xlsx)", type=["xlsx"])

@st.cache_data
def load_and_process_data(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    courses = xls.parse('Courses')
    students = xls.parse('Students')
    enrollments = xls.parse('Enrollments')
    feedback = xls.parse('Feedback')
    platform = xls.parse('Platform_Performance')

    df = enrollments.merge(students, on='Student_ID', how='left')\
                    .merge(courses, on='Course_ID', how='left')\
                    .merge(feedback[['Student_ID', 'Sentiment', 'Recommendation']], on='Student_ID', how='left')

    # Data preprocessing
    if 'Score' in df.columns:
        df['Score'].fillna(df['Score'].mean(), inplace=True)
    if 'Course_Rating' in df.columns:
        df['Course_Rating'].fillna(df['Course_Rating'].mean(), inplace=True)
    if 'Progress_Percent' in df.columns:
        df['Progress_Percent'].fillna(df['Progress_Percent'].mean(), inplace=True)
    if 'Credits' in df.columns:
        df['Credits'].fillna(df['Credits'].mean(), inplace=True)
    else:
        df['Credits'] = 1
    if 'Sentiment' in df.columns:
        df['Sentiment'].fillna('Neutral', inplace=True)
    else:
        df['Sentiment'] = 'Neutral'
    if 'Completion_Status' in df.columns:
        df['Completion_Status'] = df['Completion_Status'].astype('category').cat.codes
    else:
        df['Completion_Status'] = 1

    return df, courses, students, enrollments, feedback, platform

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    df, courses, students, enrollments, feedback, platform = load_and_process_data(uploaded_file)
    if 'Score' in df.columns:
        df['Score'].fillna(df['Score'].mean(), inplace=True)
    if 'Course_Rating' in df.columns:
        df['Course_Rating'].fillna(df['Course_Rating'].mean(), inplace=True)
    if 'Progress_Percent' in df.columns:
        df['Progress_Percent'].fillna(df['Progress_Percent'].mean(), inplace=True)
    if 'Credits' in df.columns:
        df['Credits'].fillna(df['Credits'].mean(), inplace=True)
    else:
        df['Credits'] = 1
    if 'Sentiment' in df.columns:
        df['Sentiment'].fillna('Neutral', inplace=True)
    else:
        df['Sentiment'] = 'Neutral'
    if 'Completion_Status' in df.columns:
        df['Completion_Status'] = df['Completion_Status'].astype('category').cat.codes
    else:
        df['Completion_Status'] = 1

    available_features = []
    for feat in ['Progress_Percent', 'Credits', 'Course_Rating']:
        if feat in df.columns:
            available_features.append(feat)

    @st.cache_data
    def prepare_ml_data(df, available_features):
        X = df[available_features].fillna(0)
        y = df['Score'].fillna(df['Score'].mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

    if len(available_features) > 0:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = prepare_ml_data(df, available_features)
    if page == "📊 Main Dashboard":
        st.subheader("�� Dataset Preview")

        if len(df) > 100:
            page_size = st.selectbox("Rows per page", [50, 100, 500, 1000], index=1, key="preview_page_size")
            total_pages = (len(df) // page_size) + 1
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="preview_page")

            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
            st.info(f"Showing rows {start_idx + 1} to {end_idx} of {len(df)} total rows")
        else:
            st.dataframe(df, use_container_width=True)

        # Key Performance Metrics
        st.markdown("### 📊 Key Metrics Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Average Score", f"{df['Score'].mean():.2f}")
        col2.metric("⭐ Average Course Rating", f"{df['Course_Rating'].mean():.2f}")
        col3.metric("🎯 Avg Progress (%)", f"{df['Progress_Percent'].mean():.2f}")

        st.markdown("---")
        st.subheader("📈 Visual Dashboard")

        # Create organized tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Platform Analytics", "😊 Sentiment Analysis", "📈 Performance Insights"])

        with tab1:
            if 'Platform' in df.columns and 'Score' in df.columns:
                st.write("### Platform Performance Comparison")

                fig1 = px.box(df, x='Platform', y='Score', color='Platform', title="Platform-wise Score Distribution")
                st.plotly_chart(fig1, use_container_width=True)

                st.write("---")
                st.write("### Platform Analytics & Rankings")
                
                if 'Platform' in df.columns:
                    platform_stats = df.groupby('Platform').agg({
                        'Score': ['mean', 'median', 'std', 'min', 'max', 'count'],
                        'Progress_Percent': 'mean' if 'Progress_Percent' in df.columns else lambda x: 0,
                        'Course_Rating': 'mean' if 'Course_Rating' in df.columns else lambda x: 0,
                    }).round(2)
                    
                    # Flatten column names
                    platform_stats.columns = ['Avg Score', 'Median Score', 'Std Dev', 'Min Score',
                                             'Max Score', 'Total Students', 'Avg Progress', 'Avg Rating']
                    platform_stats = platform_stats.reset_index()

                    platform_stats_sorted = platform_stats.sort_values('Avg Score', ascending=False)
                    
                    st.dataframe(platform_stats_sorted, use_container_width=True)
                    
                    st.write("### Platform Ranking by Performance")
                    fig_rank = px.bar(platform_stats_sorted, x='Platform', y='Avg Score', 
                                     color='Avg Score',
                                     title="Platforms Ranked by Average Score",
                                     labels={'Avg Score': 'Average Score'},
                                     color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig_rank, use_container_width=True)

                    st.write("### Detailed Platform Metrics Comparison")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.write("**Score Metrics**")
                        fig_score_compare = px.bar(platform_stats_sorted, x='Platform', 
                                                  y=['Avg Score', 'Median Score'],
                                                  title="Average vs Median Score by Platform",
                                                  barmode='group')
                        st.plotly_chart(fig_score_compare, use_container_width=True)
                    
                    with metrics_col2:
                        st.write("**Engagement & Rating**")
                        fig_engagement = px.scatter(platform_stats_sorted, x='Avg Progress', y='Avg Rating',
                                                  size='Total Students', color='Platform',
                                                  title="Progress vs Rating (bubble size = students)",
                                                  labels={'Avg Progress': 'Average Progress (%)',
                                                         'Avg Rating': 'Average Rating'})
                        st.plotly_chart(fig_engagement, use_container_width=True)
                    
                    with metrics_col3:
                        st.write("**Consistency (Lower is Better)**")
                        fig_consistency = px.bar(platform_stats_sorted, x='Platform', y='Std Dev',
                                                title="Score Consistency by Platform\n(Lower = More Consistent)",
                                                color='Std Dev',
                                                color_continuous_scale='Viridis')
                        st.plotly_chart(fig_consistency, use_container_width=True)
                    
                    st.write("---")
                    st.write("### 🏆 Best Platform Analysis")
                    
                    best_platform = platform_stats_sorted.iloc[0]
                    worst_platform = platform_stats_sorted.iloc[-1]
                    
                    best_col1, best_col2 = st.columns(2)
                    
                    with best_col1:
                        st.success(f"**🥇 Best Performing Platform: {best_platform['Platform']}**")
                        st.write(f"• Average Score: **{best_platform['Avg Score']:.2f}**")
                        st.write(f"• Median Score: **{best_platform['Median Score']:.2f}**")
                        st.write(f"• Total Students: **{int(best_platform['Total Students'])}**")
                        st.write(f"• Average Progress: **{best_platform['Avg Progress']:.1f}%**")
                        st.write(f"• Average Rating: **{best_platform['Avg Rating']:.2f}**")
                        st.write(f"• Score Consistency (Std Dev): **{best_platform['Std Dev']:.2f}**")
                    
                    with best_col2:
                        st.warning(f"**⚠️ Platform Needing Improvement: {worst_platform['Platform']}**")
                        st.write(f"• Average Score: **{worst_platform['Avg Score']:.2f}**")
                        st.write(f"• Median Score: **{worst_platform['Median Score']:.2f}**")
                        st.write(f"• Total Students: **{int(worst_platform['Total Students'])}**")
                        st.write(f"• Average Progress: **{worst_platform['Avg Progress']:.1f}%**")
                        st.write(f"• Average Rating: **{worst_platform['Avg Rating']:.2f}**")
                        st.write(f"• Score Consistency (Std Dev): **{worst_platform['Std Dev']:.2f}**")
                    
                    st.write("---")
                    st.write("### 💡 Comparative Insights")
                    
                    score_diff = best_platform['Avg Score'] - worst_platform['Avg Score']
                    progress_diff = best_platform['Avg Progress'] - worst_platform['Avg Progress']
                    rating_diff = best_platform['Avg Rating'] - worst_platform['Avg Rating']
                    
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    
                    with insight_col1:
                        st.metric("Score Difference", f"{score_diff:.2f} points", 
                                 f"{(score_diff/worst_platform['Avg Score']*100):.1f}% higher")
                    
                    with insight_col2:
                        st.metric("Progress Difference", f"{progress_diff:.1f}%",
                                 f"Higher engagement" if progress_diff > 0 else "Lower engagement")
                    
                    with insight_col3:
                        st.metric("Rating Difference", f"{rating_diff:.2f} stars",
                                 f"Better satisfaction" if rating_diff > 0 else "Lower satisfaction")
                    
                    st.write("---")
                    st.write("### �� Recommendations")
                    st.markdown(f"""
                    - **Best Practice Transfer**: Study {best_platform['Platform']}'s approach and implement similar strategies on {worst_platform['Platform']}
                    - **Focus Areas**: {worst_platform['Platform']} should focus on improving score metrics and student engagement
                    - **Success Factors**: {best_platform['Platform']}'s higher consistency (lower std dev) indicates reliable performance
                    - **Quality Assurance**: Review {worst_platform['Platform']}'s content, instructors, and infrastructure
                    """)
            else:
                st.info("ℹ️ Platform or Score column not available")

        with tab2:
            st.write("### Sentiment Analysis")
            if 'Sentiment' in df.columns:
                # Sentiment distribution pie chart
                fig2 = px.pie(df, names='Sentiment', title="Student Feedback Sentiment Distribution",
                             color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig2, use_container_width=True)

                # Sentiment breakdown by platform if available
                if 'Platform' in df.columns:
                    st.write("---")
                    st.write("### Sentiment by Platform")
                    sentiment_platform = pd.crosstab(df['Platform'], df['Sentiment'])
                    fig_sentiment_platform = px.bar(sentiment_platform,
                                                   title="Sentiment Distribution Across Platforms",
                                                   barmode='stack',
                                                   color_discrete_sequence=px.colors.qualitative.Set3)
                    st.plotly_chart(fig_sentiment_platform, use_container_width=True)
            else:
                st.info("ℹ️ Sentiment column not available")

        with tab3:
            st.write("### Performance Insights")

            # Progress vs Score correlation
            if 'Progress_Percent' in df.columns and 'Score' in df.columns:
                col1, col2 = st.columns(2)

                with col1:
                    color_col = 'Platform' if 'Platform' in df.columns else None
                    fig3 = px.scatter(df, x='Progress_Percent', y='Score', color=color_col,
                                      title="Progress vs Score Correlation",
                                      labels={'Progress_Percent': 'Progress (%)', 'Score': 'Score'})
                    st.plotly_chart(fig3, use_container_width=True)

                with col2:
                    # Score distribution histogram
                    fig_score_dist = px.histogram(df, x='Score', nbins=20,
                                                 title="Score Distribution",
                                                 labels={'Score': 'Score', 'count': 'Number of Students'},
                                                 color_discrete_sequence=['#636EFA'])
                    st.plotly_chart(fig_score_dist, use_container_width=True)

                # Progress distribution
                if 'Progress_Percent' in df.columns:
                    st.write("---")
                    st.write("### Progress Distribution")
                    fig_progress_dist = px.histogram(df, x='Progress_Percent', nbins=20,
                                                    title="Course Progress Distribution",
                                                    labels={'Progress_Percent': 'Progress (%)', 'count': 'Number of Students'},
                                                    color_discrete_sequence=['#00CC96'])
                    st.plotly_chart(fig_progress_dist, use_container_width=True)
            else:
                st.info("ℹ️ Progress_Percent or Score column not available")


        st.subheader("�� Advanced Analytics")
        
        analytics_tabs = st.tabs(["Student Performance", "Course Analysis", "Engagement Metrics", "Data Export"])
        
        with analytics_tabs[0]:
            st.write("### Top Performing Students")
            if 'Score' in df.columns:
                top_students = df.nlargest(10, 'Score')[['Student_ID', 'Score', 'Progress_Percent']].reset_index(drop=True)
                st.dataframe(top_students, use_container_width=True)

                st.write("### Score Distribution")
                fig_dist = px.histogram(df, x='Score', nbins=20, title="Student Score Distribution",
                                       labels={'Score': 'Score', 'count': 'Number of Students'})
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with analytics_tabs[1]:
            st.write("### Course Performance Summary")
            if 'Course_ID' in df.columns and 'Score' in df.columns:
                course_stats = df.groupby('Course_ID').agg({
                    'Score': ['mean', 'std', 'count'],
                    'Progress_Percent': 'mean'
                }).round(2)
                course_stats.columns = ['Avg Score', 'Std Dev', 'Students Enrolled', 'Avg Progress']
                st.dataframe(course_stats, use_container_width=True)

                if 'Course_Rating' in df.columns:
                    st.write("### Average Rating by Course")
                    course_rating = df.groupby('Course_ID')['Course_Rating'].mean().sort_values(ascending=False)
                    fig_course = px.bar(x=course_rating.index, y=course_rating.values,
                                       labels={'x': 'Course ID', 'y': 'Average Rating'},
                                       title="Course Ratings Comparison")
                    st.plotly_chart(fig_course, use_container_width=True)
        
        with analytics_tabs[2]:
            st.write("### Engagement & Sentiment Analysis")

            col_eng1, col_eng2 = st.columns(2)

            with col_eng1:
                if 'Progress_Percent' in df.columns:
                    avg_progress = df['Progress_Percent'].mean()
                    st.metric("�� Average Progress", f"{avg_progress:.1f}%")

            with col_eng2:
                if 'Score' in df.columns:
                    st.metric("�� Median Score", f"{df['Score'].median():.2f}")

            if 'Sentiment' in df.columns:
                st.write("### Sentiment Breakdown")
                sentiment_counts = df['Sentiment'].value_counts()
                fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                                      title="Student Feedback Sentiment Distribution")
                st.plotly_chart(fig_sentiment, use_container_width=True)

            if 'Progress_Percent' in df.columns and 'Score' in df.columns:
                st.write("### Progress vs Performance Level")
                df['Performance_Level'] = pd.cut(df['Score'],
                                               bins=[0, 50, 70, 85, 100],
                                               labels=['Below Average', 'Average', 'Good', 'Excellent'])
                perf_progress = df.groupby('Performance_Level')['Progress_Percent'].mean()
                fig_perf = px.bar(x=perf_progress.index, y=perf_progress.values,
                                labels={'x': 'Performance Level', 'y': 'Average Progress (%)'},
                                title="Average Progress by Performance Level",
                                color=perf_progress.index)
                st.plotly_chart(fig_perf, use_container_width=True)
        
        with analytics_tabs[3]:
            st.write("### Export Data")

            summary_stats = {
                'Metric': ['Total Students', 'Total Courses', 'Average Score', 'Average Progress',
                          'Average Rating', 'Total Enrollments'],
                'Value': [
                    len(df['Student_ID'].unique()) if 'Student_ID' in df.columns else 0,
                    len(df['Course_ID'].unique()) if 'Course_ID' in df.columns else 0,
                    df['Score'].mean() if 'Score' in df.columns else 0,
                    df['Progress_Percent'].mean() if 'Progress_Percent' in df.columns else 0,
                    df['Course_Rating'].mean() if 'Course_Rating' in df.columns else 0,
                    len(df)
                ]
            }
            summary_df = pd.DataFrame(summary_stats)
            st.write("### Summary Statistics")
            st.dataframe(summary_df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full Dataset (CSV)",
                              data=csv,
                              file_name="student_course_analysis.csv",
                              mime="text/csv")

            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Summary Statistics (CSV)",
                              data=summary_csv,
                              file_name="summary_statistics.csv",
                              mime="text/csv")

        st.subheader("🔍 Student & Course Search")
        
        search_col1, search_col2 = st.columns(2)
        
        with search_col1:
            st.write("### Search by Student")
            if 'Student_ID' in df.columns:
                student_search = st.selectbox("Select a Student:", 
                                             df['Student_ID'].unique(),
                                             key="student_search")
                student_data = df[df['Student_ID'] == student_search]
                
                if not student_data.empty:
                    st.write(f"**Student ID:** {student_search}")
                    if 'Score' in student_data.columns:
                        st.write(f"**Score:** {student_data['Score'].values[0]:.2f}")
                    if 'Progress_Percent' in student_data.columns:
                        st.write(f"**Progress:** {student_data['Progress_Percent'].values[0]:.1f}%")
                    if 'Course_ID' in student_data.columns:
                        st.write(f"**Courses Enrolled:** {student_data['Course_ID'].nunique()}")
        
        with search_col2:
            st.write("### Search by Course")
            if 'Course_ID' in df.columns:

                course_name_map = {}
                if 'Course_Name' in df.columns:
                    course_data_temp = df[['Course_ID', 'Course_Name']].drop_duplicates()
                    for idx, row in course_data_temp.iterrows():
                        course_name_map[row['Course_Name']] = row['Course_ID']
                else:
                    for course_id in df['Course_ID'].unique():
                        course_name_map[f"Course {course_id}"] = course_id
                
                if course_name_map:
                    course_name_search = st.selectbox("Select a Course:",
                                                      list(course_name_map.keys()),
                                                      key="course_search")
                    course_id = course_name_map[course_name_search]
                    course_data = df[df['Course_ID'] == course_id]
                    
                    if not course_data.empty:
                        st.write(f"**Course:** {course_name_search}")
                        st.write(f"**Course ID:** {course_id}")
                        if 'Score' in course_data.columns:
                            st.write(f"**Average Score:** {course_data['Score'].mean():.2f}")
                        st.write(f"**Total Enrollments:** {len(course_data)}")
                        if 'Course_Rating' in course_data.columns:
                            st.write(f"**Course Rating:** {course_data['Course_Rating'].mean():.2f}")

        st.subheader("💡 Key Insights & Recommendations")
        
        insights = []
        
        if 'Score' in df.columns:
            avg_score = df['Score'].mean()
            if avg_score >= 80:
                insights.append("✅ **Excellent Performance:** Students are performing very well overall.")
            elif avg_score >= 70:
                insights.append("�� **Good Performance:** Students are meeting expected standards.")
            else:
                insights.append("⚠️ **Improvement Needed:** Consider additional support or intervention programs.")
        
        if 'Progress_Percent' in df.columns:
            avg_progress = df['Progress_Percent'].mean()
            if avg_progress >= 80:
                insights.append("✅ **High Engagement:** Students are actively progressing through courses.")
            else:
                insights.append("📢 **Low Engagement:** Encourage students to increase course participation.")
        
        if 'Sentiment' in df.columns:
            positive_sentiment = (df['Sentiment'] == 'Positive').sum() / len(df) * 100
            if positive_sentiment >= 70:
                insights.append("😊 **Positive Feedback:** Students are satisfied with their learning experience.")
            elif positive_sentiment >= 50:
                insights.append("😐 **Mixed Feedback:** Some students are satisfied while others need support.")
            else:
                insights.append("😞 **Negative Feedback:** Address student concerns and improve course quality.")
        
        if insights:
            for insight in insights:
                st.markdown(insight)
        
        st.markdown("""
        ---
        ### 📌 Recommendations:
        - Use the prediction model to identify at-risk students and provide proactive support
        - Analyze course ratings to identify high-performing and underperforming courses
        - Monitor student sentiment to improve engagement and satisfaction
        - Regularly track progress metrics to ensure students stay on track
        """)

    elif page == "🤖 ML Models Comparison":
        st.header("🤖 Machine Learning Models Comparison")
        st.write("Compare multiple ML models to find the best predictor for student scores.")
        
        if len(available_features) == 0:
            st.warning("⚠️ Not enough features available for prediction models")
        else:
            models = {}
            results = {}

            st.subheader("🌲 Random Forest Regressor")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            models['Random Forest'] = rf_model
            results['Random Forest'] = {
                'R²': r2_score(y_test, rf_pred),
                'MSE': mean_squared_error(y_test, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
            }

            st.subheader("📈 Gradient Boosting Regressor")
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            gb_pred = gb_model.predict(X_test)
            models['Gradient Boosting'] = gb_model
            results['Gradient Boosting'] = {
                'R²': r2_score(y_test, gb_pred),
                'MSE': mean_squared_error(y_test, gb_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred))
            }

            st.subheader("📊 Linear Regression")
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            models['Linear Regression'] = lr_model
            results['Linear Regression'] = {
                'R²': r2_score(y_test, lr_pred),
                'MSE': mean_squared_error(y_test, lr_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
            }

            if XGBOOST_AVAILABLE:
                st.subheader("🚀 XGBoost Regressor")
                xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)
                models['XGBoost'] = xgb_model
                results['XGBoost'] = {
                    'R²': r2_score(y_test, xgb_pred),
                    'MSE': mean_squared_error(y_test, xgb_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred))
                }

            st.subheader("📊 Model Comparison")
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.sort_values('R²', ascending=False)
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['R²'],
                                name='R² Score', marker_color='lightblue'))
            fig.update_layout(title="Model Comparison - R² Scores",
                            xaxis_title="Model", yaxis_title="R² Score")
            st.plotly_chart(fig, use_container_width=True)

            best_model_name = comparison_df.index[0]
            st.success(f"🏆 Best Model: **{best_model_name}** with R² = {comparison_df.loc[best_model_name, 'R²']:.4f}")

            st.subheader("🔍 Advanced Model Analysis")

            analysis_tabs = st.tabs(["Cross-Validation Scores", "Feature Importance Comparison", "Model Robustness"])

            with analysis_tabs[0]:
                st.write("### Cross-Validation Performance")
                from sklearn.model_selection import cross_val_score

                cv_results = {}
                for name, model in models.items():
                    if name == 'Linear Regression':
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    cv_results[name] = {
                        'Mean CV R²': cv_scores.mean(),
                        'Std CV R²': cv_scores.std(),
                        'Min CV R²': cv_scores.min(),
                        'Max CV R²': cv_scores.max()
                    }

                cv_df = pd.DataFrame(cv_results).T.round(4)
                st.dataframe(cv_df.style.highlight_max(axis=0, color='lightgreen'))

                fig_cv = go.Figure()
                for name in cv_results.keys():
                    fig_cv.add_trace(go.Box(
                        y=[cv_results[name]['Min CV R²'], cv_results[name]['Mean CV R²']-cv_results[name]['Std CV R²'],
                           cv_results[name]['Mean CV R²'], cv_results[name]['Mean CV R²']+cv_results[name]['Std CV R²'],
                           cv_results[name]['Max CV R²']],
                        name=name,
                        boxpoints=False
                    ))
                fig_cv.update_layout(title="Cross-Validation R² Score Distribution",
                                   yaxis_title="R² Score")
                st.plotly_chart(fig_cv, use_container_width=True)

            with analysis_tabs[1]:
                st.write("### Feature Importance Comparison")

                importance_data = {}
                for name, model in models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance_data[name] = model.feature_importances_
                    elif name == 'Linear Regression' and hasattr(model, 'coef_'):
                        importance_data[name] = np.abs(model.coef_[0])

                if importance_data:
                    importance_df = pd.DataFrame(importance_data, index=available_features)
                    importance_df = importance_df.div(importance_df.sum(axis=0), axis=1)  # Normalize

                    fig_importance = px.imshow(importance_df.T,
                                             text_auto='.3f',
                                             aspect="auto",
                                             title="Feature Importance Across Models",
                                             labels=dict(x="Features", y="Models", color="Importance"))
                    st.plotly_chart(fig_importance, use_container_width=True)

                    for feature in available_features:
                        fig_feat = go.Figure()
                        for model_name in importance_data.keys():
                            fig_feat.add_trace(go.Bar(
                                name=model_name,
                                x=[feature],
                                y=[importance_df.loc[feature, model_name]]
                            ))
                        fig_feat.update_layout(title=f"Feature Importance: {feature}",
                                             barmode='group')
                        st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("Feature importance not available for all models")

            with analysis_tabs[2]:
                st.write("### Model Robustness Analysis")

                robustness_metrics = {}
                for name, model in models.items():
                    if name == 'Linear Regression':
                        y_pred_robust = model.predict(X_test_scaled)
                    else:
                        y_pred_robust = model.predict(X_test)

                    mae = mean_squared_error(y_test, y_pred_robust)
                    mape = np.mean(np.abs((y_test - y_pred_robust) / y_test)) * 100
                    max_error = np.max(np.abs(y_test - y_pred_robust))

                    robustness_metrics[name] = {
                        'MAE': mae,
                        'MAPE (%)': mape,
                        'Max Error': max_error,
                        'Prediction Std': np.std(y_pred_robust)
                    }

                robustness_df = pd.DataFrame(robustness_metrics).T.round(4)
                st.dataframe(robustness_df.style.highlight_min(axis=0, color='lightgreen'))

                fig_robust = go.Figure()
                for metric in ['MAE', 'MAPE (%)', 'Max Error']:
                    fig_robust.add_trace(go.Bar(
                        name=metric,
                        x=list(robustness_metrics.keys()),
                        y=[robustness_metrics[model][metric] for model in robustness_metrics.keys()]
                    ))
                fig_robust.update_layout(title="Model Robustness Metrics",
                                       barmode='group',
                                       yaxis_title="Metric Value")
                st.plotly_chart(fig_robust, use_container_width=True)
            

    elif page == "⚠️ At-Risk Detection":
        st.header("⚠️ At-Risk Student Detection System")
        st.write("Identify students who may need additional support.")

        if 'Student_ID' in df.columns:
            @st.cache_data
            def calculate_risk_scores(df):
                with st.spinner("Calculating risk scores..."):
                    progress_bar = st.progress(0)

                    student_agg = df.groupby('Student_ID').agg({
                        'Progress_Percent': 'mean' if 'Progress_Percent' in df.columns else lambda x: 0,
                        'Score': 'mean' if 'Score' in df.columns else lambda x: 0,
                        'Sentiment': lambda x: 'Negative' if (x == 'Negative').any() else x.mode()[0] if len(x.mode()) > 0 else 'Neutral',
                        'Student_Name': 'first' if 'Student_Name' in df.columns else lambda x: f"Student {x.name}",
                        'Course_ID': 'count'
                    }).reset_index()

                    progress_bar.progress(30)
                    student_agg.columns = ['Student_ID', 'Avg_Progress', 'Avg_Score', 'Sentiment', 'Student_Name', 'Total_Courses']

                    student_agg['Risk_Score'] = 0.0
                    student_agg['Risk_Reasons'] = ''

                    progress_risk_scores = []
                    score_risk_scores = []
                    sentiment_risk_scores = []

                    if 'Avg_Progress' in student_agg.columns:
                        progress_risk = (100 - student_agg['Avg_Progress']) / 100 * 0.4
                        student_agg['Risk_Score'] += progress_risk
                        for i, progress in enumerate(student_agg['Avg_Progress']):
                            if progress < 50:
                                progress_risk_scores.append(f"Very low progress ({progress:.1f}%)")
                            elif progress < 70:
                                progress_risk_scores.append(f"Low progress ({progress:.1f}%)")
                            else:
                                progress_risk_scores.append("")

                    progress_bar.progress(50)
                    if 'Avg_Score' in student_agg.columns:
                        score_risk = (100 - student_agg['Avg_Score']) / 100 * 0.4
                        student_agg['Risk_Score'] += score_risk
                        for i, score in enumerate(student_agg['Avg_Score']):
                            if score < 50:
                                score_risk_scores.append(f"Very low scores ({score:.1f})")
                            elif score < 65:
                                score_risk_scores.append(f"Low scores ({score:.1f})")
                            else:
                                score_risk_scores.append("")

                    if 'Sentiment' in student_agg.columns:
                        sentiment_risk = (student_agg['Sentiment'] == 'Negative').astype(int) * 0.2
                        student_agg['Risk_Score'] += sentiment_risk
                        for i, sentiment in enumerate(student_agg['Sentiment']):
                            if sentiment == 'Negative':
                                sentiment_risk_scores.append("Negative feedback/sentiment")
                            else:
                                sentiment_risk_scores.append("")

                    progress_bar.progress(80)
                    student_agg['Risk_Score'] = student_agg['Risk_Score'] * 100

                    student_agg['Risk_Level'] = pd.cut(student_agg['Risk_Score'],
                                                     bins=[0, 30, 50, 70, 100],
                                                     labels=['Low', 'Medium', 'High', 'Critical'])

                    for i in range(len(student_agg)):
                        reasons = []
                        if progress_risk_scores[i]:
                            reasons.append(progress_risk_scores[i])
                        if score_risk_scores[i]:
                            reasons.append(score_risk_scores[i])
                        if sentiment_risk_scores[i]:
                            reasons.append(sentiment_risk_scores[i])
                        if not reasons:
                            reasons.append("Multiple minor risk factors")
                        student_agg.loc[i, 'Risk_Reasons'] = "; ".join(reasons)

                    progress_bar.progress(100)
                    progress_bar.empty()
                    return student_agg

            student_risk_data = calculate_risk_scores(df)

            st.subheader("🎚️ Set Risk Threshold")
            risk_threshold = st.slider("Risk Threshold", 0, 100, 50)

            at_risk_students = student_risk_data[student_risk_data['Risk_Score'] >= risk_threshold].sort_values('Risk_Score', ascending=False)

            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("At-Risk Students", len(at_risk_students), 
                         f"{len(at_risk_students)/len(student_risk_data)*100:.1f}% of total")
            with col_metric2:
                st.metric("Total Students", len(student_risk_data))
            with col_metric3:
                safe_students = len(student_risk_data) - len(at_risk_students)
                st.metric("Safe Students", safe_students,
                         f"{safe_students/len(student_risk_data)*100:.1f}% of total")
            
            fig = px.histogram(student_risk_data, x='Risk_Score', nbins=20,
                              title="Risk Score Distribution",
                              labels={'Risk_Score': 'Risk Score (0-100)', 'count': 'Number of Students'})
            fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                         annotation_text=f"Threshold: {risk_threshold}", annotation_position="top")
            st.plotly_chart(fig, use_container_width=True)

            if 'Risk_Level' in student_risk_data.columns:
                risk_level_counts = student_risk_data['Risk_Level'].value_counts()
                fig_pie = px.pie(values=risk_level_counts.values, names=risk_level_counts.index,
                               title="Students by Risk Level",
                               color_discrete_map={'Low': '#48bb78', 'Medium': '#ed8936',
                                                 'High': '#f56565', 'Critical': '#c53030'})
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("🚨 High-Risk Students")
            if len(at_risk_students) > 0:
                display_cols = ['Student_Name', 'Student_ID', 'Risk_Score', 'Risk_Level', 'Risk_Reasons', 'Avg_Score', 'Avg_Progress', 'Total_Courses']
                if 'Sentiment' in at_risk_students.columns:
                    display_cols.append('Sentiment')
                display_cols = [col for col in display_cols if col in at_risk_students.columns]

                if len(at_risk_students) > 50:
                    show_count = st.slider("Show top N students", 10, min(100, len(at_risk_students)), 20, key="risk_show_count")
                    st.dataframe(at_risk_students[display_cols].head(show_count), use_container_width=True)
                else:
                    st.dataframe(at_risk_students[display_cols], use_container_width=True)

                st.subheader("💡 Detailed Intervention Recommendations")

                tabs = st.tabs(["📋 Risk Breakdown", "🎯 Personalized Recommendations", "📊 Risk Statistics"])

                with tabs[0]:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("### Risk Level Distribution")
                        risk_level_counts = at_risk_students['Risk_Level'].value_counts()
                        fig_risk_pie = px.pie(values=risk_level_counts.values, names=risk_level_counts.index,
                                           title="At-Risk Students by Risk Level",
                                           color_discrete_map={'Low': '#48bb78', 'Medium': '#ed8936',
                                                             'High': '#f56565', 'Critical': '#c53030'})
                        st.plotly_chart(fig_risk_pie, use_container_width=True)

                    with col2:
                        st.write("### Common Risk Factors")
                        all_reasons = []
                        for reasons in at_risk_students['Risk_Reasons'].str.split('; '):
                            if isinstance(reasons, list):
                                all_reasons.extend(reasons)

                        reason_counts = pd.Series(all_reasons).value_counts()
                        fig_reasons = px.bar(x=reason_counts.index, y=reason_counts.values,
                                           title="Most Common Risk Factors",
                                           labels={'x': 'Risk Factor', 'y': 'Count'})
                        fig_reasons.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_reasons, use_container_width=True)

                with tabs[1]:
                    st.write("### Individual Student Recommendations")
                    selected_risk_student = st.selectbox(
                        "Select a student for detailed recommendations:",
                        at_risk_students.apply(lambda x: f"{x['Student_Name']} (ID: {x['Student_ID']}) - Risk: {x['Risk_Score']:.1f}", axis=1),
                        key="risk_student_select"
                    )

                    if selected_risk_student:
                        student_id = selected_risk_student.split('(ID: ')[1].split(')')[0]
                        student_data = at_risk_students[at_risk_students['Student_ID'] == student_id].iloc[0]

                        col_info, col_recs = st.columns([1, 2])

                        with col_info:
                            st.write("### Student Information")
                            st.write(f"**Name:** {student_data['Student_Name']}")
                            st.write(f"**ID:** {student_data['Student_ID']}")
                            st.write(f"**Risk Score:** {student_data['Risk_Score']:.1f}")
                            st.write(f"**Risk Level:** {student_data['Risk_Level']}")
                            st.write(f"**Average Score:** {student_data['Avg_Score']:.1f}")
                            st.write(f"**Average Progress:** {student_data['Avg_Progress']:.1f}%")
                            st.write(f"**Total Courses:** {student_data['Total_Courses']}")

                        with col_recs:
                            st.write("### Personalized Recommendations")
                            recommendations = []

                            if "Very low progress" in student_data['Risk_Reasons']:
                                recommendations.extend([
                                    "🚨 **Immediate Action Required** - Schedule daily check-ins with instructor",
                                    "📅 **Study Schedule** - Create a structured daily study plan with specific time blocks",
                                    "👥 **Study Buddy** - Pair with a high-performing student for peer support",
                                    "🎯 **Progress Tracking** - Set weekly progress milestones and track daily"
                                ])
                            elif "Low progress" in student_data['Risk_Reasons']:
                                recommendations.extend([
                                    "⚠️ **Progress Monitoring** - Weekly check-ins with academic advisor",
                                    "📚 **Resource Access** - Ensure access to all required course materials",
                                    "🎯 **Goal Setting** - Break down course objectives into manageable weekly goals"
                                ])

                            if "Very low scores" in student_data['Risk_Reasons']:
                                recommendations.extend([
                                    "🎓 **Academic Tutoring** - Immediate enrollment in tutoring program",
                                    "📖 **Supplemental Resources** - Access to additional textbooks and practice materials",
                                    "🧠 **Learning Assessment** - Complete learning style assessment for personalized study methods",
                                    "📝 **Practice Exams** - Regular practice testing with feedback"
                                ])
                            elif "Low scores" in student_data['Risk_Reasons']:
                                recommendations.extend([
                                    "📚 **Study Groups** - Join or form study groups for collaborative learning",
                                    "🎯 **Targeted Review** - Focus on weak subject areas with additional practice",
                                    "📊 **Grade Tracking** - Monitor assignment grades and identify improvement areas"
                                ])

                            if "Negative feedback" in student_data['Risk_Reasons']:
                                recommendations.extend([
                                    "💬 **Counseling Support** - Schedule meeting with academic counselor",
                                    "😊 **Feedback Discussion** - One-on-one discussion about course experience",
                                    "🔄 **Course Adjustment** - Consider course load adjustment or alternative courses",
                                    "🌟 **Motivation Support** - Connect with student success programs"
                                ])

                            if student_data['Risk_Level'] == 'Critical':
                                recommendations.insert(0, "🚨 **CRITICAL PRIORITY** - Immediate intervention by academic affairs committee")
                                recommendations.insert(1, "📞 **Parent/Guardian Contact** - Notify family for additional support")
                                recommendations.insert(2, "🏥 **Academic Probation** - Consider academic probation status")

                            for rec in recommendations:
                                st.markdown(f"- {rec}")

                            st.write("### 📈 Improvement Timeline")
                            timeline_suggestions = [
                                "**Week 1-2:** Establish study routine and complete initial assessment",
                                "**Week 3-4:** Begin targeted interventions and monitor progress",
                                "**Week 5-8:** Implement advanced support strategies and regular check-ins",
                                "**Week 9-12:** Comprehensive review and long-term planning"
                            ]
                            for timeline in timeline_suggestions:
                                st.markdown(f"- {timeline}")

                with tabs[2]:
                    st.write("### Risk Analysis Statistics")
                    col_stats1, col_stats2 = st.columns(2)

                    with col_stats1:
                        st.metric("Average Risk Score", f"{at_risk_students['Risk_Score'].mean():.1f}")
                        st.metric("Highest Risk Score", f"{at_risk_students['Risk_Score'].max():.1f}")
                        st.metric("Median Risk Score", f"{at_risk_students['Risk_Score'].median():.1f}")

                    with col_stats2:
                        st.metric("Most Common Risk Level", at_risk_students['Risk_Level'].mode().iloc[0])
                        st.metric("Students Needing Immediate Help", len(at_risk_students[at_risk_students['Risk_Level'] == 'Critical']))
                        st.metric("Students with Progress Issues", len(at_risk_students[at_risk_students['Avg_Progress'] < 50]))

            else:
                st.success("✅ No students above risk threshold!")
                st.balloons()
        else:
            st.error("Student_ID column not found in dataset")

    elif page == "🎯 Course Recommendations":
        st.header("🎯 Personalized Course Recommendation Engine")
        st.write("Get course recommendations based on student performance and preferences.")

        recommendation_tabs = st.tabs(["🎓 General Recommendations", "🚨 At-Risk Student Support", "📊 Course Analytics"])

        if 'Student_ID' in df.columns and 'Course_ID' in df.columns:
            with recommendation_tabs[0]:
                student_id = st.selectbox("Select Student", df['Student_ID'].unique(), key="general_rec_student")

                student_data = df[df['Student_ID'] == student_id]
                student_avg_score = student_data['Score'].mean()
                student_avg_rating = student_data['Course_Rating'].mean() if 'Course_Rating' in student_data.columns else 0

                student_courses = student_data['Course_ID'].unique()

                if 'Course_Name' in df.columns:
                    course_mapping = df[['Course_ID', 'Course_Name']].drop_duplicates().set_index('Course_ID')['Course_Name'].to_dict()
                    all_courses_with_names = df[['Course_ID', 'Course_Name']].drop_duplicates()
                    available_courses = all_courses_with_names[~all_courses_with_names['Course_ID'].isin(student_courses)]
                else:
                    all_course_ids = df['Course_ID'].unique()
                    available_courses = pd.DataFrame({
                        'Course_ID': [c for c in all_course_ids if c not in student_courses],
                        'Course_Name': [f"Course {c}" for c in all_course_ids if c not in student_courses]
                    })
                    course_mapping = {cid: f"Course {cid}" for cid in all_course_ids}

                if len(available_courses) > 0:
                    recommendations = []
                    for idx, row in available_courses.iterrows():
                        course_id = row['Course_ID']
                        course_name = row['Course_Name'] if 'Course_Name' in row else course_mapping.get(course_id, f"Course {course_id}")

                        course_data = df[df['Course_ID'] == course_id]
                        course_avg_score = course_data['Score'].mean()
                        course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0

                        score_similarity = 1 - abs(student_avg_score - course_avg_score) / 100
                        rating_similarity = 1 - abs(student_avg_rating - course_avg_rating) / 5 if course_avg_rating > 0 else 0.5
                        recommendation_score = (score_similarity * 0.6 + rating_similarity * 0.4) * 100

                        recommendations.append({
                            'Course_Name': course_name,
                            'Course_ID': course_id,
                            'Recommendation_Score': recommendation_score,
                            'Avg_Course_Score': course_avg_score,
                            'Avg_Course_Rating': course_avg_rating,
                            'Enrollments': len(course_data)
                        })

                    rec_df = pd.DataFrame(recommendations).sort_values('Recommendation_Score', ascending=False)

                    st.subheader(f"📚 Recommended Courses for Student {student_id}")
                    display_cols = ['Course_Name', 'Recommendation_Score', 'Avg_Course_Score', 'Avg_Course_Rating', 'Enrollments']
                    st.dataframe(rec_df[display_cols].head(10), use_container_width=True)

                    fig = px.bar(rec_df.head(10), x='Course_Name', y='Recommendation_Score',
                               title="Top 10 Course Recommendations",
                               labels={'Recommendation_Score': 'Recommendation Score', 'Course_Name': 'Course Name'})
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Student has enrolled in all available courses!")

            with recommendation_tabs[1]:
                st.subheader("🚨 Support Courses for At-Risk Students")
                st.write("Specialized course recommendations for students needing additional support")

                if 'Student_ID' in df.columns:
                    @st.cache_data
                    def get_at_risk_students(df):
                        student_risk_data = df.groupby('Student_ID').agg({
                            'Progress_Percent': 'mean' if 'Progress_Percent' in df.columns else lambda x: 0,
                            'Score': 'mean' if 'Score' in df.columns else lambda x: 0,
                            'Sentiment': lambda x: 'Negative' if (x == 'Negative').any() else x.mode()[0] if len(x.mode()) > 0 else 'Neutral',
                            'Student_Name': 'first' if 'Student_Name' in df.columns else lambda x: f"Student {x.name}"
                        }).reset_index()

                        student_risk_data.columns = ['Student_ID', 'Avg_Progress', 'Avg_Score', 'Sentiment', 'Student_Name']

                        student_risk_data['Risk_Score'] = 0.0
                        if 'Avg_Progress' in student_risk_data.columns:
                            progress_risk = (100 - student_risk_data['Avg_Progress']) / 100 * 0.4
                            student_risk_data['Risk_Score'] += progress_risk
                        if 'Avg_Score' in student_risk_data.columns:
                            score_risk = (100 - student_risk_data['Avg_Score']) / 100 * 0.4
                            student_risk_data['Risk_Score'] += score_risk
                        if 'Sentiment' in student_risk_data.columns:
                            sentiment_risk = (student_risk_data['Sentiment'] == 'Negative').astype(int) * 0.2
                            student_risk_data['Risk_Score'] += sentiment_risk
                        student_risk_data['Risk_Score'] = student_risk_data['Risk_Score'] * 100

                        return student_risk_data[student_risk_data['Risk_Score'] >= 50].sort_values('Risk_Score', ascending=False)

                    at_risk_students_list = get_at_risk_students(df)

                    if len(at_risk_students_list) > 0:
                        risk_student_options = at_risk_students_list.apply(
                            lambda x: f"{x['Student_Name']} (ID: {x['Student_ID']}) - Risk: {x['Risk_Score']:.1f}", axis=1
                        )

                        selected_risk_student = st.selectbox(
                            "Select at-risk student for support course recommendations:",
                            risk_student_options,
                            key="risk_support_student"
                        )

                        if selected_risk_student:
                            student_id = selected_risk_student.split('(ID: ')[1].split(')')[0]
                            student_data = df[df['Student_ID'] == student_id]
                            student_info = at_risk_students_list[at_risk_students_list['Student_ID'] == student_id].iloc[0]

                            col_profile, col_recommendations = st.columns([1, 2])

                            with col_profile:
                                st.write("### Student Risk Profile")
                                st.write(f"**Name:** {student_info['Student_Name']}")
                                st.write(f"**Risk Score:** {student_info['Risk_Score']:.1f}")
                                st.write(f"**Average Score:** {student_info['Avg_Score']:.1f}")
                                st.write(f"**Average Progress:** {student_info['Avg_Progress']:.1f}%")

                                risk_factors = []
                                if student_info['Avg_Score'] < 60:
                                    risk_factors.append("Low Academic Performance")
                                if student_info['Avg_Progress'] < 50:
                                    risk_factors.append("Poor Course Progress")
                                if student_info['Sentiment'] == 'Negative':
                                    risk_factors.append("Negative Feedback")

                                st.write("**Primary Risk Factors:**")
                                for factor in risk_factors:
                                    st.write(f"- {factor}")

                            with col_recommendations:
                                st.write("### 🎓 Recommended Support Courses")

                                @st.cache_data
                                def get_support_course_recommendations(df, student_id):
                                    student_data = df[df['Student_ID'] == student_id]
                                    student_avg_score = student_data['Score'].mean()
                                    student_courses = student_data['Course_ID'].unique()

                                    all_courses = df['Course_ID'].unique()
                                    available_courses = [c for c in all_courses if c not in student_courses]

                                    support_recommendations = []

                                    for course_id in available_courses:
                                        course_data = df[df['Course_ID'] == course_id]
                                        course_avg_score = course_data['Score'].mean()
                                        course_avg_progress = course_data['Progress_Percent'].mean() if 'Progress_Percent' in course_data.columns else 0
                                        course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0

                                        course_name = f"Course {course_id}"
                                        if 'Course_Name' in df.columns:
                                            name_df = df[df['Course_ID'] == course_id][['Course_Name']].drop_duplicates()
                                            if not name_df.empty:
                                                course_name = name_df['Course_Name'].iloc[0]

                                        support_score = 0
                                        reasons = []

                                        if course_avg_score > student_avg_score + 10:
                                            support_score += 30
                                            reasons.append("Higher success rate")
                                        elif course_avg_score > student_avg_score:
                                            support_score += 20
                                            reasons.append("Moderate success rate")

                                        if course_avg_progress > 75:
                                            support_score += 25
                                            reasons.append("High completion rate")
                                        elif course_avg_progress > 60:
                                            support_score += 15
                                            reasons.append("Good completion rate")

                                        if course_avg_rating >= 4.0:
                                            support_score += 20
                                            reasons.append("Highly rated")
                                        elif course_avg_rating >= 3.5:
                                            support_score += 10
                                            reasons.append("Well rated")

                                        support_score += min(len(course_data) * 0.5, 15)  # Popularity bonus
                                        reasons.append(f"{len(course_data)} students enrolled")

                                        if course_avg_score < 70:
                                            support_score += 10
                                            reasons.append("Suitable difficulty level")

                                        support_recommendations.append({
                                            'Course_Name': course_name,
                                            'Course_ID': course_id,
                                            'Support_Score': support_score,
                                            'Reasons': '; '.join(reasons),
                                            'Avg_Score': course_avg_score,
                                            'Avg_Progress': course_avg_progress,
                                            'Rating': course_avg_rating,
                                            'Enrollments': len(course_data)
                                        })

                                    return pd.DataFrame(support_recommendations).sort_values('Support_Score', ascending=False)

                                support_courses = get_support_course_recommendations(df, student_id)

                                if len(support_courses) > 0:
                                    display_cols = ['Course_Name', 'Support_Score', 'Reasons', 'Avg_Score', 'Avg_Progress', 'Rating']
                                    st.dataframe(support_courses[display_cols].head(8), use_container_width=True)

                                    st.write("### 📋 Implementation Plan")
                                    top_course = support_courses.iloc[0]

                                    plan_steps = [
                                        f"1. **Enroll in {top_course['Course_Name']}** - Primary support course with highest compatibility",
                                        f"2. **Schedule weekly tutoring** for {top_course['Course_Name']} (2 hours/week)",
                                        "3. **Join study group** within the first week of enrollment",
                                        "4. **Set up progress check-ins** with instructor (bi-weekly)",
                                        "5. **Access supplemental materials** and practice resources",
                                        "6. **Monitor progress weekly** and adjust study strategies as needed"
                                    ]

                                    for step in plan_steps:
                                        st.markdown(f"- {step}")

                                    st.write("### 🎯 Success Metrics")
                                    metrics = [
                                        "Complete at least 75% of weekly assignments",
                                        "Maintain average score above 70%",
                                        "Attend all tutoring sessions",
                                        "Submit progress reports weekly",
                                        "Complete course with passing grade"
                                    ]

                                    for metric in metrics:
                                        st.markdown(f"- ✅ {metric}")
                                else:
                                    st.info("No additional courses available for this student")
                    else:
                        st.info("No at-risk students identified in the current dataset")

            with recommendation_tabs[2]:
                st.subheader("📊 Course Performance Analytics")

                if 'Course_ID' in df.columns:
                    course_stats = df.groupby('Course_ID').agg({
                        'Score': ['mean', 'std', 'count'],
                        'Progress_Percent': 'mean',
                        'Course_Rating': 'mean'
                    }).round(2)

                    course_stats.columns = ['Avg_Score', 'Score_Std', 'Enrollments', 'Avg_Progress', 'Avg_Rating']
                    course_stats = course_stats.reset_index()

                    if 'Course_Name' in df.columns:
                        course_names = df[['Course_ID', 'Course_Name']].drop_duplicates().set_index('Course_ID')['Course_Name']
                        course_stats['Course_Name'] = course_stats['Course_ID'].map(course_names)
                    else:
                        course_stats['Course_Name'] = course_stats['Course_ID'].apply(lambda x: f"Course {x}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("### Top Performing Courses")
                        top_courses = course_stats.sort_values('Avg_Score', ascending=False).head(10)
                        fig_top = px.bar(top_courses, x='Course_Name', y='Avg_Score',
                                       title="Highest Scoring Courses",
                                       labels={'Avg_Score': 'Average Score', 'Course_Name': 'Course'})
                        fig_top.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_top, use_container_width=True)

                    with col2:
                        st.write("### Most Popular Courses")
                        popular_courses = course_stats.sort_values('Enrollments', ascending=False).head(10)
                        fig_popular = px.bar(popular_courses, x='Course_Name', y='Enrollments',
                                           title="Most Enrolled Courses",
                                           labels={'Enrollments': 'Number of Students', 'Course_Name': 'Course'})
                        fig_popular.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_popular, use_container_width=True)

                    st.write("### Course Difficulty Analysis")
                    course_stats['Difficulty_Level'] = pd.cut(course_stats['Avg_Score'],
                                                            bins=[0, 60, 75, 85, 100],
                                                            labels=['Challenging', 'Moderate', 'Easy', 'Very Easy'])

                    difficulty_counts = course_stats['Difficulty_Level'].value_counts()
                    fig_difficulty = px.pie(values=difficulty_counts.values, names=difficulty_counts.index,
                                          title="Course Difficulty Distribution")
                    st.plotly_chart(fig_difficulty, use_container_width=True)

                    st.dataframe(course_stats[['Course_Name', 'Avg_Score', 'Avg_Progress', 'Avg_Rating', 'Enrollments', 'Difficulty_Level']], use_container_width=True)
        else:
            st.warning("Student_ID or Course_ID not found in dataset")

    elif page == "📄 Report Generator":
        st.header("📄 Automated Report Generator")
        st.write("Generate comprehensive reports for individual students or courses.")

        report_type = st.selectbox("Select Report Type",
                                  ["Student Performance", "Course Analysis"])

        if report_type == "Student Performance":
            if 'Student_ID' in df.columns:
                if 'Student_Name' in df.columns:
                    student_options = df[['Student_ID', 'Student_Name']].drop_duplicates()
                    student_options = student_options.sort_values('Student_Name')
                    student_display = [f"{row['Student_Name']} (ID: {row['Student_ID']})" 
                                    for idx, row in student_options.iterrows()]
                    student_mapping = {display: row['Student_ID'] 
                                     for display, (idx, row) in zip(student_display, student_options.iterrows())}
                    selected_student_display = st.selectbox("Select Student", student_display, key="student_select_report")
                    selected_student = student_mapping[selected_student_display]
                else:
                    student_list = sorted(df['Student_ID'].unique())
                    selected_student = st.selectbox("Select Student", student_list, key="student_select_report")
                    selected_student_display = f"Student {selected_student}"
                
                student_data = df[df['Student_ID'] == selected_student]
                
                if not student_data.empty:
                    if 'Student_Name' in student_data.columns:
                        student_name = student_data['Student_Name'].iloc[0]
                    else:
                        student_name = None

                    student_avg_score = student_data['Score'].mean() if 'Score' in student_data.columns else 0
                    student_avg_progress = student_data['Progress_Percent'].mean() if 'Progress_Percent' in student_data.columns else 0
                    student_avg_rating = student_data['Course_Rating'].mean() if 'Course_Rating' in student_data.columns else 0
                    num_courses = student_data['Course_ID'].nunique() if 'Course_ID' in student_data.columns else 0

                    courses_taken = []
                    if 'Course_ID' in student_data.columns:
                        if 'Course_Name' in student_data.columns:
                            courses_taken = student_data[['Course_ID', 'Course_Name']].drop_duplicates()
                            course_list = "\n".join([f"- {row['Course_Name']} (ID: {row['Course_ID']})"
                                                    for idx, row in courses_taken.iterrows()])
                        else:
                            course_ids = student_data['Course_ID'].unique()
                            course_list = "\n".join([f"- Course {cid}" for cid in course_ids])
                    else:
                        course_list = "N/A"

                    if student_avg_score >= 85:
                        performance_level = "Excellent"
                        performance_emoji = "🏆"
                    elif student_avg_score >= 70:
                        performance_level = "Good"
                        performance_emoji = "👍"
                    elif student_avg_score >= 50:
                        performance_level = "Average"
                        performance_emoji = "📊"
                    else:
                        performance_level = "Below Average"
                        performance_emoji = "⚠️"

                    if 'Sentiment' in student_data.columns:
                        sentiment_counts = student_data['Sentiment'].value_counts()
                        sentiment_summary = ", ".join([f"{sent}: {count}" for sent, count in sentiment_counts.items()])
                    else:
                        sentiment_summary = "N/A"
                    if student_name:
                        report_content = f"""
# Student Performance Report
**Student Name:** {student_name}
**Student ID:** {selected_student}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Student Overview

### Basic Information
- **Student Name:** {student_name}
- **Student ID:** {selected_student}
- **Total Courses Enrolled:** {num_courses}
- **Performance Level:** {performance_emoji} {performance_level}

### Performance Metrics
- **Average Score:** {student_avg_score:.2f} / 100
- **Average Progress:** {student_avg_progress:.1f}%
- **Average Course Rating Given:** {student_avg_rating:.2f} / 5.0

---

## 📚 Courses Enrolled

{course_list}

---

## 📈 Detailed Performance Analysis

### Score Breakdown
"""
                    else:
                        report_content = f"""
# Student Performance Report
**Student ID:** {selected_student}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Student Overview

### Basic Information
- **Student ID:** {selected_student}
- **Total Courses Enrolled:** {num_courses}
- **Performance Level:** {performance_emoji} {performance_level}

### Performance Metrics
- **Average Score:** {student_avg_score:.2f} / 100
- **Average Progress:** {student_avg_progress:.1f}%
- **Average Course Rating Given:** {student_avg_rating:.2f} / 5.0

---

## 📚 Courses Enrolled

{course_list}

---

## 📈 Detailed Performance Analysis

### Score Breakdown
"""
                    # Individual course scores
                    if 'Score' in student_data.columns and 'Course_ID' in student_data.columns:
                        if 'Course_Name' in student_data.columns:
                            course_scores = student_data[['Course_Name', 'Course_ID', 'Score', 'Progress_Percent']].drop_duplicates()
                            for idx, row in course_scores.iterrows():
                                report_content += f"- **{row['Course_Name']}:** Score: {row['Score']:.2f}, Progress: {row['Progress_Percent']:.1f}%\n"
                        else:
                            course_scores = student_data[['Course_ID', 'Score', 'Progress_Percent']].drop_duplicates()
                            for idx, row in course_scores.iterrows():
                                report_content += f"- **Course {row['Course_ID']}:** Score: {row['Score']:.2f}, Progress: {row['Progress_Percent']:.1f}%\n"
                    
                    report_content += f"""
### Sentiment Analysis
- **Feedback Sentiment:** {sentiment_summary}

---

## 🎯 Performance Insights

"""
                    if student_avg_score >= 85:
                        report_content += "- ✅ **Excellent Performance:** Student is performing exceptionally well across all courses.\n"
                    elif student_avg_score >= 70:
                        report_content += "- 👍 **Good Performance:** Student is meeting expected standards.\n"
                    elif student_avg_score >= 50:
                        report_content += "- 📊 **Average Performance:** Student shows potential for improvement with additional support.\n"
                    else:
                        report_content += "- ⚠️ **Below Average Performance:** Student needs immediate attention and support.\n"
                    
                    if student_avg_progress >= 80:
                        report_content += "- ✅ **High Engagement:** Student is actively progressing through courses.\n"
                    elif student_avg_progress >= 50:
                        report_content += "- 📈 **Moderate Engagement:** Student is making progress but could improve.\n"
                    else:
                        report_content += "- 📢 **Low Engagement:** Student needs encouragement to increase course participation.\n"
                    
                    report_content += f"""
---

## 💡 Recommendations

"""
                    if student_avg_score < 70:
                        report_content += "1. **Academic Support:** Provide additional tutoring or study resources\n"
                    if student_avg_progress < 50:
                        report_content += "2. **Engagement Strategy:** Implement interventions to improve course progress\n"
                    if 'Sentiment' in student_data.columns and 'Negative' in student_data['Sentiment'].values:
                        report_content += "3. **Feedback Review:** Address concerns raised in negative feedback\n"
                    if student_avg_score >= 85:
                        report_content += "1. **Maintain Excellence:** Continue current study strategies and consider advanced courses\n"
                    
                    report_content += f"""
---

**Report Generated by:** Student Course Analyzer Platform
**Version:** 1.0
"""
                else:
                    st.warning(f"No data found for Student {selected_student}")
                    report_content = f"# Student Performance Report\n\nNo data available for Student {selected_student}."
            else:
                st.error("Student_ID column not found in dataset")
                report_content = "# Student Performance Report\n\nStudent_ID column not found in dataset."
        
        elif report_type == "Course Analysis":
            # Course selection
            if 'Course_ID' in df.columns:
                # Create course selection with names if available
                if 'Course_Name' in df.columns:
                    # Get unique courses with names
                    course_options = df[['Course_ID', 'Course_Name']].drop_duplicates()
                    # Sort by course name for better UX
                    course_options = course_options.sort_values('Course_Name')
                    # Create display list
                    course_display = []
                    course_mapping = {}
                    for idx, row in course_options.iterrows():
                        display_name = f"{row['Course_Name']} (ID: {row['Course_ID']})"
                        course_display.append(display_name)
                        course_mapping[display_name] = row['Course_ID']
                else:
                    # If no Course_Name, use Course_ID
                    course_ids = sorted(df['Course_ID'].unique())
                    course_display = [f"Course {cid}" for cid in course_ids]
                    course_mapping = {display: cid for display, cid in zip(course_display, course_ids)}
                
                if len(course_display) > 0:
                    selected_course_display = st.selectbox("Select Course", course_display, key="course_select_report")
                    selected_course_id = course_mapping.get(selected_course_display)
                    
                    if selected_course_id is not None:
                        # Get course data
                        course_data = df[df['Course_ID'] == selected_course_id]
                        
                        if not course_data.empty:
                            # Calculate course-specific metrics
                            course_avg_score = course_data['Score'].mean() if 'Score' in course_data.columns else 0
                            course_avg_progress = course_data['Progress_Percent'].mean() if 'Progress_Percent' in course_data.columns else 0
                            course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0
                            num_students = course_data['Student_ID'].nunique() if 'Student_ID' in course_data.columns else 0
                            total_enrollments = len(course_data)
                            
                            # Get course name
                            if 'Course_Name' in course_data.columns:
                                course_name = course_data['Course_Name'].iloc[0]
                            else:
                                course_name = f"Course {selected_course_id}"
                            
                            # Score distribution
                            if 'Score' in course_data.columns:
                                excellent = len(course_data[course_data['Score'] >= 85])
                                good = len(course_data[(course_data['Score'] >= 70) & (course_data['Score'] < 85)])
                                average = len(course_data[(course_data['Score'] >= 50) & (course_data['Score'] < 70)])
                                below_avg = len(course_data[course_data['Score'] < 50])
                            else:
                                excellent = good = average = below_avg = 0
                            
                            # Sentiment analysis
                            if 'Sentiment' in course_data.columns:
                                sentiment_counts = course_data['Sentiment'].value_counts()
                                sentiment_summary = ", ".join([f"{sent}: {count}" for sent, count in sentiment_counts.items()])
                            else:
                                sentiment_summary = "N/A"
                            
                            # Top students in this course
                            top_students_list = ""
                            if 'Score' in course_data.columns and 'Student_ID' in course_data.columns:
                                top_students = course_data.nlargest(5, 'Score')[['Student_ID', 'Score']].values
                                top_students_list = "\n".join([f"- Student {row[0]}: {row[1]:.2f}" for row in top_students])
                            
                            # Calculate statistical values before using in f-string
                            if 'Score' in course_data.columns:
                                median_score = f"{course_data['Score'].median():.2f}"
                                std_score = f"{course_data['Score'].std():.2f}"
                                min_score = f"{course_data['Score'].min():.2f}"
                                max_score = f"{course_data['Score'].max():.2f}"
                            else:
                                median_score = "N/A"
                                std_score = "N/A"
                                min_score = "N/A"
                                max_score = "N/A"
                            
                            # Generate report
                            report_content = f"""
# Course Analysis Report
**Course:** {course_name}
**Course ID:** {selected_course_id}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📚 Course Overview

### Basic Information
- **Course Name:** {course_name}
- **Course ID:** {selected_course_id}
- **Total Students Enrolled:** {num_students}
- **Total Enrollments:** {total_enrollments}

### Performance Metrics
- **Average Score:** {course_avg_score:.2f} / 100
- **Average Progress:** {course_avg_progress:.1f}%
- **Average Course Rating:** {course_avg_rating:.2f} / 5.0

---

## 📊 Performance Distribution

### Score Categories
- **Excellent (≥85):** {excellent} students ({excellent/total_enrollments*100:.1f}% of enrollments)
- **Good (70-84):** {good} students ({good/total_enrollments*100:.1f}% of enrollments)
- **Average (50-69):** {average} students ({average/total_enrollments*100:.1f}% of enrollments)
- **Below Average (<50):** {below_avg} students ({below_avg/total_enrollments*100:.1f}% of enrollments)

### Statistical Analysis
- **Median Score:** {median_score}
- **Standard Deviation:** {std_score}
- **Score Range:** {min_score} - {max_score}

---

## 🏆 Top Performing Students

{top_students_list if top_students_list else "N/A"}

---

## 📈 Engagement Metrics

- **Average Progress:** {course_avg_progress:.1f}%
- **Students with >80% Progress:** {len(course_data[course_data['Progress_Percent'] >= 80]) if 'Progress_Percent' in course_data.columns else 0}

---

## 😊 Student Sentiment

- **Feedback Sentiment:** {sentiment_summary}

---

## 🎯 Course Insights

"""
                            if course_avg_score >= 80:
                                report_content += "- ✅ **High Performance Course:** Students are performing very well in this course.\n"
                            elif course_avg_score >= 70:
                                report_content += "- 👍 **Good Performance Course:** Students are meeting expected standards.\n"
                            elif course_avg_score >= 50:
                                report_content += "- 📊 **Average Performance Course:** Course shows potential for improvement.\n"
                            else:
                                report_content += "- ⚠️ **Low Performance Course:** Course needs immediate review and enhancement.\n"
                            
                            if course_avg_rating >= 4.0:
                                report_content += "- ⭐ **Highly Rated:** Students are satisfied with this course.\n"
                            elif course_avg_rating >= 3.0:
                                report_content += "- 📝 **Moderately Rated:** Course has room for improvement.\n"
                            else:
                                report_content += "- ⚠️ **Low Rating:** Course requires significant improvements.\n"
                            
                            if course_avg_progress >= 80:
                                report_content += "- ✅ **High Engagement:** Students are actively progressing through the course.\n"
                            else:
                                report_content += "- 📢 **Low Engagement:** Students need more support to complete the course.\n"
                            
                            report_content += f"""
---

## 💡 Recommendations

"""
                            if course_avg_score < 70:
                                report_content += "1. **Content Review:** Review course materials and update content as needed\n"
                            if course_avg_rating < 3.5:
                                report_content += "2. **Quality Enhancement:** Improve course quality based on student feedback\n"
                            if course_avg_progress < 50:
                                report_content += "3. **Engagement Strategy:** Implement initiatives to improve student progress\n"
                            if below_avg > total_enrollments * 0.3:
                                report_content += "4. **Support Programs:** Provide additional support for struggling students\n"
                            if course_avg_score >= 80:
                                report_content += "1. **Maintain Excellence:** Continue current teaching methods and share best practices\n"
                            
                            report_content += f"""
---

**Report Generated by:** Student Course Analyzer Platform
**Version:** 1.0
"""
                        else:
                            st.warning(f"No data found for Course {selected_course_id}")
                            report_content = f"# Course Analysis Report\n\nNo data available for Course {selected_course_id}."
                    else:
                        st.error("Error: Could not find course ID for selected course")
                        report_content = "# Course Analysis Report\n\nError: Could not find course ID for selected course."
                else:
                    st.warning("No courses available in the dataset")
                    report_content = "# Course Analysis Report\n\nNo courses available in the dataset."
            else:
                st.error("Course_ID column not found in dataset")
                report_content = "# Course Analysis Report\n\nCourse_ID column not found in dataset."
        
        # Display report (only if report_content is defined)
        if 'report_content' in locals():
            st.subheader(f"📄 {report_type} Report")
            st.markdown(report_content)
            
            # Download options
            st.subheader("📥 Download Report")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as text file
                report_txt = report_content.encode('utf-8')
                report_filename = report_type.lower().replace(' ', '_')
                if report_type == "Student Performance" and 'selected_student' in locals():
                    report_filename += f"_student_{selected_student}"
                elif report_type == "Course Analysis" and 'selected_course_id' in locals():
                    report_filename += f"_course_{selected_course_id}"
                
                st.download_button(
                    label="📄 Download as Text File",
                    data=report_txt,
                    file_name=f"report_{report_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download summary statistics as CSV
                if report_type == "Student Performance" and 'student_data' in locals() and not student_data.empty:
                    summary_stats = {
                        'Metric': ['Student ID', 'Average Score', 'Average Progress', 'Average Rating', 'Courses Enrolled'],
                        'Value': [
                            selected_student if 'selected_student' in locals() else 'N/A',
                            student_avg_score if 'student_avg_score' in locals() else 0,
                            student_avg_progress if 'student_avg_progress' in locals() else 0,
                            student_avg_rating if 'student_avg_rating' in locals() else 0,
                            num_courses if 'num_courses' in locals() else 0
                        ]
                    }
                elif report_type == "Course Analysis" and 'course_data' in locals() and not course_data.empty:
                    summary_stats = {
                        'Metric': ['Course ID', 'Course Name', 'Average Score', 'Average Progress', 'Average Rating', 'Students Enrolled'],
                        'Value': [
                            selected_course_id if 'selected_course_id' in locals() else 'N/A',
                            course_name if 'course_name' in locals() else 'N/A',
                            course_avg_score if 'course_avg_score' in locals() else 0,
                            course_avg_progress if 'course_avg_progress' in locals() else 0,
                            course_avg_rating if 'course_avg_rating' in locals() else 0,
                            num_students if 'num_students' in locals() else 0
                        ]
                    }
                else:
                    summary_stats = {'Metric': ['No Data'], 'Value': ['N/A']}
                
                summary_df = pd.DataFrame(summary_stats)
                summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download Summary Statistics (CSV)",
                    data=summary_csv,
                    file_name=f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            st.info("💡 For PDF export functionality, additional libraries (reportlab, fpdf) can be installed.")
        else:
            st.warning("Please select a report type and make a selection to generate a report.")

else:
    st.info("👆 Please upload your Excel file to begin analysis.")
