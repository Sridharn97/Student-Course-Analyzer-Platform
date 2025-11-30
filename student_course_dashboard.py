

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Try importing advanced libraries (optional)
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

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Main Dashboard", "🤖 ML Models Comparison", 
     "⚠️ At-Risk Detection", "🎯 Course Recommendations", "📄 Report Generator"]
)

uploaded_file = st.file_uploader("📂 Upload your Excel file (Course_Analysis_Prediction_Advanced.xlsx)", type=["xlsx"])

if uploaded_file:
    # Load Excel sheets
    xls = pd.ExcelFile(uploaded_file)
    st.success("✅ File uploaded successfully!")

    courses = xls.parse('Courses')
    students = xls.parse('Students')
    enrollments = xls.parse('Enrollments')
    feedback = xls.parse('Feedback')
    platform = xls.parse('Platform_Performance')

    # Merge datasets
    df = enrollments.merge(students, on='Student_ID', how='left')\
                    .merge(courses, on='Course_ID', how='left')\
                    .merge(feedback[['Student_ID', 'Sentiment', 'Recommendation']], on='Student_ID', how='left')

    # Data preprocessing (existing code)
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

    # Prepare features for ML models
    available_features = []
    for feat in ['Progress_Percent', 'Credits', 'Course_Rating']:
        if feat in df.columns:
            available_features.append(feat)
    
    if len(available_features) > 0:
        X = df[available_features].fillna(0)
        y = df['Score'].fillna(df['Score'].mean())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # ============================================
    # PAGE 1: MAIN DASHBOARD (Existing Features)
    # ============================================
    if page == "📊 Main Dashboard":
        st.subheader("�� Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Average Score", f"{df['Score'].mean():.2f}")
        col2.metric("⭐ Average Course Rating", f"{df['Course_Rating'].mean():.2f}")
        col3.metric("🎯 Avg Progress (%)", f"{df['Progress_Percent'].mean():.2f}")

        
        st.subheader("📈 ANOVA Test – Platform vs Score")
        if 'Platform' in df.columns and 'Score' in df.columns:
            groups = [g["Score"].values for _, g in df.groupby("Platform")]
            if len(groups) > 1:
                anova_result = f_oneway(*groups)
                st.write(f"**F-Statistic:** {anova_result.statistic:.4f}, **p-value:** {anova_result.pvalue:.4f}")
                if anova_result.pvalue < 0.05:
                    st.success("✅ Significant difference in scores across platforms.")
                else:
                    st.warning("❌ No significant difference found between platforms.")
            else:
                st.info("ℹ️ Not enough groups for ANOVA test")
        else:
            st.info("ℹ️ Platform or Score column not found in data")

        st.subheader("📉 Chi-Square Test – Sentiment vs Completion")
        if 'Sentiment' in df.columns and 'Completion_Status' in df.columns:
            chi_table = pd.crosstab(df['Sentiment'], df['Completion_Status'])
            chi2, p, dof, exp = chi2_contingency(chi_table)
            st.write(f"**Chi2:** {chi2:.4f}, **p-value:** {p:.4f}")
            if p < 0.05:
                st.success("✅ Sentiment and completion status are significantly related.")
            else:
                st.warning("❌ No significant relationship between sentiment and completion.")
        else:
            st.info("ℹ️ Sentiment or Completion_Status column not found in data")

        # -----------------------------------------
        # 6️⃣ Random Forest Regression (Predict Scores)
        # -----------------------------------------
        st.subheader("🤖 Random Forest Regression – Predict Student Scores")

        # Use available features for the model
        available_features = []
        for feat in ['Progress_Percent', 'Credits', 'Course_Rating']:
            if feat in df.columns:
                available_features.append(feat)
        
        if len(available_features) == 0:
            st.warning("⚠️ Not enough features available for prediction model")
        else:
            features = available_features
            target = 'Score'

            X = df[features].fillna(0)
            y = df[target].fillna(df[target].mean())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**Mean Squared Error:** {mse:.3f}")

            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            st.bar_chart(importance.set_index('Feature'))

        # -----------------------------------------
        # 7️⃣ Visual Dashboard
        # -----------------------------------------
        st.subheader("�� Visual Dashboard")

        tab1, tab2, tab3, tab4 = st.tabs(["Platform Comparison", "Sentiment", "Progress vs Score", "Feature Importance"])

        with tab1:
            if 'Platform' in df.columns and 'Score' in df.columns:
                st.write("### Platform Performance Comparison")
                
                # Box plot
                fig1 = px.box(df, x='Platform', y='Score', color='Platform', title="Platform-wise Score Distribution")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Detailed platform comparison
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
                    
                    # Sort by average score
                    platform_stats_sorted = platform_stats.sort_values('Avg Score', ascending=False)
                    
                    st.dataframe(platform_stats_sorted, use_container_width=True)
                    
                    # Platform ranking visualization
                    st.write("### Platform Ranking by Performance")
                    fig_rank = px.bar(platform_stats_sorted, x='Platform', y='Avg Score', 
                                     color='Avg Score',
                                     title="Platforms Ranked by Average Score",
                                     labels={'Avg Score': 'Average Score'},
                                     color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig_rank, use_container_width=True)
                    
                    # Performance metrics comparison
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
                    
                    # Best Platform Analysis
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
                    
                    # Comparative insights
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
                    
                    # Recommendations
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
            if 'Sentiment' in df.columns:
                fig2 = px.pie(df, names='Sentiment', title="Feedback Sentiment Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Sentiment column not available")

        with tab3:
            if 'Progress_Percent' in df.columns and 'Score' in df.columns:
                color_col = 'Platform' if 'Platform' in df.columns else None
                size_col = 'Credits' if 'Credits' in df.columns else None
                fig3 = px.scatter(df, x='Progress_Percent', y='Score', color=color_col, size=size_col,
                                  title="Progress vs Score Correlation")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("ℹ️ Progress_Percent or Score column not available")

        with tab4:
            if len(available_features) > 0:
                fig4 = px.bar(importance, x='Feature', y='Importance', color='Feature',
                              title="Feature Importance (Random Forest)")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("ℹ️ Not enough features for importance visualization")

        # -----------------------------------------
        # 8️⃣ Live Prediction Section
        # -----------------------------------------
        if len(available_features) > 0:
            st.subheader("🔮 Live Performance Prediction")

            with st.form("prediction_form"):
                st.write("Enter student/course details below:")
                input_data = []
                
                if len(available_features) >= 1:
                    col1 = st.columns(1)
                    for i, feat in enumerate(available_features):
                        if feat == 'Progress_Percent':
                            val = st.number_input(f"{feat} (%)", min_value=0, max_value=100, value=70)
                        elif feat == 'Credits':
                            val = st.number_input(f"{feat}", min_value=1, max_value=10, value=3)
                        elif feat == 'Course_Rating':
                            val = st.number_input(f"{feat}", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
                        else:
                            val = st.number_input(f"{feat}", value=0.0)
                        input_data.append(val)

                submitted = st.form_submit_button("🔍 Predict Expected Score")

            if submitted:
                input_array = np.array([input_data])
                predicted_score = model.predict(input_array)[0]
                st.success(f"✅ Predicted Student Score: **{predicted_score:.2f}** / 100")

                if predicted_score >= 85:
                    st.balloons()
                    st.info("🏆 Excellent Performance Expected!")
                elif predicted_score >= 70:
                    st.info("👍 Good Performance Expected.")
                elif predicted_score >= 50:
                    st.warning("⚠️ Average Performance — improvement possible.")
                else:
                    st.error("❌ Below Expected Level — needs attention.")

        # -----------------------------------------
        # 9️⃣ Advanced Analytics Section
        # -----------------------------------------
        st.subheader("�� Advanced Analytics")
        
        analytics_tabs = st.tabs(["Student Performance", "Course Analysis", "Engagement Metrics", "Data Export"])
        
        with analytics_tabs[0]:
            st.write("### Top Performing Students")
            if 'Score' in df.columns:
                top_students = df.nlargest(10, 'Score')[['Student_ID', 'Score', 'Progress_Percent']].reset_index(drop=True)
                st.dataframe(top_students, use_container_width=True)
                
                # Performance distribution
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
                
                # Course rating comparison
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
            
            # Sentiment breakdown
            if 'Sentiment' in df.columns:
                st.write("### Sentiment Breakdown")
                sentiment_counts = df['Sentiment'].value_counts()
                fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                                      title="Student Feedback Sentiment Distribution")
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Progress by performance level
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
            
            # Summary statistics download
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
            
            # Download buttons
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

        # -----------------------------------------
        # 🔟 Student Search & Filter Section
        # -----------------------------------------
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
                # Create a mapping of course names to IDs
                course_name_map = {}
                if 'Course_Name' in df.columns:
                    course_data_temp = df[['Course_ID', 'Course_Name']].drop_duplicates()
                    for idx, row in course_data_temp.iterrows():
                        course_name_map[row['Course_Name']] = row['Course_ID']
                else:
                    # If no course name, use ID as name
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

        # -----------------------------------------
        # 💡 Insights Summary
        # -----------------------------------------
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

    # ============================================
    # PAGE 2: ML MODELS COMPARISON
    # ============================================
    elif page == "🤖 ML Models Comparison":
        st.header("🤖 Machine Learning Models Comparison")
        st.write("Compare multiple ML models to find the best predictor for student scores.")
        
        if len(available_features) == 0:
            st.warning("⚠️ Not enough features available for prediction models")
        else:
            models = {}
            results = {}
            
            # Random Forest
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
            
            # Gradient Boosting
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
            
            # Linear Regression
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
            
            # XGBoost (if available)
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
            
            # Comparison Table
            st.subheader("📊 Model Comparison")
            comparison_df = pd.DataFrame(results).T
            comparison_df = comparison_df.sort_values('R²', ascending=False)
            st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['R²'], 
                                name='R² Score', marker_color='lightblue'))
            fig.update_layout(title="Model Comparison - R² Scores", 
                            xaxis_title="Model", yaxis_title="R² Score")
            st.plotly_chart(fig, use_container_width=True)
            
            # Best Model
            best_model_name = comparison_df.index[0]
            st.success(f"🏆 Best Model: **{best_model_name}** with R² = {comparison_df.loc[best_model_name, 'R²']:.4f}")
            
            # Hyperparameter Tuning
            st.subheader("⚙️ Hyperparameter Tuning")
            if st.checkbox("Enable Grid Search (may take time)"):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                }
                grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                                         param_grid, cv=3, scoring='r2')
                grid_search.fit(X_train, y_train)
                st.write(f"Best Parameters: {grid_search.best_params_}")
                st.write(f"Best CV Score: {grid_search.best_score_:.4f}")

    # ============================================
    # PAGE 3: AT-RISK STUDENT DETECTION
    # ============================================
    elif page == "⚠️ At-Risk Detection":
        st.header("⚠️ At-Risk Student Detection System")
        st.write("Identify students who may need additional support.")
        
        # Risk scoring
        df['Risk_Score'] = 0.0
        
        # Low progress risk
        if 'Progress_Percent' in df.columns:
            progress_risk = (100 - df['Progress_Percent']) / 100 * 0.4
            df['Risk_Score'] += progress_risk
        
        # Low score risk
        if 'Score' in df.columns:
            score_risk = (100 - df['Score']) / 100 * 0.4
            df['Risk_Score'] += score_risk
        
        # Negative sentiment risk
        if 'Sentiment' in df.columns:
            sentiment_risk = (df['Sentiment'] == 'Negative').astype(int) * 0.2
            df['Risk_Score'] += sentiment_risk
        
        # Normalize risk score
        df['Risk_Score'] = df['Risk_Score'] * 100
        
        # Risk categories
        df['Risk_Level'] = pd.cut(df['Risk_Score'], 
                                 bins=[0, 30, 50, 70, 100],
                                 labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Display at-risk students
        risk_threshold = st.slider("Risk Threshold", 0, 100, 50)
        at_risk = df[df['Risk_Score'] >= risk_threshold].sort_values('Risk_Score', ascending=False)
        
        st.metric("At-Risk Students", len(at_risk), 
                 f"{len(at_risk)/len(df)*100:.1f}% of total")
        
        # Risk distribution
        fig = px.histogram(df, x='Risk_Score', nbins=20, 
                          title="Risk Score Distribution",
                          labels={'Risk_Score': 'Risk Score', 'count': 'Number of Students'})
        st.plotly_chart(fig, use_container_width=True)
        
        # At-risk students table
        st.subheader("🚨 High-Risk Students")
        if len(at_risk) > 0:
            display_cols = ['Student_ID', 'Risk_Score', 'Risk_Level', 'Score', 'Progress_Percent']
            display_cols = [col for col in display_cols if col in at_risk.columns]
            st.dataframe(at_risk[display_cols].head(20), use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Intervention Recommendations")
            for idx, row in at_risk.head(10).iterrows():
                st.write(f"**Student {row['Student_ID']}** (Risk: {row['Risk_Score']:.1f})")
                if row['Progress_Percent'] < 50:
                    st.write("- ⚠️ Low progress detected - recommend additional study time")
                if row['Score'] < 60:
                    st.write("- 📚 Low score - suggest tutoring or extra resources")
                if row.get('Sentiment') == 'Negative':
                    st.write("- 😞 Negative feedback - schedule counseling session")
                st.write("---")
        else:
            st.success("✅ No students above risk threshold!")

    # ============================================
    # PAGE 4: COURSE RECOMMENDATIONS
    # ============================================
    elif page == "🎯 Course Recommendations":
        st.header("🎯 Personalized Course Recommendation Engine")
        st.write("Get course recommendations based on student performance and preferences.")
        
        if 'Student_ID' in df.columns and 'Course_ID' in df.columns:
            student_id = st.selectbox("Select Student", df['Student_ID'].unique())
            
            # Get student's current courses and performance
            student_data = df[df['Student_ID'] == student_id]
            student_avg_score = student_data['Score'].mean()
            student_avg_rating = student_data['Course_Rating'].mean() if 'Course_Rating' in student_data.columns else 0
            
            # Get courses the student has already taken
            student_courses = student_data['Course_ID'].unique()
            
            # Get all available courses (with course names if available)
            if 'Course_Name' in df.columns:
                # Create mapping of Course_ID to Course_Name
                course_mapping = df[['Course_ID', 'Course_Name']].drop_duplicates().set_index('Course_ID')['Course_Name'].to_dict()
                all_courses_with_names = df[['Course_ID', 'Course_Name']].drop_duplicates()
                available_courses = all_courses_with_names[~all_courses_with_names['Course_ID'].isin(student_courses)]
            else:
                # If no Course_Name column, create a simple mapping
                all_course_ids = df['Course_ID'].unique()
                available_courses = pd.DataFrame({
                    'Course_ID': [c for c in all_course_ids if c not in student_courses],
                    'Course_Name': [f"Course {c}" for c in all_course_ids if c not in student_courses]
                })
                course_mapping = {cid: f"Course {cid}" for cid in all_course_ids}
            
            if len(available_courses) > 0:
                # Score courses based on similarity
                recommendations = []
                for idx, row in available_courses.iterrows():
                    course_id = row['Course_ID']
                    course_name = row['Course_Name'] if 'Course_Name' in row else course_mapping.get(course_id, f"Course {course_id}")
                    
                    course_data = df[df['Course_ID'] == course_id]
                    course_avg_score = course_data['Score'].mean()
                    course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0
                    
                    # Similarity score
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
                # Display with Course_Name as the primary column
                display_cols = ['Course_Name', 'Recommendation_Score', 'Avg_Course_Score', 'Avg_Course_Rating', 'Enrollments']
                st.dataframe(rec_df[display_cols].head(10), use_container_width=True)
                
                # Visualization - using Course_Name instead of Course_ID
                fig = px.bar(rec_df.head(10), x='Course_Name', y='Recommendation_Score',
                           title="Top 10 Course Recommendations",
                           labels={'Recommendation_Score': 'Recommendation Score', 'Course_Name': 'Course Name'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Student has enrolled in all available courses!")
        else:
            st.warning("Student_ID or Course_ID not found in dataset")

    # ============================================
    # PAGE 5: REPORT GENERATOR
    # ============================================
    elif page == "📄 Report Generator":
        st.header("📄 Automated Report Generator")
        st.write("Generate comprehensive reports for individual students or courses.")
        
        report_type = st.selectbox("Select Report Type", 
                                  ["Student Performance", "Course Analysis"])
        
        if report_type == "Student Performance":
            # Student selection
            if 'Student_ID' in df.columns:
                # Create student selection with names if available
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
                
                # Get student data
                student_data = df[df['Student_ID'] == selected_student]
                
                if not student_data.empty:
                    # Get student name if available
                    if 'Student_Name' in student_data.columns:
                        student_name = student_data['Student_Name'].iloc[0]
                    else:
                        student_name = None
                    
                    # Calculate student-specific metrics
                    student_avg_score = student_data['Score'].mean() if 'Score' in student_data.columns else 0
                    student_avg_progress = student_data['Progress_Percent'].mean() if 'Progress_Percent' in student_data.columns else 0
                    student_avg_rating = student_data['Course_Rating'].mean() if 'Course_Rating' in student_data.columns else 0
                    num_courses = student_data['Course_ID'].nunique() if 'Course_ID' in student_data.columns else 0
                    
                    # Get course names if available
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
                    
                    # Performance level
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
                    
                    # Sentiment analysis
                    if 'Sentiment' in student_data.columns:
                        sentiment_counts = student_data['Sentiment'].value_counts()
                        sentiment_summary = ", ".join([f"{sent}: {count}" for sent, count in sentiment_counts.items()])
                    else:
                        sentiment_summary = "N/A"
                    
                    # Generate report
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
