"""Main Dashboard: advanced analytics tabs, search, and insights."""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_dashboard_analytics(df):
    st.subheader("Advanced Analytics")
    analytics_tabs = st.tabs(["Student Performance", "Course Analysis", "Engagement Metrics", "Data Export"])

    with analytics_tabs[0]:
        st.write("### Top Performing Students")
        if 'Score' in df.columns:
            top_students = df.nlargest(10, 'Score')[['Student_ID', 'Score', 'Progress_Percent']].reset_index(drop=True)
            st.dataframe(top_students, use_container_width=True)
            st.write("### Score Distribution")
            fig_dist = px.histogram(
                df, x='Score', nbins=20, title="Student Score Distribution",
                labels={'Score': 'Score', 'count': 'Number of Students'}
            )
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
                fig_course = px.bar(
                    x=course_rating.index, y=course_rating.values,
                    labels={'x': 'Course ID', 'y': 'Average Rating'},
                    title="Course Ratings Comparison"
                )
                st.plotly_chart(fig_course, use_container_width=True)

    with analytics_tabs[2]:
        st.write("### Engagement & Sentiment Analysis")
        col_eng1, col_eng2 = st.columns(2)
        with col_eng1:
            if 'Progress_Percent' in df.columns:
                st.metric("Average Progress", f"{df['Progress_Percent'].mean():.1f}%")
        with col_eng2:
            if 'Score' in df.columns:
                st.metric("Median Score", f"{df['Score'].median():.2f}")
        if 'Sentiment' in df.columns:
            st.write("### Sentiment Breakdown")
            sentiment_counts = df['Sentiment'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values, names=sentiment_counts.index,
                title="Student Feedback Sentiment Distribution"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        if 'Progress_Percent' in df.columns and 'Score' in df.columns:
            st.write("### Progress vs Performance Level")
            df_copy = df.copy()
            df_copy['Performance_Level'] = pd.cut(
                df_copy['Score'], bins=[0, 50, 70, 85, 100],
                labels=['Below Average', 'Average', 'Good', 'Excellent']
            )
            perf_progress = df_copy.groupby('Performance_Level')['Progress_Percent'].mean()
            fig_perf = px.bar(
                x=perf_progress.index, y=perf_progress.values,
                labels={'x': 'Performance Level', 'y': 'Average Progress (%)'},
                title="Average Progress by Performance Level", color=perf_progress.index
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    with analytics_tabs[3]:
        st.write("### Export Data")
        summary_stats = {
            'Metric': [
                'Total Students', 'Total Courses', 'Average Score', 'Average Progress',
                'Average Rating', 'Total Enrollments'
            ],
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
        st.download_button(label="Download Full Dataset (CSV)", data=csv,
                          file_name="student_course_analysis.csv", mime="text/csv")
        summary_csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Summary Statistics (CSV)", data=summary_csv,
                          file_name="summary_statistics.csv", mime="text/csv")

    st.subheader("Student & Course Search")
    search_col1, search_col2 = st.columns(2)
    with search_col1:
        st.write("### Search by Student")
        if 'Student_ID' in df.columns:
            student_search = st.selectbox("Select a Student", df['Student_ID'].unique(), key="student_search")
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
                course_name_search = st.selectbox("Select a Course", list(course_name_map.keys()), key="course_search")
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

    st.subheader("Key Insights & Recommendations")
    insights = []
    if 'Score' in df.columns:
        avg_score = df['Score'].mean()
        if avg_score >= 80:
            insights.append("**Excellent Performance:** Students are performing very well overall.")
        elif avg_score >= 70:
            insights.append("**Good Performance:** Students are meeting expected standards.")
        else:
            insights.append("**Improvement Needed:** Consider additional support or intervention programs.")
    if 'Progress_Percent' in df.columns:
        avg_progress = df['Progress_Percent'].mean()
        if avg_progress >= 80:
            insights.append("**High Engagement:** Students are actively progressing through courses.")
        else:
            insights.append("**Low Engagement:** Encourage students to increase course participation.")
    if 'Sentiment' in df.columns:
        positive_sentiment = (df['Sentiment'] == 'Positive').sum() / len(df) * 100
        if positive_sentiment >= 70:
            insights.append("**Positive Feedback:** Students are satisfied with their learning experience.")
        elif positive_sentiment >= 50:
            insights.append("**Mixed Feedback:** Some students are satisfied while others need support.")
        else:
            insights.append("**Negative Feedback:** Address student concerns and improve course quality.")
    if insights:
        for insight in insights:
            st.markdown(insight)
    st.markdown("""
    ---
    ### Recommendations
    - Use the prediction model to identify at-risk students and provide proactive support
    - Analyze course ratings to identify high-performing and underperforming courses
    - Monitor student sentiment to improve engagement and satisfaction
    - Regularly track progress metrics to ensure students stay on track
    """)
