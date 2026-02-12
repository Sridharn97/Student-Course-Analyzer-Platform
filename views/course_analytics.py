"""Course Analytics tab content for the Course Recommendations page."""

import pandas as pd
import streamlit as st
import plotly.express as px


def render_course_analytics_tab(df):
    """Render the Course Performance Analytics tab content."""
    st.subheader("Course Performance Analytics")
    if 'Course_ID' not in df.columns:
        return
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
        fig_top = px.bar(top_courses, x='Course_Name', y='Avg_Score', title="Highest Scoring Courses",
                         labels={'Avg_Score': 'Average Score', 'Course_Name': 'Course'})
        fig_top.update_xaxes(tickangle=45)
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        st.write("### Most Popular Courses")
        popular_courses = course_stats.sort_values('Enrollments', ascending=False).head(10)
        fig_popular = px.bar(popular_courses, x='Course_Name', y='Enrollments', title="Most Enrolled Courses",
                             labels={'Enrollments': 'Number of Students', 'Course_Name': 'Course'})
        fig_popular.update_xaxes(tickangle=45)
        st.plotly_chart(fig_popular, use_container_width=True)

    st.write("### Course Difficulty Analysis")
    course_stats['Difficulty_Level'] = pd.cut(course_stats['Avg_Score'], bins=[0, 60, 75, 85, 100],
                                              labels=['Challenging', 'Moderate', 'Easy', 'Very Easy'])
    difficulty_counts = course_stats['Difficulty_Level'].value_counts()
    fig_difficulty = px.pie(values=difficulty_counts.values, names=difficulty_counts.index, title="Course Difficulty Distribution")
    st.plotly_chart(fig_difficulty, use_container_width=True)
    st.dataframe(course_stats[['Course_Name', 'Avg_Score', 'Avg_Progress', 'Avg_Rating', 'Enrollments', 'Difficulty_Level']], use_container_width=True)
