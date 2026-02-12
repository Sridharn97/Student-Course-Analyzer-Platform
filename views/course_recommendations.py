"""Course Recommendations page: general recs, at-risk support, and course analytics tab."""

import pandas as pd
import streamlit as st
import plotly.express as px

from .course_analytics import render_course_analytics_tab


@st.cache_data
def _get_at_risk_students(df):
    """Return dataframe of at-risk students (risk score >= 50) for course recs page."""
    agg = df.groupby('Student_ID').agg({
        'Progress_Percent': 'mean' if 'Progress_Percent' in df.columns else lambda x: 0,
        'Score': 'mean' if 'Score' in df.columns else lambda x: 0,
        'Sentiment': lambda x: 'Negative' if (x == 'Negative').any() else x.mode()[0] if len(x.mode()) > 0 else 'Neutral',
        'Student_Name': 'first' if 'Student_Name' in df.columns else lambda x: f"Student {x.name}"
    }).reset_index()
    agg.columns = ['Student_ID', 'Avg_Progress', 'Avg_Score', 'Sentiment', 'Student_Name']
    agg['Risk_Score'] = 0.0
    if 'Avg_Progress' in agg.columns:
        agg['Risk_Score'] += (100 - agg['Avg_Progress']) / 100 * 0.4
    if 'Avg_Score' in agg.columns:
        agg['Risk_Score'] += (100 - agg['Avg_Score']) / 100 * 0.4
    if 'Sentiment' in agg.columns:
        agg['Risk_Score'] += (agg['Sentiment'] == 'Negative').astype(int) * 0.2
    agg['Risk_Score'] = agg['Risk_Score'] * 100
    return agg[agg['Risk_Score'] >= 50].sort_values('Risk_Score', ascending=False)


@st.cache_data
def _get_support_course_recommendations(df, student_id):
    """Return support course recommendations for an at-risk student."""
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
        support_score += min(len(course_data) * 0.5, 15)
        reasons.append(f"{len(course_data)} students enrolled")
        if course_avg_score < 70:
            support_score += 10
            reasons.append("Suitable difficulty level")
        support_recommendations.append({
            'Course_Name': course_name, 'Course_ID': course_id, 'Support_Score': support_score,
            'Reasons': '; '.join(reasons), 'Avg_Score': course_avg_score, 'Avg_Progress': course_avg_progress,
            'Rating': course_avg_rating, 'Enrollments': len(course_data)
        })
    return pd.DataFrame(support_recommendations).sort_values('Support_Score', ascending=False)


def render_course_recommendations(df):
    st.header("Personalized Course Recommendation Engine")
    st.write("Get course recommendations based on student performance and preferences.")
    recommendation_tabs = st.tabs(["General Recommendations", "At-Risk Student Support", "Course Analytics"])

    if 'Student_ID' not in df.columns or 'Course_ID' not in df.columns:
        st.warning("Student_ID or Course_ID not found in dataset")
        return

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
            available_courses = pd.DataFrame({'Course_ID': [c for c in all_course_ids if c not in student_courses], 'Course_Name': [f"Course {c}" for c in all_course_ids if c not in student_courses]})
            course_mapping = {cid: f"Course {cid}" for cid in all_course_ids}

        if len(available_courses) > 0:
            recommendations = []
            for idx, row in available_courses.iterrows():
                course_id = row['Course_ID']
                course_name = row.get('Course_Name', course_mapping.get(course_id, f"Course {course_id}"))
                course_data = df[df['Course_ID'] == course_id]
                course_avg_score = course_data['Score'].mean()
                course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0
                score_similarity = 1 - abs(student_avg_score - course_avg_score) / 100
                rating_similarity = 1 - abs(student_avg_rating - course_avg_rating) / 5 if course_avg_rating > 0 else 0.5
                recommendation_score = (score_similarity * 0.6 + rating_similarity * 0.4) * 100
                recommendations.append({'Course_Name': course_name, 'Course_ID': course_id, 'Recommendation_Score': recommendation_score, 'Avg_Course_Score': course_avg_score, 'Avg_Course_Rating': course_avg_rating, 'Enrollments': len(course_data)})
            rec_df = pd.DataFrame(recommendations).sort_values('Recommendation_Score', ascending=False)
            st.subheader(f"Recommended Courses for Student {student_id}")
            display_cols = ['Course_Name', 'Recommendation_Score', 'Avg_Course_Score', 'Avg_Course_Rating', 'Enrollments']
            st.dataframe(rec_df[display_cols].head(10), use_container_width=True)
            fig = px.bar(rec_df.head(10), x='Course_Name', y='Recommendation_Score', title="Top 10 Course Recommendations", labels={'Recommendation_Score': 'Recommendation Score', 'Course_Name': 'Course Name'})
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Student has enrolled in all available courses.")

    with recommendation_tabs[1]:
        st.subheader("Support Courses for At-Risk Students")
        st.write("Specialized course recommendations for students needing additional support.")
        at_risk_students_list = _get_at_risk_students(df)
        if len(at_risk_students_list) > 0:
            risk_student_options = at_risk_students_list.apply(lambda x: f"{x['Student_Name']} (ID: {x['Student_ID']}) - Risk: {x['Risk_Score']:.1f}", axis=1)
            selected_risk_student = st.selectbox("Select at-risk student for support course recommendations", risk_student_options, key="risk_support_student")
            if selected_risk_student:
                student_id = selected_risk_student.split('(ID: ')[1].split(')')[0]
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
                    st.write("### Recommended Support Courses")
                    support_courses = _get_support_course_recommendations(df, student_id)
                    if len(support_courses) > 0:
                        display_cols = ['Course_Name', 'Support_Score', 'Reasons', 'Avg_Score', 'Avg_Progress', 'Rating']
                        st.dataframe(support_courses[display_cols].head(8), use_container_width=True)
                        st.write("### Implementation Plan")
                        top_course = support_courses.iloc[0]
                        for step in [f"1. **Enroll in {top_course['Course_Name']}** - Primary support course", f"2. **Schedule weekly tutoring** for {top_course['Course_Name']} (2 hours/week)", "3. **Join study group** within the first week", "4. **Set up progress check-ins** with instructor (bi-weekly)", "5. **Access supplemental materials**", "6. **Monitor progress weekly**"]:
                            st.markdown(f"- {step}")
                        st.write("### Success Metrics")
                        for metric in ["Complete at least 75% of weekly assignments", "Maintain average score above 70%", "Attend all tutoring sessions", "Submit progress reports weekly", "Complete course with passing grade"]:
                            st.markdown(f"- {metric}")
                    else:
                        st.info("No additional courses available for this student.")
        else:
            st.info("No at-risk students identified in the current dataset.")

    with recommendation_tabs[2]:
        render_course_analytics_tab(df)
