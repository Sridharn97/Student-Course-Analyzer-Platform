"""Helpers for at-risk student risk scoring and recommendation text."""

import pandas as pd
import streamlit as st


@st.cache_data
def calculate_risk_scores(df):
    """Aggregate by student and compute risk score and reasons."""
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
            for progress in student_agg['Avg_Progress']:
                progress_risk_scores.append(f"Very low progress ({progress:.1f}%)" if progress < 50 else (f"Low progress ({progress:.1f}%)" if progress < 70 else ""))

        progress_bar.progress(50)
        if 'Avg_Score' in student_agg.columns:
            score_risk = (100 - student_agg['Avg_Score']) / 100 * 0.4
            student_agg['Risk_Score'] += score_risk
            for score in student_agg['Avg_Score']:
                score_risk_scores.append(f"Very low scores ({score:.1f})" if score < 50 else (f"Low scores ({score:.1f})" if score < 65 else ""))

        if 'Sentiment' in student_agg.columns:
            sentiment_risk = (student_agg['Sentiment'] == 'Negative').astype(int) * 0.2
            student_agg['Risk_Score'] += sentiment_risk
            for sentiment in student_agg['Sentiment']:
                sentiment_risk_scores.append("Negative feedback/sentiment" if sentiment == 'Negative' else "")

        progress_bar.progress(80)
        student_agg['Risk_Score'] = student_agg['Risk_Score'] * 100
        student_agg['Risk_Level'] = pd.cut(student_agg['Risk_Score'], bins=[0, 30, 50, 70, 100], labels=['Low', 'Medium', 'High', 'Critical'])

        for i in range(len(student_agg)):
            reasons = [r for r in [progress_risk_scores[i], score_risk_scores[i], sentiment_risk_scores[i]] if r]
            student_agg.loc[student_agg.index[i], 'Risk_Reasons'] = "; ".join(reasons) if reasons else "Multiple minor risk factors"

        progress_bar.progress(100)
        progress_bar.empty()
    return student_agg


def get_recommendations_for_student(student_data):
    """Return list of recommendation strings for an at-risk student row."""
    recs = []
    reasons = student_data['Risk_Reasons']
    if "Very low progress" in reasons:
        recs.extend(["**Immediate Action Required** - Schedule daily check-ins with instructor", "**Study Schedule** - Create a structured daily study plan", "**Study Buddy** - Pair with a high-performing student", "**Progress Tracking** - Set weekly progress milestones"])
    elif "Low progress" in reasons:
        recs.extend(["**Progress Monitoring** - Weekly check-ins with academic advisor", "**Resource Access** - Ensure access to all required materials", "**Goal Setting** - Break down objectives into weekly goals"])
    if "Very low scores" in reasons:
        recs.extend(["**Academic Tutoring** - Immediate enrollment in tutoring", "**Supplemental Resources** - Additional practice materials", "**Learning Assessment** - Learning style assessment", "**Practice Exams** - Regular practice with feedback"])
    elif "Low scores" in reasons:
        recs.extend(["**Study Groups** - Join or form study groups", "**Targeted Review** - Focus on weak areas", "**Grade Tracking** - Monitor assignment grades"])
    if "Negative feedback" in reasons:
        recs.extend(["**Counseling Support** - Meeting with academic counselor", "**Feedback Discussion** - One-on-one about course experience", "**Course Adjustment** - Consider course load adjustment", "**Motivation Support** - Student success programs"])
    if student_data['Risk_Level'] == 'Critical':
        recs = ["**CRITICAL PRIORITY** - Immediate intervention by academic affairs", "**Parent/Guardian Contact** - Notify family", "**Academic Probation** - Consider probation status"] + recs
    return recs
