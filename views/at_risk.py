"""At-Risk Student Detection page."""

import pandas as pd
import streamlit as st
from .ui_utils import apply_premium_plotly_layout
import plotly.express as px

from .at_risk_utils import calculate_risk_scores, get_recommendations_for_student
from .xai_utils import generate_shap_explanation


def render_at_risk(df):
    st.header("At-Risk Student Detection System")
    st.write("Identify students who may need additional support.")

    if 'Student_ID' not in df.columns:
        st.error("Student_ID column not found in dataset")
        return

    student_risk_data = calculate_risk_scores(df)
    st.subheader("Set Risk Threshold")
    risk_threshold = st.slider("Risk Threshold", 0, 100, 50)
    at_risk_students = student_risk_data[student_risk_data['Risk_Score'] >= risk_threshold].sort_values('Risk_Score', ascending=False)

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    with col_metric1:
        st.metric("At-Risk Students", len(at_risk_students), f"{len(at_risk_students)/len(student_risk_data)*100:.1f}% of total")
    with col_metric2:
        st.metric("Total Students", len(student_risk_data))
    with col_metric3:
        safe_students = len(student_risk_data) - len(at_risk_students)
        st.metric("Safe Students", safe_students, f"{safe_students/len(student_risk_data)*100:.1f}% of total")

    fig = px.histogram(student_risk_data, x='Risk_Score', nbins=20, title="Risk Score Distribution",
                       labels={'Risk_Score': 'Risk Score (0-100)', 'count': 'Number of Students'})
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {risk_threshold}", annotation_position="top")
    st.plotly_chart(apply_premium_plotly_layout(fig), use_container_width=True)

    if 'Risk_Level' in student_risk_data.columns:
        risk_level_counts = student_risk_data['Risk_Level'].value_counts()
        fig_pie = px.pie(values=risk_level_counts.values, names=risk_level_counts.index, title="Students by Risk Level",
                         color_discrete_map={'Low': '#48bb78', 'Medium': '#ed8936', 'High': '#f56565', 'Critical': '#c53030'})
        st.plotly_chart(apply_premium_plotly_layout(fig_pie), use_container_width=True)

    st.subheader("High-Risk Students")
    if len(at_risk_students) == 0:
        st.success("No students above risk threshold.")
        return

    display_cols = ['Student_Name', 'Student_ID', 'Risk_Score', 'Risk_Level', 'Risk_Reasons', 'Avg_Score', 'Avg_Progress', 'Total_Courses']
    if 'Sentiment' in at_risk_students.columns:
        display_cols.append('Sentiment')
    display_cols = [c for c in display_cols if c in at_risk_students.columns]
    if len(at_risk_students) > 50:
        show_count = st.slider("Show top N students", 10, min(100, len(at_risk_students)), 20, key="risk_show_count")
        st.dataframe(at_risk_students[display_cols].head(show_count), use_container_width=True)
    else:
        st.dataframe(at_risk_students[display_cols], use_container_width=True)

    st.subheader("Detailed Intervention Recommendations")
    tabs = st.tabs(["Risk Breakdown", "Personalized Recommendations", "Risk Statistics"])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Risk Level Distribution")
            rlc = at_risk_students['Risk_Level'].value_counts()
            fig_risk_pie = px.pie(values=rlc.values, names=rlc.index, title="At-Risk Students by Risk Level",
                                  color_discrete_map={'Low': '#48bb78', 'Medium': '#ed8936', 'High': '#f56565', 'Critical': '#c53030'})
            st.plotly_chart(apply_premium_plotly_layout(fig_risk_pie), use_container_width=True)
        with col2:
            st.write("### Common Risk Factors")
            all_reasons = []
            for reasons in at_risk_students['Risk_Reasons'].str.split('; '):
                if isinstance(reasons, list):
                    all_reasons.extend(reasons)
            reason_counts = pd.Series(all_reasons).value_counts()
            fig_reasons = px.bar(x=reason_counts.index, y=reason_counts.values, title="Most Common Risk Factors", labels={'x': 'Risk Factor', 'y': 'Count'})
            fig_reasons.update_xaxes(tickangle=45)
            st.plotly_chart(apply_premium_plotly_layout(fig_reasons), use_container_width=True)

    with tabs[1]:
        st.write("### Individual Student Recommendations")
        options = at_risk_students.apply(lambda x: f"{x['Student_Name']} (ID: {x['Student_ID']}) - Risk: {x['Risk_Score']:.1f}", axis=1)
        selected_risk_student = st.selectbox("Select a student for detailed recommendations", options, key="risk_student_select")
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
                for rec in get_recommendations_for_student(student_data):
                    st.markdown(f"- {rec}")
                st.write("### Improvement Timeline")
                for t in ["**Week 1-2:** Establish study routine and initial assessment", "**Week 3-4:** Targeted interventions and progress monitoring", "**Week 5-8:** Advanced support and regular check-ins", "**Week 9-12:** Comprehensive review and long-term planning"]:
                    st.markdown(f"- {t}")

            st.write("---")
            st.write("### AI Score Explanation (XAI)")
            with st.spinner("Generating AI explanation..."):
                base_features = ['Progress_Percent', 'Credits', 'Course_Rating', 'Completion_Status']
                available_features = [f for f in base_features if f in df.columns]
                
                fig_shap, shap_text = generate_shap_explanation(student_id, df, available_features)
                if fig_shap:
                    st.markdown(shap_text)
                    st.plotly_chart(apply_premium_plotly_layout(fig_shap), use_container_width=True)
                else:
                    st.info("Could not generate explanation for this student.")

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
