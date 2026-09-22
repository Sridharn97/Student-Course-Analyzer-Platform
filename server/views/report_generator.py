"""Report Generator page: select type, student/course, display report, and download."""

from datetime import datetime
import pandas as pd
import streamlit as st

from .report_student import build_student_report
from .report_course import build_course_report


def render_report_generator(df):
    st.header("Automated Report Generator")
    st.write("Generate comprehensive reports for individual students or courses.")
    report_type = st.selectbox("Select Report Type", ["Student Performance", "Course Analysis"])

    report_content = None
    summary_dict = None
    report_filename_suffix = ""

    if report_type == "Student Performance":
        if 'Student_ID' not in df.columns:
            st.error("Student_ID column not found in dataset")
            report_content = "# Student Performance Report\n\nStudent_ID column not found in dataset."
        else:
            if 'Student_Name' in df.columns:
                student_options = df[['Student_ID', 'Student_Name']].drop_duplicates().sort_values('Student_Name')
                student_display = [f"{row['Student_Name']} (ID: {row['Student_ID']})" for idx, row in student_options.iterrows()]
                student_mapping = {d: row['Student_ID'] for d, (idx, row) in zip(student_display, student_options.iterrows())}
                selected_display = st.selectbox("Select Student", student_display, key="student_select_report")
                selected_student = student_mapping[selected_display]
            else:
                student_list = sorted(df['Student_ID'].unique())
                selected_student = st.selectbox("Select Student", student_list, key="student_select_report")
            report_content, summary_dict = build_student_report(df, selected_student)
            report_filename_suffix = f"_student_{selected_student}"

    elif report_type == "Course Analysis":
        if 'Course_ID' not in df.columns:
            st.error("Course_ID column not found in dataset")
            report_content = "# Course Analysis Report\n\nCourse_ID column not found in dataset."
        else:
            if 'Course_Name' in df.columns:
                course_options = df[['Course_ID', 'Course_Name']].drop_duplicates().sort_values('Course_Name')
                course_display = [f"{row['Course_Name']} (ID: {row['Course_ID']})" for idx, row in course_options.iterrows()]
                course_mapping = {d: row['Course_ID'] for d, (idx, row) in zip(course_display, course_options.iterrows())}
                selected_display = st.selectbox("Select Course", course_display, key="course_select_report")
                selected_course_id = course_mapping.get(selected_display)
            else:
                course_ids = sorted(df['Course_ID'].unique())
                selected_course_id = st.selectbox("Select Course", course_ids, format_func=lambda x: f"Course {x}", key="course_select_report")
            if selected_course_id is not None:
                report_content, summary_dict = build_course_report(df, selected_course_id)
                report_filename_suffix = f"_course_{selected_course_id}"
            else:
                report_content = "# Course Analysis Report\n\nNo course selected."
                report_filename_suffix = ""

    if report_content is not None:
        st.subheader(f"{report_type} Report")
        st.markdown(report_content)
        st.subheader("Download Report")
        col1, col2 = st.columns(2)
        with col1:
            report_txt = report_content.encode('utf-8')
            fn = report_type.lower().replace(' ', '_') + report_filename_suffix + "_" + datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(label="Download as Text File", data=report_txt, file_name=f"report_{fn}.txt", mime="text/plain")
        with col2:
            if summary_dict:
                summary_df = pd.DataFrame(summary_dict)
                summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Summary Statistics (CSV)", data=summary_csv,
                                  file_name=f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
            else:
                st.caption("Summary CSV not available for this report.")
        st.info("For PDF export functionality, additional libraries (reportlab, fpdf) can be installed.")
    else:
        st.warning("Please select a report type and make a selection to generate a report.")
