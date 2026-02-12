"""Build student performance report content and summary for download."""

from datetime import datetime
import pandas as pd


def build_student_report(df, selected_student):
    """Build report markdown and summary dict for a selected student. Returns (report_content, summary_dict)."""
    student_data = df[df['Student_ID'] == selected_student]
    if student_data.empty:
        return f"# Student Performance Report\n\nNo data available for Student {selected_student}.", None

    student_name = student_data['Student_Name'].iloc[0] if 'Student_Name' in student_data.columns else None
    student_avg_score = student_data['Score'].mean() if 'Score' in student_data.columns else 0
    student_avg_progress = student_data['Progress_Percent'].mean() if 'Progress_Percent' in student_data.columns else 0
    student_avg_rating = student_data['Course_Rating'].mean() if 'Course_Rating' in student_data.columns else 0
    num_courses = student_data['Course_ID'].nunique() if 'Course_ID' in student_data.columns else 0

    if 'Course_ID' in student_data.columns:
        if 'Course_Name' in student_data.columns:
            courses_taken = student_data[['Course_ID', 'Course_Name']].drop_duplicates()
            course_list = "\n".join([f"- {row['Course_Name']} (ID: {row['Course_ID']})" for idx, row in courses_taken.iterrows()])
        else:
            course_list = "\n".join([f"- Course {cid}" for cid in student_data['Course_ID'].unique()])
    else:
        course_list = "N/A"

    if student_avg_score >= 85:
        performance_level = "Excellent"
    elif student_avg_score >= 70:
        performance_level = "Good"
    elif student_avg_score >= 50:
        performance_level = "Average"
    else:
        performance_level = "Below Average"

    sentiment_summary = "N/A"
    if 'Sentiment' in student_data.columns:
        sentiment_counts = student_data['Sentiment'].value_counts()
        sentiment_summary = ", ".join([f"{s}: {c}" for s, c in sentiment_counts.items()])

    if student_name:
        report_content = f"""
# Student Performance Report
**Student Name:** {student_name}
**Student ID:** {selected_student}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Student Overview
- **Student Name:** {student_name}
- **Student ID:** {selected_student}
- **Total Courses Enrolled:** {num_courses}
- **Performance Level:** {performance_level}
- **Average Score:** {student_avg_score:.2f} / 100
- **Average Progress:** {student_avg_progress:.1f}%
- **Average Course Rating Given:** {student_avg_rating:.2f} / 5.0

---

## Courses Enrolled
{course_list}

---

## Detailed Performance Analysis
### Score Breakdown
"""
    else:
        report_content = f"""
# Student Performance Report
**Student ID:** {selected_student}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Student Overview
- **Student ID:** {selected_student}
- **Total Courses Enrolled:** {num_courses}
- **Performance Level:** {performance_level}
- **Average Score:** {student_avg_score:.2f} / 100
- **Average Progress:** {student_avg_progress:.1f}%
- **Average Course Rating Given:** {student_avg_rating:.2f} / 5.0

---

## Courses Enrolled
{course_list}

---

## Detailed Performance Analysis
### Score Breakdown
"""

    if 'Score' in student_data.columns and 'Course_ID' in student_data.columns:
        if 'Course_Name' in student_data.columns:
            for idx, row in student_data[['Course_Name', 'Course_ID', 'Score', 'Progress_Percent']].drop_duplicates().iterrows():
                report_content += f"- **{row['Course_Name']}:** Score: {row['Score']:.2f}, Progress: {row['Progress_Percent']:.1f}%\n"
        else:
            for idx, row in student_data[['Course_ID', 'Score', 'Progress_Percent']].drop_duplicates().iterrows():
                report_content += f"- **Course {row['Course_ID']}:** Score: {row['Score']:.2f}, Progress: {row['Progress_Percent']:.1f}%\n"

    report_content += f"""
### Sentiment Analysis
- **Feedback Sentiment:** {sentiment_summary}

---

## Performance Insights
"""
    if student_avg_score >= 85:
        report_content += "- **Excellent Performance:** Student is performing exceptionally well.\n"
    elif student_avg_score >= 70:
        report_content += "- **Good Performance:** Student is meeting expected standards.\n"
    elif student_avg_score >= 50:
        report_content += "- **Average Performance:** Student shows potential for improvement.\n"
    else:
        report_content += "- **Below Average Performance:** Student needs immediate attention.\n"
    if student_avg_progress >= 80:
        report_content += "- **High Engagement:** Student is actively progressing.\n"
    elif student_avg_progress >= 50:
        report_content += "- **Moderate Engagement:** Student is making progress but could improve.\n"
    else:
        report_content += "- **Low Engagement:** Student needs encouragement.\n"

    report_content += "\n---\n## Recommendations\n\n"
    if student_avg_score < 70:
        report_content += "1. **Academic Support:** Provide additional tutoring or study resources\n"
    if student_avg_progress < 50:
        report_content += "2. **Engagement Strategy:** Implement interventions to improve progress\n"
    if 'Sentiment' in student_data.columns and (student_data['Sentiment'] == 'Negative').any():
        report_content += "3. **Feedback Review:** Address concerns from negative feedback\n"
    if student_avg_score >= 85:
        report_content += "1. **Maintain Excellence:** Continue current strategies and consider advanced courses\n"
    report_content += "\n---\n**Report Generated by:** Student Course Analyzer Platform\n**Version:** 1.0\n"

    summary_dict = {
        'Metric': ['Student ID', 'Average Score', 'Average Progress', 'Average Rating', 'Courses Enrolled'],
        'Value': [selected_student, student_avg_score, student_avg_progress, student_avg_rating, num_courses]
    }
    return report_content, summary_dict
