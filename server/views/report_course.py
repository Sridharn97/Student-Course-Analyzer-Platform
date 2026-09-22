"""Build course analysis report content and summary for download."""

from datetime import datetime
import pandas as pd


def build_course_report(df, selected_course_id):
    """Build report markdown and summary dict for a selected course. Returns (report_content, summary_dict)."""
    course_data = df[df['Course_ID'] == selected_course_id]
    if course_data.empty:
        return f"# Course Analysis Report\n\nNo data available for Course {selected_course_id}.", None

    course_name = course_data['Course_Name'].iloc[0] if 'Course_Name' in course_data.columns else f"Course {selected_course_id}"
    course_avg_score = course_data['Score'].mean() if 'Score' in course_data.columns else 0
    course_avg_progress = course_data['Progress_Percent'].mean() if 'Progress_Percent' in course_data.columns else 0
    course_avg_rating = course_data['Course_Rating'].mean() if 'Course_Rating' in course_data.columns else 0
    num_students = course_data['Student_ID'].nunique() if 'Student_ID' in course_data.columns else 0
    total_enrollments = len(course_data)

    if 'Score' in course_data.columns:
        excellent = len(course_data[course_data['Score'] >= 85])
        good = len(course_data[(course_data['Score'] >= 70) & (course_data['Score'] < 85)])
        average = len(course_data[(course_data['Score'] >= 50) & (course_data['Score'] < 70)])
        below_avg = len(course_data[course_data['Score'] < 50])
    else:
        excellent = good = average = below_avg = 0

    sentiment_summary = "N/A"
    if 'Sentiment' in course_data.columns:
        sentiment_counts = course_data['Sentiment'].value_counts()
        sentiment_summary = ", ".join([f"{s}: {c}" for s, c in sentiment_counts.items()])

    top_students_list = ""
    if 'Score' in course_data.columns and 'Student_ID' in course_data.columns:
        top_students = course_data.nlargest(5, 'Score')[['Student_ID', 'Score']].values
        top_students_list = "\n".join([f"- Student {row[0]}: {row[1]:.2f}" for row in top_students])

    if 'Score' in course_data.columns:
        median_score = f"{course_data['Score'].median():.2f}"
        std_score = f"{course_data['Score'].std():.2f}"
        min_score = f"{course_data['Score'].min():.2f}"
        max_score = f"{course_data['Score'].max():.2f}"
    else:
        median_score = std_score = min_score = max_score = "N/A"

    progress_80 = len(course_data[course_data['Progress_Percent'] >= 80]) if 'Progress_Percent' in course_data.columns else 0

    report_content = f"""
# Course Analysis Report
**Course:** {course_name}
**Course ID:** {selected_course_id}
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Course Overview
- **Course Name:** {course_name}
- **Course ID:** {selected_course_id}
- **Total Students Enrolled:** {num_students}
- **Total Enrollments:** {total_enrollments}
- **Average Score:** {course_avg_score:.2f} / 100
- **Average Progress:** {course_avg_progress:.1f}%
- **Average Course Rating:** {course_avg_rating:.2f} / 5.0

---

## Performance Distribution
- **Excellent (>=85):** {excellent} students ({excellent/total_enrollments*100:.1f}%)
- **Good (70-84):** {good} students ({good/total_enrollments*100:.1f}%)
- **Average (50-69):** {average} students ({average/total_enrollments*100:.1f}%)
- **Below Average (<50):** {below_avg} students ({below_avg/total_enrollments*100:.1f}%)
- **Median Score:** {median_score}
- **Standard Deviation:** {std_score}
- **Score Range:** {min_score} - {max_score}

---

## Top Performing Students
{top_students_list if top_students_list else "N/A"}

---

## Engagement Metrics
- **Average Progress:** {course_avg_progress:.1f}%
- **Students with >80% Progress:** {progress_80}

---

## Student Sentiment
- **Feedback Sentiment:** {sentiment_summary}

---

## Course Insights
"""
    if course_avg_score >= 80:
        report_content += "- **High Performance Course:** Students are performing very well.\n"
    elif course_avg_score >= 70:
        report_content += "- **Good Performance Course:** Students are meeting expected standards.\n"
    elif course_avg_score >= 50:
        report_content += "- **Average Performance Course:** Course shows potential for improvement.\n"
    else:
        report_content += "- **Low Performance Course:** Course needs immediate review.\n"
    if course_avg_rating >= 4.0:
        report_content += "- **Highly Rated:** Students are satisfied.\n"
    elif course_avg_rating >= 3.0:
        report_content += "- **Moderately Rated:** Course has room for improvement.\n"
    else:
        report_content += "- **Low Rating:** Course requires significant improvements.\n"
    if course_avg_progress >= 80:
        report_content += "- **High Engagement:** Students are actively progressing.\n"
    else:
        report_content += "- **Low Engagement:** Students need more support.\n"

    report_content += "\n---\n## Recommendations\n\n"
    if course_avg_score < 70:
        report_content += "1. **Content Review:** Review course materials and update content\n"
    if course_avg_rating < 3.5:
        report_content += "2. **Quality Enhancement:** Improve based on student feedback\n"
    if course_avg_progress < 50:
        report_content += "3. **Engagement Strategy:** Implement initiatives to improve progress\n"
    if below_avg > total_enrollments * 0.3:
        report_content += "4. **Support Programs:** Provide additional support for struggling students\n"
    if course_avg_score >= 80:
        report_content += "1. **Maintain Excellence:** Continue current teaching methods\n"
    report_content += "\n---\n**Report Generated by:** Student Course Analyzer Platform\n**Version:** 1.0\n"

    summary_dict = {
        'Metric': ['Course ID', 'Course Name', 'Average Score', 'Average Progress', 'Average Rating', 'Students Enrolled'],
        'Value': [selected_course_id, course_name, course_avg_score, course_avg_progress, course_avg_rating, num_students]
    }
    return report_content, summary_dict
