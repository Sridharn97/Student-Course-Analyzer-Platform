"""Main Dashboard: dataset preview, key metrics, and platform/sentiment/performance tabs."""

import streamlit as st
import pandas as pd
import plotly.express as px


def render_main_dashboard(df):
    st.subheader("Dataset Preview")

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

    st.markdown("### Key Metrics Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{df['Score'].mean():.2f}")
    col2.metric("Average Course Rating", f"{df['Course_Rating'].mean():.2f}")
    col3.metric("Avg Progress (%)", f"{df['Progress_Percent'].mean():.2f}")

    st.markdown("---")
    st.subheader("Visual Dashboard")
    tab1, tab2, tab3 = st.tabs(["Platform Analytics", "Sentiment Analysis", "Performance Insights"])

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
                platform_stats.columns = [
                    'Avg Score', 'Median Score', 'Std Dev', 'Min Score', 'Max Score',
                    'Total Students', 'Avg Progress', 'Avg Rating'
                ]
                platform_stats = platform_stats.reset_index()
                platform_stats_sorted = platform_stats.sort_values('Avg Score', ascending=False)
                st.dataframe(platform_stats_sorted, use_container_width=True)
                st.write("### Platform Ranking by Performance")
                fig_rank = px.bar(
                    platform_stats_sorted, x='Platform', y='Avg Score', color='Avg Score',
                    title="Platforms Ranked by Average Score",
                    labels={'Avg Score': 'Average Score'}, color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_rank, use_container_width=True)
                st.write("### Detailed Platform Metrics Comparison")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.write("**Score Metrics**")
                    fig_score_compare = px.bar(
                        platform_stats_sorted, x='Platform', y=['Avg Score', 'Median Score'],
                        title="Average vs Median Score by Platform", barmode='group'
                    )
                    st.plotly_chart(fig_score_compare, use_container_width=True)
                with metrics_col2:
                    st.write("**Engagement & Rating**")
                    fig_engagement = px.scatter(
                        platform_stats_sorted, x='Avg Progress', y='Avg Rating',
                        size='Total Students', color='Platform',
                        title="Progress vs Rating (bubble size = students)",
                        labels={'Avg Progress': 'Average Progress (%)', 'Avg Rating': 'Average Rating'}
                    )
                    st.plotly_chart(fig_engagement, use_container_width=True)
                with metrics_col3:
                    st.write("**Consistency (Lower is Better)**")
                    fig_consistency = px.bar(
                        platform_stats_sorted, x='Platform', y='Std Dev',
                        title="Score Consistency by Platform (Lower = More Consistent)",
                        color='Std Dev', color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_consistency, use_container_width=True)
                st.write("---")
                st.write("### Best Platform Analysis")
                best_platform = platform_stats_sorted.iloc[0]
                worst_platform = platform_stats_sorted.iloc[-1]
                best_col1, best_col2 = st.columns(2)
                with best_col1:
                    st.success(f"**Best Performing Platform: {best_platform['Platform']}**")
                    st.write(f"- Average Score: **{best_platform['Avg Score']:.2f}**")
                    st.write(f"- Median Score: **{best_platform['Median Score']:.2f}**")
                    st.write(f"- Total Students: **{int(best_platform['Total Students'])}**")
                    st.write(f"- Average Progress: **{best_platform['Avg Progress']:.1f}%**")
                    st.write(f"- Average Rating: **{best_platform['Avg Rating']:.2f}**")
                    st.write(f"- Score Consistency (Std Dev): **{best_platform['Std Dev']:.2f}**")
                with best_col2:
                    st.warning(f"**Platform Needing Improvement: {worst_platform['Platform']}**")
                    st.write(f"- Average Score: **{worst_platform['Avg Score']:.2f}**")
                    st.write(f"- Median Score: **{worst_platform['Median Score']:.2f}**")
                    st.write(f"- Total Students: **{int(worst_platform['Total Students'])}**")
                    st.write(f"- Average Progress: **{worst_platform['Avg Progress']:.1f}%**")
                    st.write(f"- Average Rating: **{worst_platform['Avg Rating']:.2f}**")
                    st.write(f"- Score Consistency (Std Dev): **{worst_platform['Std Dev']:.2f}**")
                st.write("---")
                st.write("### Comparative Insights")
                score_diff = best_platform['Avg Score'] - worst_platform['Avg Score']
                progress_diff = best_platform['Avg Progress'] - worst_platform['Avg Progress']
                rating_diff = best_platform['Avg Rating'] - worst_platform['Avg Rating']
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                with insight_col1:
                    st.metric("Score Difference", f"{score_diff:.2f} points",
                             f"{(score_diff/worst_platform['Avg Score']*100):.1f}% higher")
                with insight_col2:
                    st.metric("Progress Difference", f"{progress_diff:.1f}%",
                             "Higher engagement" if progress_diff > 0 else "Lower engagement")
                with insight_col3:
                    st.metric("Rating Difference", f"{rating_diff:.2f} stars",
                             "Better satisfaction" if rating_diff > 0 else "Lower satisfaction")
                st.write("---")
                st.write("### Recommendations")
                st.markdown(f"""
                - **Best Practice Transfer**: Study {best_platform['Platform']}'s approach and implement similar strategies on {worst_platform['Platform']}
                - **Focus Areas**: {worst_platform['Platform']} should focus on improving score metrics and student engagement
                - **Success Factors**: {best_platform['Platform']}'s higher consistency (lower std dev) indicates reliable performance
                - **Quality Assurance**: Review {worst_platform['Platform']}'s content, instructors, and infrastructure
                """)
        else:
            st.info("Platform or Score column not available")

    with tab2:
        st.write("### Sentiment Analysis")
        if 'Sentiment' in df.columns:
            fig2 = px.pie(
                df, names='Sentiment', title="Student Feedback Sentiment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig2, use_container_width=True)
            if 'Platform' in df.columns:
                st.write("---")
                st.write("### Sentiment by Platform")
                sentiment_platform = pd.crosstab(df['Platform'], df['Sentiment'])
                fig_sentiment_platform = px.bar(
                    sentiment_platform, title="Sentiment Distribution Across Platforms",
                    barmode='stack', color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_sentiment_platform, use_container_width=True)
        else:
            st.info("Sentiment column not available")

    with tab3:
        st.write("### Performance Insights")
        if 'Progress_Percent' in df.columns and 'Score' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                color_col = 'Platform' if 'Platform' in df.columns else None
                fig3 = px.scatter(
                    df, x='Progress_Percent', y='Score', color=color_col,
                    title="Progress vs Score Correlation",
                    labels={'Progress_Percent': 'Progress (%)', 'Score': 'Score'}
                )
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                fig_score_dist = px.histogram(
                    df, x='Score', nbins=20, title="Score Distribution",
                    labels={'Score': 'Score', 'count': 'Number of Students'},
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig_score_dist, use_container_width=True)
            if 'Progress_Percent' in df.columns:
                st.write("---")
                st.write("### Progress Distribution")
                fig_progress_dist = px.histogram(
                    df, x='Progress_Percent', nbins=20,
                    title="Course Progress Distribution",
                    labels={'Progress_Percent': 'Progress (%)', 'count': 'Number of Students'},
                    color_discrete_sequence=['#00CC96']
                )
                st.plotly_chart(fig_progress_dist, use_container_width=True)
        else:
            st.info("Progress_Percent or Score column not available")
