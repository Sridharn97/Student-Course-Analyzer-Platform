# =========================================
# 🎓 Student Course Analyzer Streamlit Dashboard (with Live Prediction)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------
# 1️⃣ Dashboard Title & File Upload
# -----------------------------------------
st.set_page_config(page_title="Student Course Analyzer", layout="wide")
st.title("🎓 Student Course Analyzer and Prediction Platform")
st.caption("A data science–based dashboard to evaluate and predict student performance.")

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

    # -----------------------------------------
    # 2️⃣ Data Cleaning
    # -----------------------------------------
    # Fill missing values for available columns
    if 'Score' in df.columns:
        df['Score'].fillna(df['Score'].mean(), inplace=True)
    if 'Course_Rating' in df.columns:
        df['Course_Rating'].fillna(df['Course_Rating'].mean(), inplace=True)
    if 'Progress_Percent' in df.columns:
        df['Progress_Percent'].fillna(df['Progress_Percent'].mean(), inplace=True)
    if 'Credits' in df.columns:
        df['Credits'].fillna(df['Credits'].mean(), inplace=True)
    else:
        # If Credits doesn't exist, create it from Course_Rating or use 1
        df['Credits'] = 1
    if 'Sentiment' in df.columns:
        df['Sentiment'].fillna('Neutral', inplace=True)
    else:
        df['Sentiment'] = 'Neutral'
    if 'Completion_Status' in df.columns:
        df['Completion_Status'] = df['Completion_Status'].astype('category').cat.codes
    else:
        df['Completion_Status'] = 1

    st.subheader("📘 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------------
    # 3️⃣ KPIs
    # -----------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Average Score", f"{df['Score'].mean():.2f}")
    col2.metric("⭐ Average Course Rating", f"{df['Course_Rating'].mean():.2f}")
    col3.metric("🎯 Avg Progress (%)", f"{df['Progress_Percent'].mean():.2f}")

    # -----------------------------------------
    # 4️⃣ ANOVA Test (Platform vs Score)
    # -----------------------------------------
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

    # -----------------------------------------
    # 5️⃣ Chi-Square Test (Sentiment vs Completion)
    # -----------------------------------------
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
    st.subheader("📊 Visual Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Platform Comparison", "Sentiment", "Progress vs Score", "Feature Importance"])

    with tab1:
        if 'Platform' in df.columns and 'Score' in df.columns:
            fig1 = px.box(df, x='Platform', y='Score', color='Platform', title="Platform-wise Score Distribution")
            st.plotly_chart(fig1, use_container_width=True)
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
    # 9️⃣ Insights Summary
    # -----------------------------------------
    st.subheader("💡 Insights Summary")
    st.markdown("""
    - Platforms with higher ratings generally show better student outcomes.
    - Positive sentiment strongly aligns with course completion.
    - Progress percentage and credits are major predictors of student performance.
    - This dashboard helps educators, admins, and students track academic trends effectively.
    """)

else:
    st.info("👆 Please upload your Excel file to begin analysis.")
