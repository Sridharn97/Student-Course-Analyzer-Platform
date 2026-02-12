"""Data loading, preprocessing, and ML feature preparation."""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data
def load_and_process_data(uploaded_file):
    """Load Excel sheets and merge into main dataframe with basic preprocessing."""
    xls = pd.ExcelFile(uploaded_file)
    courses = xls.parse('Courses')
    students = xls.parse('Students')
    enrollments = xls.parse('Enrollments')
    feedback = xls.parse('Feedback')
    platform = xls.parse('Platform_Performance')

    df = (
        enrollments.merge(students, on='Student_ID', how='left')
        .merge(courses, on='Course_ID', how='left')
        .merge(feedback[['Student_ID', 'Sentiment', 'Recommendation']], on='Student_ID', how='left')
    )

    if 'Score' in df.columns:
        df['Score'].fillna(df['Score'].mean(), inplace=True)
    if 'Course_Rating' in df.columns:
        df['Course_Rating'].fillna(df['Course_Rating'].mean(), inplace=True)
    if 'Progress_Percent' in df.columns:
        df['Progress_Percent'].fillna(df['Progress_Percent'].mean(), inplace=True)
    if 'Credits' in df.columns:
        df['Credits'].fillna(df['Credits'].mean(), inplace=True)
    else:
        df['Credits'] = 1
    if 'Sentiment' in df.columns:
        df['Sentiment'].fillna('Neutral', inplace=True)
    else:
        df['Sentiment'] = 'Neutral'
    if 'Completion_Status' in df.columns:
        df['Completion_Status'] = df['Completion_Status'].astype('category').cat.codes
    else:
        df['Completion_Status'] = 1

    return df, courses, students, enrollments, feedback, platform


def ensure_df_filled(df):
    """Apply same fills/codes to df after load (for consistency)."""
    if 'Score' in df.columns:
        df['Score'].fillna(df['Score'].mean(), inplace=True)
    if 'Course_Rating' in df.columns:
        df['Course_Rating'].fillna(df['Course_Rating'].mean(), inplace=True)
    if 'Progress_Percent' in df.columns:
        df['Progress_Percent'].fillna(df['Progress_Percent'].mean(), inplace=True)
    if 'Credits' in df.columns:
        df['Credits'].fillna(df['Credits'].mean(), inplace=True)
    else:
        df['Credits'] = 1
    if 'Sentiment' in df.columns:
        df['Sentiment'].fillna('Neutral', inplace=True)
    else:
        df['Sentiment'] = 'Neutral'
    if 'Completion_Status' in df.columns:
        df['Completion_Status'] = df['Completion_Status'].astype('category').cat.codes
    else:
        df['Completion_Status'] = 1


@st.cache_data
def prepare_ml_data(df, base_feat_list):
    """Build feature matrix X and train/test split; return scaled data and feature names."""
    X = df[[c for c in base_feat_list]].copy()
    if 'Sentiment' in df.columns:
        X['Sentiment_Code'] = df['Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2}).fillna(1)
    if 'Recommendation' in df.columns:
        rec = df['Recommendation']
        if pd.api.types.is_numeric_dtype(rec):
            X['Recommendation'] = rec.fillna(0)
        else:
            X['Recommendation_Code'] = (
                rec.astype(str).str.lower().str.contains('yes', na=False)
            ).astype(int)
    X = X.fillna(0)
    y = df['Score'].fillna(df['Score'].mean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, list(X.columns)
