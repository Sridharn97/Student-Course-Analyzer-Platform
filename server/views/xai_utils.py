"""XAI utilities for explaining student score predictions using SHAP."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from data_loader import prepare_ml_data
import shap

@st.cache_resource
def get_trained_model_and_explainer(df, available_features):
    """Train a Random Forest model on the data and initialize a SHAP explainer."""
    X_train, X_test, y_train, y_test, _, _, _, feat_cols = prepare_ml_data(df, available_features)
    
    # Train a quick RF model for explanations
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Initialize the TreeExplainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_train, feat_cols

def generate_shap_explanation(student_id, df, available_features):
    """Generate SHAP values and visualization for a specific student."""
    model, explainer, X_train, feat_cols = get_trained_model_and_explainer(df, available_features)
    
    # Find the student's records in the dataframe
    student_records = df[df['Student_ID'] == student_id]
    if student_records.empty:
        return None, None
        
    # Preprocess student features to match X_train
    X_student = student_records[[c for c in available_features]].copy()
    if 'Sentiment' in student_records.columns:
        X_student['Sentiment_Code'] = student_records['Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2}).fillna(1)
    if 'Recommendation' in student_records.columns:
        rec = student_records['Recommendation']
        if pd.api.types.is_numeric_dtype(rec):
            X_student['Recommendation'] = rec.fillna(0)
        else:
            X_student['Recommendation_Code'] = (
                rec.astype(str).str.lower().str.contains('yes', na=False)
            ).astype(int)
            
    X_student = X_student.fillna(0)
    # Ensure columns match training data
    X_student = X_student[feat_cols]
    
    # Calculate average feature vector for the student
    X_student_mean = X_student.mean(axis=0).to_frame().T
    
    shap_values = explainer.shap_values(X_student_mean)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = expected_value[0]
        
    # Get the 1D arrays
    if isinstance(shap_values, list):
        shap_val_row = shap_values[0][0]
    else:
        shap_val_row = shap_values[0]
        
    prediction = expected_value + sum(shap_val_row)
    
    # Dictionary for human-readable feature names
    feature_name_mapping = {
        'Progress_Percent': 'Course Progress',
        'Credits': 'Course Credits',
        'Course_Rating': "Student's Course Rating",
        'Completion_Status': 'Course Completion Status',
        'Sentiment_Code': 'Feedback Sentiment',
        'Recommendation_Code': 'Course Recommendation'
    }
    
    def get_human_name(feat):
        return feature_name_mapping.get(feat, feat.replace('_', ' '))

    # Prepare text summary
    negative_features = [(feat_cols[i], shap_val_row[i]) for i in range(len(feat_cols)) if shap_val_row[i] < 0]
    negative_features.sort(key=lambda x: x[1]) # sort ascending (most negative first)
    
    text_summary = f"This student is predicted to score **{prediction:.1f}**, compared to an average student's score of **{expected_value:.1f}**. "
    
    if negative_features:
        most_neg_feat, most_neg_val = negative_features[0]
        human_feat = get_human_name(most_neg_feat).lower()
        text_summary += f"The main reason for this lower prediction is their **{human_feat}**, which dropped their expected score by {-most_neg_val:.1f} points."
    else:
        text_summary += "There are no significant negative factors driving their score down."
        
    # Create the Waterfall chart data with human-readable labels
    human_feat_cols = [get_human_name(f) for f in feat_cols]
    
    x_labels = ["Typical Average"] + human_feat_cols + ["Student's Predicted Score"]
    measures = ["absolute"] + ["relative"] * len(feat_cols) + ["total"]
    y_values = [expected_value] + list(shap_val_row) + [prediction]
    text_values = [f"{expected_value:.1f}"] + [f"{v:+.1f}" for v in shap_val_row] + [f"{prediction:.1f}"]
    
    fig = go.Figure(go.Waterfall(
        name="Explanation",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=text_values,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#EF553B"}},
        increasing={"marker": {"color": "#00CC96"}},
        totals={"marker": {"color": "#636EFA"}}
    ))
    
    fig.update_layout(
        title="Why is this student predicted this score?",
        showlegend=False,
        waterfallgap=0.3,
        xaxis_tickangle=-45,
        margin=dict(t=50, b=100) # give space for rotated labels
    )
    
    return fig, text_summary
