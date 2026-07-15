"""ML Models Comparison page: train and compare regression models."""

import numpy as np
import pandas as pd
import streamlit as st
from .ui_utils import apply_premium_plotly_layout
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import XGBOOST_AVAILABLE
if XGBOOST_AVAILABLE:
    import xgboost as xgb


def render_ml_models(df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, available_features):
    st.header("Machine Learning Models Comparison")
    st.write("Compare multiple ML models to find the best predictor for student scores.")

    if len(available_features) == 0:
        st.warning("Not enough features available for prediction models.")
        return

    models = {}
    results = {}

    st.subheader("Random Forest Regressor")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'R²': r2_score(y_test, rf_pred),
        'MSE': mean_squared_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred))
    }

    st.subheader("Gradient Boosting Regressor")
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'R²': r2_score(y_test, gb_pred),
        'MSE': mean_squared_error(y_test, gb_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred))
    }

    st.subheader("Ridge Regression")
    lr_model = Ridge(alpha=1.0, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    models['Ridge Regression'] = lr_model
    results['Ridge Regression'] = {
        'R²': r2_score(y_test, lr_pred),
        'MSE': mean_squared_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred))
    }

    if XGBOOST_AVAILABLE:
        st.subheader("XGBoost Regressor")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, min_child_weight=3, random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        models['XGBoost'] = xgb_model
        results['XGBoost'] = {
            'R²': r2_score(y_test, xgb_pred),
            'MSE': mean_squared_error(y_test, xgb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred))
        }

    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comparison_df.index, y=comparison_df['R²'], name='R² Score', marker_color='lightblue'))
    fig.update_layout(title="Model Comparison - R² Scores", xaxis_title="Model", yaxis_title="R² Score")
    st.plotly_chart(apply_premium_plotly_layout(fig), use_container_width=True)
    best_model_name = comparison_df.index[0]
    st.success(f"Best Model: **{best_model_name}** with R² = {comparison_df.loc[best_model_name, 'R²']:.4f}")

    st.subheader("Advanced Model Analysis")
    analysis_tabs = st.tabs(["Cross-Validation Scores", "Feature Importance Comparison", "Model Robustness"])

    with analysis_tabs[0]:
        st.write("### Cross-Validation Performance")
        cv_results = {}
        for name, model in models.items():
            if name == 'Ridge Regression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_results[name] = {
                'Mean CV R²': cv_scores.mean(),
                'Std CV R²': cv_scores.std(),
                'Min CV R²': cv_scores.min(),
                'Max CV R²': cv_scores.max()
            }
        cv_df = pd.DataFrame(cv_results).T.round(4)
        st.dataframe(cv_df.style.highlight_max(axis=0, color='lightgreen'))
        fig_cv = go.Figure()
        for name in cv_results.keys():
            r = cv_results[name]
            fig_cv.add_trace(go.Box(
                y=[r['Min CV R²'], r['Mean CV R²'] - r['Std CV R²'], r['Mean CV R²'],
                   r['Mean CV R²'] + r['Std CV R²'], r['Max CV R²']],
                name=name, boxpoints=False
            ))
        fig_cv.update_layout(title="Cross-Validation R² Score Distribution", yaxis_title="R² Score")
        st.plotly_chart(apply_premium_plotly_layout(fig_cv), use_container_width=True)

    with analysis_tabs[1]:
        st.write("### Feature Importance Comparison")
        importance_data = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
            elif name == 'Ridge Regression' and hasattr(model, 'coef_'):
                importance_data[name] = np.abs(model.coef_)
        if importance_data:
            importance_df = pd.DataFrame(importance_data, index=available_features)
            importance_df = importance_df.div(importance_df.sum(axis=0), axis=1)
            fig_importance = px.imshow(
                importance_df.T, text_auto='.3f', aspect="auto",
                title="Feature Importance Across Models",
                labels=dict(x="Features", y="Models", color="Importance")
            )
            st.plotly_chart(apply_premium_plotly_layout(fig_importance), use_container_width=True)
            for feature in available_features:
                fig_feat = go.Figure()
                for model_name in importance_data.keys():
                    fig_feat.add_trace(go.Bar(
                        name=model_name, x=[feature],
                        y=[importance_df.loc[feature, model_name]]
                    ))
                fig_feat.update_layout(title=f"Feature Importance: {feature}", barmode='group')
                st.plotly_chart(apply_premium_plotly_layout(fig_feat), use_container_width=True)
        else:
            st.info("Feature importance not available for all models")

    with analysis_tabs[2]:
        st.write("### Model Robustness Analysis")
        robustness_metrics = {}
        for name, model in models.items():
            if name == 'Ridge Regression':
                y_pred_robust = model.predict(X_test_scaled)
            else:
                y_pred_robust = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_robust)
            mask_nonzero = y_test != 0
            mape = (np.abs((y_test[mask_nonzero] - y_pred_robust[mask_nonzero]) / y_test[mask_nonzero]).mean() * 100) if mask_nonzero.any() else 0.0
            max_error = np.max(np.abs(y_test - y_pred_robust))
            robustness_metrics[name] = {
                'MAE': mae, 'MAPE (%)': mape, 'Max Error': max_error,
                'Prediction Std': np.std(y_pred_robust)
            }
        robustness_df = pd.DataFrame(robustness_metrics).T.round(4)
        st.dataframe(robustness_df.style.highlight_min(axis=0, color='lightgreen'))
        fig_robust = go.Figure()
        for metric in ['MAE', 'MAPE (%)', 'Max Error']:
            fig_robust.add_trace(go.Bar(
                name=metric, x=list(robustness_metrics.keys()),
                y=[robustness_metrics[m][metric] for m in robustness_metrics.keys()]
            ))
        fig_robust.update_layout(title="Model Robustness Metrics", barmode='group', yaxis_title="Metric Value")
        st.plotly_chart(apply_premium_plotly_layout(fig_robust), use_container_width=True)
