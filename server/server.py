"""FastAPI backend for the Student Course Analyzer Platform.

Run with: uvicorn server:app --reload --port 8000
"""

import io
import os
import uuid
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── XGBoost optional ──────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

app = FastAPI(title="Student Course Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict[str, pd.DataFrame] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════

def _load_and_process(file_bytes: bytes) -> pd.DataFrame:
    """Load Excel file and merge all sheets into one master DataFrame."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = xls.sheet_names

    enrollments = xls.parse("Enrollments") if "Enrollments" in sheets else pd.DataFrame()
    students = xls.parse("Students") if "Students" in sheets else pd.DataFrame()
    courses = xls.parse("Courses") if "Courses" in sheets else pd.DataFrame()
    feedback = xls.parse("Feedback") if "Feedback" in sheets else pd.DataFrame()

    df = enrollments.copy()
    if not students.empty and "Student_ID" in df.columns:
        s_cols = [c for c in students.columns if c not in df.columns or c == "Student_ID"]
        df = df.merge(students[s_cols], on="Student_ID", how="left")
    if not courses.empty and "Course_ID" in df.columns:
        c_cols = [c for c in courses.columns if c not in df.columns or c == "Course_ID"]
        df = df.merge(courses[c_cols], on="Course_ID", how="left")
    if not feedback.empty and "Student_ID" in df.columns:
        f_cols = [c for c in ["Student_ID", "Sentiment", "Recommendation"] if c in feedback.columns and (c not in df.columns or c == "Student_ID")]
        df = df.merge(feedback[f_cols], on="Student_ID", how="left")

    # Fill missing values
    for col in ["Score", "Course_Rating", "Progress_Percent", "Credits"]:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
        elif col == "Credits":
            df["Credits"] = 1.0

    if "Sentiment" in df.columns:
        df["Sentiment"].fillna("Neutral", inplace=True)
    else:
        df["Sentiment"] = "Neutral"

    if "Completion_Status" in df.columns:
        df["Completion_Status"] = df["Completion_Status"].astype("category").cat.codes
    else:
        df["Completion_Status"] = 1

    return df


def _get_session(session_id: str) -> pd.DataFrame:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    return _sessions[session_id]


def _prepare_ml_data(df: pd.DataFrame):
    """Build feature matrix and train/test split."""
    base_features = ["Progress_Percent", "Credits", "Course_Rating", "Completion_Status"]
    available = [f for f in base_features if f in df.columns]
    if not available:
        return None

    X = df[available].copy()
    if "Sentiment" in df.columns:
        X["Sentiment_Code"] = df["Sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2}).fillna(1)
    if "Recommendation" in df.columns:
        rec = df["Recommendation"]
        if pd.api.types.is_numeric_dtype(rec):
            X["Recommendation"] = rec.fillna(0)
        else:
            X["Recommendation_Code"] = (
                rec.astype(str).str.lower().str.contains("yes", na=False)
            ).astype(int)
    X = X.fillna(0)
    y = df["Score"].fillna(df["Score"].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, list(X.columns)


def _calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate students and compute risk scores."""
    agg_dict = {}
    if "Progress_Percent" in df.columns:
        agg_dict["Progress_Percent"] = "mean"
    if "Score" in df.columns:
        agg_dict["Score"] = "mean"
    if "Sentiment" in df.columns:
        agg_dict["Sentiment"] = lambda x: "Negative" if (x == "Negative").any() else (x.mode()[0] if len(x.mode()) > 0 else "Neutral")
    if "Student_Name" in df.columns:
        agg_dict["Student_Name"] = "first"
    if "Course_ID" in df.columns:
        agg_dict["Course_ID"] = "count"

    student_agg = df.groupby("Student_ID").agg(agg_dict).reset_index()

    rename_map = {"Progress_Percent": "Avg_Progress", "Score": "Avg_Score", "Course_ID": "Total_Courses"}
    student_agg.rename(columns=rename_map, inplace=True)

    for col in ["Avg_Progress", "Avg_Score", "Student_Name", "Total_Courses"]:
        if col not in student_agg.columns:
            student_agg[col] = 0 if col != "Student_Name" else "Unknown"

    student_agg["Risk_Score"] = 0.0
    if "Avg_Progress" in student_agg.columns:
        student_agg["Risk_Score"] += (100 - student_agg["Avg_Progress"]) / 100 * 0.4
    if "Avg_Score" in student_agg.columns:
        student_agg["Risk_Score"] += (100 - student_agg["Avg_Score"]) / 100 * 0.4
    if "Sentiment" in student_agg.columns:
        student_agg["Risk_Score"] += (student_agg["Sentiment"] == "Negative").astype(int) * 0.2

    student_agg["Risk_Score"] = (student_agg["Risk_Score"] * 100).round(1)
    student_agg["Risk_Level"] = pd.cut(
        student_agg["Risk_Score"], bins=[-0.1, 30, 50, 70, 100.1],
        labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)

    def build_reasons(row):
        r = []
        if row.get("Avg_Progress", 100) < 50:
            r.append(f"Very low progress ({row['Avg_Progress']:.1f}%)")
        elif row.get("Avg_Progress", 100) < 70:
            r.append(f"Low progress ({row['Avg_Progress']:.1f}%)")
        if row.get("Avg_Score", 100) < 50:
            r.append(f"Very low scores ({row['Avg_Score']:.1f})")
        elif row.get("Avg_Score", 100) < 65:
            r.append(f"Low scores ({row['Avg_Score']:.1f})")
        if row.get("Sentiment", "") == "Negative":
            r.append("Negative feedback/sentiment")
        return "; ".join(r) if r else "Multiple minor risk factors"

    student_agg["Risk_Reasons"] = student_agg.apply(build_reasons, axis=1)
    return student_agg


def _safe_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe records."""
    return df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Student Course Analyzer API is running", "xgboost": XGBOOST_AVAILABLE}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload Excel file and create a session. Returns session_id."""
    content = await file.read()
    try:
        df = _load_and_process(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = df
    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": list(df.columns),
        "students": int(df["Student_ID"].nunique()) if "Student_ID" in df.columns else 0,
        "courses": int(df["Course_ID"].nunique()) if "Course_ID" in df.columns else 0,
    }


@app.post("/api/load-sample")
def load_sample():
    """Load default dataset from disk if available."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "Course Analysis Prediction Dataset.xlsx"),
        "Course Analysis Prediction Dataset.xlsx",
    ]
    sample_path = None
    for p in candidates:
        if os.path.exists(p):
            sample_path = p
            break

    if not sample_path:
        raise HTTPException(status_code=404, detail="Default dataset file not found on server.")

    with open(sample_path, "rb") as f:
        content = f.read()

    try:
        df = _load_and_process(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = df
    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": list(df.columns),
        "students": int(df["Student_ID"].nunique()) if "Student_ID" in df.columns else 0,
        "courses": int(df["Course_ID"].nunique()) if "Course_ID" in df.columns else 0,
    }


@app.get("/api/dashboard/{session_id}")
def get_dashboard(session_id: str):
    """Return key metrics and chart data for the main dashboard."""
    df = _get_session(session_id)

    metrics = {
        "total_students": int(df["Student_ID"].nunique()) if "Student_ID" in df.columns else 0,
        "total_courses": int(df["Course_ID"].nunique()) if "Course_ID" in df.columns else 0,
        "total_enrollments": len(df),
        "avg_score": round(float(df["Score"].mean()), 2) if "Score" in df.columns else 0,
        "avg_rating": round(float(df["Course_Rating"].mean()), 2) if "Course_Rating" in df.columns else 0,
        "avg_progress": round(float(df["Progress_Percent"].mean()), 2) if "Progress_Percent" in df.columns else 0,
    }

    # Platform analytics
    platform_stats = []
    if "Platform" in df.columns and "Score" in df.columns:
        ps = df.groupby("Platform").agg(
            avg_score=("Score", "mean"),
            median_score=("Score", "median"),
            std_dev=("Score", "std"),
            min_score=("Score", "min"),
            max_score=("Score", "max"),
            total_students=("Score", "count"),
            avg_progress=("Progress_Percent", "mean") if "Progress_Percent" in df.columns else ("Score", "count"),
            avg_rating=("Course_Rating", "mean") if "Course_Rating" in df.columns else ("Score", "count"),
        ).reset_index().round(2)
        platform_stats = _safe_records(ps)

    # Score distribution
    score_hist = []
    if "Score" in df.columns:
        counts, edges = np.histogram(df["Score"].dropna(), bins=20)
        score_hist = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(counts[i])} for i in range(len(counts))]

    # Progress distribution
    progress_hist = []
    if "Progress_Percent" in df.columns:
        counts, edges = np.histogram(df["Progress_Percent"].dropna(), bins=20)
        progress_hist = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(counts[i])} for i in range(len(counts))]

    # Sentiment breakdown
    sentiment_data = []
    if "Sentiment" in df.columns:
        sc = df["Sentiment"].value_counts().reset_index()
        sc.columns = ["sentiment", "count"]
        sentiment_data = _safe_records(sc)

    # Sentiment by platform
    sentiment_platform = []
    if "Sentiment" in df.columns and "Platform" in df.columns:
        sp = pd.crosstab(df["Platform"], df["Sentiment"]).reset_index()
        sentiment_platform = _safe_records(sp)

    # Top 10 students
    top_students = []
    if "Score" in df.columns and "Student_ID" in df.columns:
        cols = [c for c in ["Student_ID", "Student_Name", "Score", "Progress_Percent"] if c in df.columns]
        ts = df.nlargest(10, "Score")[cols].reset_index(drop=True)
        top_students = _safe_records(ts)

    # Course performance
    course_perf = []
    if "Course_ID" in df.columns and "Score" in df.columns:
        cp = df.groupby("Course_ID").agg(
            avg_score=("Score", "mean"),
            std_dev=("Score", "std"),
            students_enrolled=("Score", "count"),
            avg_progress=("Progress_Percent", "mean") if "Progress_Percent" in df.columns else ("Score", "count"),
        ).reset_index().round(2)
        if "Course_Name" in df.columns:
            cn = df[["Course_ID", "Course_Name"]].drop_duplicates()
            cp = cp.merge(cn, on="Course_ID", how="left")
        course_perf = _safe_records(cp)

    # Performance levels
    perf_levels = []
    if "Score" in df.columns and "Progress_Percent" in df.columns:
        df2 = df.copy()
        df2["Performance_Level"] = pd.cut(
            df2["Score"], bins=[0, 50, 70, 85, 100],
            labels=["Below Average", "Average", "Good", "Excellent"]
        )
        pl = df2.groupby("Performance_Level")["Progress_Percent"].mean().reset_index()
        pl.columns = ["level", "avg_progress"]
        perf_levels = _safe_records(pl)

    # Insights
    insights = []
    if "Score" in df.columns:
        avg = df["Score"].mean()
        if avg >= 80:
            insights.append({"type": "success", "text": "Excellent Performance: Students are performing very well overall."})
        elif avg >= 70:
            insights.append({"type": "info", "text": "Good Performance: Students are meeting expected standards."})
        else:
            insights.append({"type": "warning", "text": "Improvement Needed: Consider additional support or intervention programs."})
    if "Progress_Percent" in df.columns:
        if df["Progress_Percent"].mean() >= 80:
            insights.append({"type": "success", "text": "High Engagement: Students are actively progressing through courses."})
        else:
            insights.append({"type": "warning", "text": "Low Engagement: Encourage students to increase course participation."})
    if "Sentiment" in df.columns:
        pos_pct = (df["Sentiment"] == "Positive").sum() / len(df) * 100
        if pos_pct >= 70:
            insights.append({"type": "success", "text": "Positive Feedback: Students are satisfied with their learning experience."})
        elif pos_pct >= 50:
            insights.append({"type": "info", "text": "Mixed Feedback: Some students are satisfied while others need support."})
        else:
            insights.append({"type": "warning", "text": "Negative Feedback: Address student concerns and improve course quality."})

    return {
        "metrics": metrics,
        "platform_stats": platform_stats,
        "score_histogram": score_hist,
        "progress_histogram": progress_hist,
        "sentiment_data": sentiment_data,
        "sentiment_by_platform": sentiment_platform,
        "top_students": top_students,
        "course_performance": course_perf,
        "performance_levels": perf_levels,
        "insights": insights,
    }


@app.get("/api/ml-models/{session_id}")
def get_ml_models(session_id: str):
    """Train and compare ML models, return metrics and feature importance."""
    df = _get_session(session_id)
    result = _prepare_ml_data(df)
    if result is None:
        raise HTTPException(status_code=400, detail="Not enough features available for ML models.")

    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, features = result

    models_config = {
        "Random Forest": (RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42), X_train, X_test),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, learning_rate=0.1, random_state=42), X_train, X_test),
        "Ridge Regression": (Ridge(alpha=1.0), X_train_sc, X_test_sc),
    }
    if XGBOOST_AVAILABLE:
        models_config["XGBoost"] = (xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, min_child_weight=3, random_state=42, verbosity=0), X_train, X_test)

    results = []
    feature_importance = []
    cv_results = []

    for name, (model, Xtr, Xte) in models_config.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        r2 = float(r2_score(y_test, pred))
        mse = float(mean_squared_error(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))
        max_err = float(np.max(np.abs(y_test.values - pred)))

        mask = y_test.values != 0
        mape = float(np.abs((y_test.values[mask] - pred[mask]) / y_test.values[mask]).mean() * 100) if mask.any() else 0.0

        results.append({"model": name, "r2": round(r2, 4), "mse": round(mse, 4), "rmse": round(np.sqrt(mse), 4), "mae": round(mae, 4), "mape": round(mape, 4), "max_error": round(max_err, 4)})

        # CV
        cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2")
        cv_results.append({"model": name, "mean_cv_r2": round(float(cv_scores.mean()), 4), "std_cv_r2": round(float(cv_scores.std()), 4), "min_cv_r2": round(float(cv_scores.min()), 4), "max_cv_r2": round(float(cv_scores.max()), 4)})

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            importances = np.zeros(len(features))
        total = importances.sum()
        if total > 0:
            importances = importances / total
        for feat, imp in zip(features, importances):
            feature_importance.append({"model": name, "feature": feat, "importance": round(float(imp), 4)})

    results_sorted = sorted(results, key=lambda x: x["r2"], reverse=True)
    best = results_sorted[0]["model"]

    return {
        "results": results_sorted,
        "best_model": best,
        "cv_results": cv_results,
        "feature_importance": feature_importance,
        "features": features,
        "xgboost_available": XGBOOST_AVAILABLE,
    }


@app.get("/api/at-risk/{session_id}")
def get_at_risk(session_id: str, threshold: int = Query(default=50, ge=0, le=100)):
    """Return all students with risk scores, filtered by threshold."""
    df = _get_session(session_id)
    if "Student_ID" not in df.columns:
        raise HTTPException(status_code=400, detail="Student_ID column not found.")

    risk_df = _calculate_risk_scores(df)
    all_students = _safe_records(risk_df.sort_values("Risk_Score", ascending=False))

    at_risk = risk_df[risk_df["Risk_Score"] >= threshold].sort_values("Risk_Score", ascending=False)

    # Risk distribution histogram
    counts, edges = np.histogram(risk_df["Risk_Score"].dropna(), bins=20, range=(0, 100))
    risk_hist = [{"range": f"{edges[i]:.0f}-{edges[i+1]:.0f}", "count": int(counts[i]), "threshold_line": threshold} for i in range(len(counts))]

    # Risk level counts
    risk_level_counts = risk_df["Risk_Level"].value_counts().reset_index()
    risk_level_counts.columns = ["risk_level", "count"]

    # At-risk level counts
    at_risk_level_counts = at_risk["Risk_Level"].value_counts().reset_index()
    at_risk_level_counts.columns = ["risk_level", "count"]

    # Common risk factors
    all_reasons = []
    for reasons in at_risk["Risk_Reasons"].str.split("; "):
        if isinstance(reasons, list):
            all_reasons.extend([r for r in reasons if r])
    reason_counts = pd.Series(all_reasons).value_counts().reset_index()
    reason_counts.columns = ["factor", "count"]

    # Stats
    stats = {
        "total": len(risk_df),
        "at_risk_count": len(at_risk),
        "at_risk_pct": round(len(at_risk) / len(risk_df) * 100, 1) if len(risk_df) > 0 else 0,
        "safe_count": len(risk_df) - len(at_risk),
        "avg_risk_score": round(float(at_risk["Risk_Score"].mean()), 1) if len(at_risk) > 0 else 0,
        "max_risk_score": round(float(at_risk["Risk_Score"].max()), 1) if len(at_risk) > 0 else 0,
        "median_risk_score": round(float(at_risk["Risk_Score"].median()), 1) if len(at_risk) > 0 else 0,
        "critical_count": int((at_risk["Risk_Level"] == "Critical").sum()) if len(at_risk) > 0 else 0,
        "low_progress_count": int((at_risk["Avg_Progress"] < 50).sum()) if "Avg_Progress" in at_risk.columns and len(at_risk) > 0 else 0,
    }

    return {
        "threshold": threshold,
        "stats": stats,
        "at_risk_students": _safe_records(at_risk),
        "all_students": all_students,
        "risk_histogram": risk_hist,
        "risk_level_counts": _safe_records(risk_level_counts),
        "at_risk_level_counts": _safe_records(at_risk_level_counts),
        "common_risk_factors": _safe_records(reason_counts),
    }


@app.get("/api/student/{session_id}/{student_id}/recommendations")
def get_student_recommendations(session_id: str, student_id: str):
    """Return personalized course recommendations for a student."""
    df = _get_session(session_id)
    if "Student_ID" not in df.columns or "Course_ID" not in df.columns:
        raise HTTPException(status_code=400, detail="Required columns not found.")

    student_data = df[df["Student_ID"].astype(str) == str(student_id)]
    if student_data.empty:
        raise HTTPException(status_code=404, detail="Student not found.")

    student_avg_score = float(student_data["Score"].mean()) if "Score" in student_data.columns else 50
    student_avg_rating = float(student_data["Course_Rating"].mean()) if "Course_Rating" in student_data.columns else 3
    student_courses = set(student_data["Course_ID"].unique())

    if "Course_Name" in df.columns:
        all_courses = df[["Course_ID", "Course_Name"]].drop_duplicates()
    else:
        all_courses = df[["Course_ID"]].drop_duplicates()
        all_courses["Course_Name"] = all_courses["Course_ID"].apply(lambda x: f"Course {x}")

    available = all_courses[~all_courses["Course_ID"].isin(student_courses)]
    recommendations = []

    for _, row in available.iterrows():
        cid = row["Course_ID"]
        cname = row.get("Course_Name", f"Course {cid}")
        cdata = df[df["Course_ID"] == cid]
        c_avg_score = float(cdata["Score"].mean()) if "Score" in cdata.columns else 50
        c_avg_rating = float(cdata["Course_Rating"].mean()) if "Course_Rating" in cdata.columns else 3

        score_sim = 1 - abs(student_avg_score - c_avg_score) / 100
        rating_sim = 1 - abs(student_avg_rating - c_avg_rating) / 5 if c_avg_rating > 0 else 0.5
        rec_score = (score_sim * 0.6 + rating_sim * 0.4) * 100
        recommendations.append({
            "course_id": str(cid),
            "course_name": str(cname),
            "recommendation_score": round(rec_score, 1),
            "avg_course_score": round(c_avg_score, 1),
            "avg_course_rating": round(c_avg_rating, 2),
            "enrollments": len(cdata),
        })

    recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)

    # Student info
    student_name = student_data["Student_Name"].iloc[0] if "Student_Name" in student_data.columns else f"Student {student_id}"
    enrolled = []
    for cid in list(student_courses)[:10]:
        cdata = df[df["Course_ID"] == cid]
        cname = ""
        if "Course_Name" in cdata.columns and not cdata.empty:
            cname = str(cdata["Course_Name"].iloc[0])
        avg_s = float(cdata["Score"].mean()) if "Score" in cdata.columns else 0
        enrolled.append({"course_id": str(cid), "course_name": cname or f"Course {cid}", "score": round(avg_s, 1)})

    return {
        "student_id": student_id,
        "student_name": str(student_name),
        "avg_score": round(student_avg_score, 1),
        "avg_rating": round(student_avg_rating, 2),
        "enrolled_courses": enrolled,
        "recommendations": recommendations[:15],
    }


@app.get("/api/students/{session_id}")
def get_students(session_id: str):
    """Return list of all students."""
    df = _get_session(session_id)
    if "Student_ID" not in df.columns:
        return {"students": []}
    cols = [c for c in ["Student_ID", "Student_Name"] if c in df.columns]
    students = df[cols].drop_duplicates().reset_index(drop=True)
    return {"students": _safe_records(students)}


@app.get("/api/courses/{session_id}")
def get_courses(session_id: str):
    """Return list of all courses."""
    df = _get_session(session_id)
    if "Course_ID" not in df.columns:
        return {"courses": []}
    cols = [c for c in ["Course_ID", "Course_Name"] if c in df.columns]
    courses = df[cols].drop_duplicates().reset_index(drop=True)
    return {"courses": _safe_records(courses)}


@app.get("/api/report/{session_id}")
def get_report(session_id: str, report_type: str = Query(...), target_id: str = Query(...)):
    """Generate a text report for a student or course."""
    df = _get_session(session_id)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if report_type == "student":
        student_df = df[df["Student_ID"].astype(str) == str(target_id)]
        if student_df.empty:
            raise HTTPException(status_code=404, detail="Student not found.")

        name = student_df["Student_Name"].iloc[0] if "Student_Name" in student_df.columns else target_id
        avg_score = round(float(student_df["Score"].mean()), 2) if "Score" in student_df.columns else "N/A"
        avg_progress = round(float(student_df["Progress_Percent"].mean()), 1) if "Progress_Percent" in student_df.columns else "N/A"
        avg_rating = round(float(student_df["Course_Rating"].mean()), 2) if "Course_Rating" in student_df.columns else "N/A"
        total_courses = int(student_df["Course_ID"].nunique()) if "Course_ID" in student_df.columns else 0
        sentiment = student_df["Sentiment"].mode().iloc[0] if "Sentiment" in student_df.columns else "N/A"

        score_grade = "Excellent" if isinstance(avg_score, float) and avg_score >= 85 else ("Good" if isinstance(avg_score, float) and avg_score >= 70 else ("Average" if isinstance(avg_score, float) and avg_score >= 55 else "Below Average"))

        lines = [
            f"# Student Performance Report",
            f"**Generated:** {now}",
            f"",
            f"## Student Profile",
            f"- **Name:** {name}",
            f"- **Student ID:** {target_id}",
            f"- **Total Courses Enrolled:** {total_courses}",
            f"",
            f"## Academic Performance",
            f"- **Average Score:** {avg_score}",
            f"- **Performance Grade:** {score_grade}",
            f"- **Average Progress:** {avg_progress}%",
            f"- **Average Course Rating Given:** {avg_rating}",
            f"- **Overall Sentiment:** {sentiment}",
            f"",
            f"## Course Breakdown",
        ]
        if "Course_ID" in student_df.columns:
            for _, row in student_df.iterrows():
                cname = row.get("Course_Name", f"Course {row['Course_ID']}")
                score = round(float(row["Score"]), 1) if "Score" in student_df.columns else "N/A"
                progress = round(float(row["Progress_Percent"]), 1) if "Progress_Percent" in student_df.columns else "N/A"
                lines.append(f"- **{cname}** — Score: {score}, Progress: {progress}%")

        lines += ["", "## Recommendations"]
        if isinstance(avg_score, float) and avg_score < 65:
            lines.append("- Enroll in tutoring sessions to improve academic performance")
        if isinstance(avg_progress, float) and avg_progress < 60:
            lines.append("- Increase course engagement and set weekly progress goals")
        if sentiment == "Negative":
            lines.append("- Schedule a counseling session to address learning challenges")
        lines.append("- Continue to monitor progress and seek help when needed")

        report_text = "\n".join(lines)
        summary = {
            "Metric": ["Average Score", "Average Progress", "Total Courses", "Sentiment"],
            "Value": [avg_score, avg_progress, total_courses, sentiment],
        }

    elif report_type == "course":
        course_df = df[df["Course_ID"].astype(str) == str(target_id)]
        if course_df.empty:
            raise HTTPException(status_code=404, detail="Course not found.")

        cname = course_df["Course_Name"].iloc[0] if "Course_Name" in course_df.columns else f"Course {target_id}"
        avg_score = round(float(course_df["Score"].mean()), 2) if "Score" in course_df.columns else "N/A"
        avg_rating = round(float(course_df["Course_Rating"].mean()), 2) if "Course_Rating" in course_df.columns else "N/A"
        avg_progress = round(float(course_df["Progress_Percent"].mean()), 1) if "Progress_Percent" in course_df.columns else "N/A"
        total_enrolled = len(course_df)

        lines = [
            f"# Course Analysis Report",
            f"**Generated:** {now}",
            f"",
            f"## Course Profile",
            f"- **Course Name:** {cname}",
            f"- **Course ID:** {target_id}",
            f"- **Total Students Enrolled:** {total_enrolled}",
            f"",
            f"## Performance Metrics",
            f"- **Average Score:** {avg_score}",
            f"- **Average Rating:** {avg_rating}/5",
            f"- **Average Progress:** {avg_progress}%",
            f"",
            f"## Insights & Recommendations",
        ]
        if isinstance(avg_score, float) and avg_score < 65:
            lines.append("- Course content may be too difficult; consider adding supplemental materials")
        if isinstance(avg_rating, float) and avg_rating < 3.5:
            lines.append("- Review course quality and gather student feedback for improvements")
        if isinstance(avg_progress, float) and avg_progress < 60:
            lines.append("- Low completion rate; consider restructuring course milestones")
        lines.append("- Regular monitoring of student performance is recommended")

        report_text = "\n".join(lines)
        summary = {
            "Metric": ["Average Score", "Average Rating", "Average Progress", "Total Enrolled"],
            "Value": [avg_score, avg_rating, avg_progress, total_enrolled],
        }
    else:
        raise HTTPException(status_code=400, detail="report_type must be 'student' or 'course'")

    return {"report": report_text, "summary": summary}


@app.get("/api/export/{session_id}")
def export_csv(session_id: str):
    """Export the full processed dataset as CSV."""
    df = _get_session(session_id)
    csv_content = df.to_csv(index=False)
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=student_course_data.csv"},
    )
