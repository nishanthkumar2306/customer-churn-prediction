"""Load the saved model and turn customer data into predictions."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import prepare_features
from src.explainability import compute_effects, top_factors
from src.recommendations import get_recommendations

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

# Risk thresholds (documented in README)
LOW_MAX = 0.30      # probability < 0.30  -> LOW
MEDIUM_MAX = 0.60   # 0.30 <= p < 0.60    -> MEDIUM, p >= 0.60 -> HIGH
DECISION_THRESHOLD = 0.5  # p >= 0.5 -> "Likely to Churn"


def risk_level(probability: float) -> str:
    if probability < LOW_MAX:
        return "LOW"
    if probability < MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.getenv("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run: python src/train_model.py")
    return joblib.load(path)


def predict_dataframe(df: pd.DataFrame, explain: bool = True, top_n: int = 3):
    """Predict for many customers. Returns (results_df, warnings).

    results_df has the cleaned features plus: churn_probability, prediction, risk_level,
    risk_factors, retention_suggestions. Factors are only computed for MEDIUM/HIGH rows
    (they are not very useful for low-risk customers and it keeps large files fast).
    """
    bundle = load_model()
    X, warnings = prepare_features(df)
    X = X.reset_index(drop=True)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]

    out = X.copy()
    out["churn_probability"] = proba.round(4)
    out["prediction"] = ["Likely to Churn" if p >= DECISION_THRESHOLD else "Likely to Stay" for p in proba]
    out["risk_level"] = [risk_level(p) for p in proba]

    factors = [[] for _ in range(len(X))]
    protective = [[] for _ in range(len(X))]
    if explain:
        idx = [i for i, p in enumerate(proba) if p >= LOW_MAX] if len(X) > 1 else list(range(len(X)))
        if idx:
            sub = X.iloc[idx].reset_index(drop=True)
            effects = compute_effects(bundle["pipeline"], bundle["reference"], sub, proba[idx])
            for j, i in enumerate(idx):
                factors[i] = top_factors(sub.iloc[j], effects.iloc[j], top_n, "up")
                protective[i] = top_factors(sub.iloc[j], effects.iloc[j], top_n, "down")
    out["risk_factors"] = factors
    out["protective_factors"] = protective
    out["retention_suggestions"] = [get_recommendations(r) for _, r in X.iterrows()]
    return out, warnings


def predict_single(customer: dict) -> dict:
    """customer: dict using Telco column names. Returns one result dict."""
    res, _ = predict_dataframe(pd.DataFrame([customer]))
    row = res.iloc[0]
    return {
        "prediction": row["prediction"],
        "churn_probability": float(row["churn_probability"]),
        "risk_level": row["risk_level"],
        "important_factors": row["risk_factors"],
        "protective_factors": row["protective_factors"],
        "retention_suggestions": row["retention_suggestions"],
    }


def results_to_csv_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Flatten list columns into text so the results can be saved as CSV."""
    flat = results.drop(columns=["risk_factors", "protective_factors", "retention_suggestions"]).copy()
    flat["important_factors"] = [
        "; ".join(f"{f['feature']}={f['value']}" for f in fs) for fs in results["risk_factors"]]
    flat["retention_suggestions"] = [
        " | ".join(s["action"] for s in ss) for ss in results["retention_suggestions"]]
    return flat
