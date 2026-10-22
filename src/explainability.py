"""Simple, model-agnostic explanations ("what pushed this prediction up or down?").

Method (marginal-effect / what-if analysis):
  For each feature, replace the customer's value with typical values from the
  training data (weighted by how common they are) and average the model's
  churn probability. The difference
        effect = P(customer) - average P(with that feature set to typical values)
  tells us how much that feature's value moves THIS prediction up (+) or down (-).

Limitations: effects are computed one feature at a time, so interactions between
features are only partly captured, and they describe the MODEL, not real-world causes.
"""
import numpy as np
import pandas as pd

DISPLAY_NAMES = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner",
    "Dependents": "Dependents", "tenure": "Tenure (months)", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}
MIN_EFFECT = 0.01  # ignore effects smaller than 1 percentage point


def compute_effects(pipeline, reference: dict, X: pd.DataFrame, base_proba=None) -> pd.DataFrame:
    """Return a DataFrame (same index as X) with one column of effects per feature."""
    X = X.reset_index(drop=True)
    if base_proba is None:
        base_proba = pipeline.predict_proba(X)[:, 1]
    effects = {}
    for col in X.columns:
        ref = reference[col]
        avg = np.zeros(len(X))
        for value, weight in zip(ref["values"], ref["weights"]):
            X_mod = X.copy()
            X_mod[col] = value
            avg += weight * pipeline.predict_proba(X_mod)[:, 1]
        effects[col] = base_proba - avg
    return pd.DataFrame(effects)


def _format_value(col, value):
    if col in ("MonthlyCharges", "TotalCharges") and pd.notna(value):
        return f"{value:.2f}"
    if col == "tenure" and pd.notna(value):
        return f"{int(value)}"
    return str(value)


def top_factors(x_row: pd.Series, effects_row: pd.Series, n: int = 3, direction: str = "up") -> list:
    """Top n features pushing the prediction up (risk) or down (protective)."""
    effects_row = effects_row[x_row[effects_row.index].notna().values]  # skip fields the user did not provide
    s = effects_row[effects_row > MIN_EFFECT] if direction == "up" else effects_row[effects_row < -MIN_EFFECT]
    s = s.sort_values(ascending=(direction != "up")).head(n)
    return [{
        "feature": DISPLAY_NAMES.get(col, col),
        "value": _format_value(col, x_row[col]),
        "effect_on_probability": round(float(eff), 4),
        "direction": "increases churn probability" if eff > 0 else "decreases churn probability",
    } for col, eff in s.items()]
