import pandas as pd
import pytest

from src.predict import load_model, predict_dataframe, predict_single, risk_level


def test_model_loads():
    bundle = load_model()
    assert "pipeline" in bundle and bundle["model_name"]


def test_prediction_and_probability_range(sample_customer):
    result = predict_single(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["prediction"] in ("Likely to Churn", "Likely to Stay")
    assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_high_risk_profile_scores_higher_than_low_risk_profile():
    risky = predict_single({"tenure": 1, "MonthlyCharges": 95, "Contract": "Month-to-month"})
    safe = predict_single({"tenure": 70, "MonthlyCharges": 40, "Contract": "Two year"})
    assert risky["churn_probability"] > safe["churn_probability"]


def test_batch_prediction_shape(sample_customer):
    df = pd.DataFrame([sample_customer, {"tenure": 60, "MonthlyCharges": 30, "Contract": "Two year"}])
    out, _ = predict_dataframe(df)
    assert len(out) == 2 and out["churn_probability"].between(0, 1).all()


@pytest.mark.parametrize("p,expected", [(0.20, "LOW"), (0.45, "MEDIUM"), (0.80, "HIGH"),
                                        (0.30, "MEDIUM"), (0.60, "HIGH"), (0.0, "LOW"), (1.0, "HIGH")])
def test_risk_levels(p, expected):
    assert risk_level(p) == expected


def test_factors_only_use_provided_fields():
    result = predict_single({"tenure": 2, "MonthlyCharges": 95, "Contract": "Month-to-month"})
    allowed = {"Tenure (months)", "Monthly charges", "Contract", "Total charges"}
    assert {f["feature"] for f in result["important_factors"]} <= allowed
