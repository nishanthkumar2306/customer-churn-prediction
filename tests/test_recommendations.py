from src.recommendations import get_recommendations


def actions(customer):
    return " ".join(r["action"] for r in get_recommendations(customer))


def test_month_to_month_gets_contract_offer():
    assert "annual or two-year contract" in actions({"Contract": "Month-to-month"})


def test_high_charges_get_plan_review():
    assert "pricing or plan review" in actions({"Contract": "Two year", "MonthlyCharges": 99})


def test_no_tech_support_gets_support_offer():
    assert "support" in actions({"Contract": "Two year", "TechSupport": "No"})


def test_low_tenure_gets_onboarding():
    assert "onboarding" in actions({"Contract": "Two year", "tenure": 3})


def test_multiple_risk_factors_give_multiple_recommendations():
    recs = get_recommendations({"Contract": "Month-to-month", "MonthlyCharges": 95, "tenure": 2,
                                "TechSupport": "No"})
    assert len(recs) >= 4


def test_safe_customer_gets_default_message():
    recs = get_recommendations({"Contract": "Two year", "MonthlyCharges": 30, "tenure": 60,
                                "TechSupport": "Yes", "OnlineSecurity": "Yes"})
    assert len(recs) == 1 and "No specific rule" in recs[0]["action"]
