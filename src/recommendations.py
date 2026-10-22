"""Rule-based retention suggestions, generated from the customer's actual features."""
import pandas as pd

HIGH_MONTHLY_CHARGES = 80.0   # roughly the top third of Telco customers
LOW_TENURE_MONTHS = 12

DISCLAIMER = ("Recommendations are rule-based suggestions intended to support "
              "customer-retention decisions. They do not guarantee that a customer will stay.")


def get_recommendations(customer) -> list:
    """customer: dict or pandas Series with Telco column names. Returns list of {action, reason}."""
    c = dict(customer)
    out = []

    def add(action, reason):
        out.append({"action": action, "reason": reason})

    if c.get("Contract") == "Month-to-month":
        add("Offer a discounted annual or two-year contract.",
            "Customer is on a month-to-month contract.")
    monthly = c.get("MonthlyCharges")
    if pd.notna(monthly) and monthly >= HIGH_MONTHLY_CHARGES:
        add("Provide a personalized pricing or plan review.",
            f"Monthly charges ({monthly:.2f}) are high.")
    if c.get("TechSupport") == "No":
        add("Offer technical support assistance or a free support trial.",
            "Customer has internet service but no tech support.")
    if c.get("OnlineSecurity") == "No":
        add("Offer an online security add-on trial.",
            "Customer has internet service but no online security.")
    tenure = c.get("tenure")
    if pd.notna(tenure) and tenure <= LOW_TENURE_MONTHS:
        add("Provide an onboarding check-in or a loyalty offer.",
            f"Customer is new (tenure {int(tenure)} months).")
    if c.get("PaymentMethod") == "Electronic check":
        add("Encourage automatic payment (card or bank transfer) with a small incentive.",
            "Customer pays by electronic check.")
    if c.get("SeniorCitizen") == "Yes" and c.get("TechSupport") == "No":
        add("Offer a dedicated, easy-access support contact.",
            "Senior customer without tech support.")

    if not out:
        add("No specific rule triggered; continue standard customer engagement.",
            "None of the retention rules matched this customer.")
    return out
