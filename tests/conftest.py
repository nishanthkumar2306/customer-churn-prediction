import pytest

from src.predict import DEFAULT_MODEL_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """Train the model once if it has not been trained yet."""
    if not DEFAULT_MODEL_PATH.exists():
        from src.train_model import main
        main()


@pytest.fixture
def sample_customer():
    return {"tenure": 2, "MonthlyCharges": 95.4, "Contract": "Month-to-month",
            "InternetService": "Fiber optic", "TechSupport": "No",
            "PaymentMethod": "Electronic check"}
