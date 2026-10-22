import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
GOOD_CSV = (b"tenure,MonthlyCharges,Contract\n2,95.4,Month-to-month\n60,30,Two year\n")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] and "roc_auc" in body["evaluation"]


def test_predict_minimal_input():
    r = client.post("/predict", json={"tenure": 5, "monthly_charges": 85.5, "contract": "Month-to-month"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert body["retention_suggestions"] and body["disclaimer"]


@pytest.mark.parametrize("payload", [
    {"tenure": -1, "monthly_charges": 10, "contract": "Month-to-month"},
    {"tenure": 5, "monthly_charges": 10, "contract": "Weekly"},
    {"monthly_charges": 10, "contract": "One year"},
])
def test_predict_rejects_invalid_input(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_valid_csv():
    r = client.post("/predict/batch", files={"file": ("c.csv", GOOD_CSV, "text/csv")})
    assert r.status_code == 200 and r.json()["n_customers"] == 2


def test_batch_missing_columns_gives_400():
    r = client.post("/predict/batch", files={"file": ("c.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Missing required" in r.json()["detail"]


def test_batch_wrong_extension_and_garbage():
    assert client.post("/predict/batch", files={"file": ("c.txt", b"x", "text/plain")}).status_code == 400
    assert client.post("/predict/batch", files={"file": ("c.csv", b"", "text/csv")}).status_code == 400
    assert client.post("/predict/batch", files={"file": ("c.csv", b"\x00\xff\xfe", "text/csv")}).status_code == 400
