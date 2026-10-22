"""FastAPI backend.   Run:  uvicorn backend.main:app --reload --port 8000

No CORS middleware: the Streamlit dashboard calls the Python code directly,
so no browser cross-origin requests are made to this API.
"""
import json
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.schemas import BatchResponse, CustomerInput, PredictionResponse
from backend.services import prediction_service as service
from src.predict import LOW_MAX, MEDIUM_MAX, load_model
from src.train_model import METRICS_PATH

logger = logging.getLogger("churn_api")
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0",
              description="Predicts churn probability, risk level, influencing factors and retention suggestions.")


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model/info")
def model_info():
    try:
        bundle = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    info = {"model_name": bundle["model_name"], "features": bundle["features"],
            "risk_thresholds": {"low_below": LOW_MAX, "high_from": MEDIUM_MAX}}
    if METRICS_PATH.exists():
        report = json.loads(METRICS_PATH.read_text())
        info["dataset"] = report["dataset"]
        info["evaluation"] = report["models"][bundle["model_name"]]
    return info


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        return service.predict_one(customer.to_model_row())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    # read at most limit+1 bytes so huge uploads are not loaded into memory
    content = await file.read(service.MAX_UPLOAD_BYTES + 1)
    try:
        df = service.read_csv_bytes(content)
        flat, warnings = service.predict_batch(df)
    except service.InvalidUpload as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed.")
    summary = {"risk_counts": flat["risk_level"].value_counts().to_dict(),
               "predicted_churn": int((flat["prediction"] == "Likely to Churn").sum())}
    records = json.loads(flat.to_json(orient="records"))
    return {"n_customers": len(flat), "warnings": warnings, "summary": summary, "results": records}
