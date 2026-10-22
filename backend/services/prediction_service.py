"""Thin service layer between the API and the ML code in src/."""
import io

import pandas as pd

from src.predict import predict_dataframe, predict_single, results_to_csv_frame
from src.recommendations import DISCLAIMER

MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_ROWS = 5000


class InvalidUpload(ValueError):
    """Raised for bad CSV files (turned into HTTP 400)."""


def predict_one(customer_row: dict) -> dict:
    result = predict_single(customer_row)
    result["disclaimer"] = DISCLAIMER
    return result


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    if len(content) == 0:
        raise InvalidUpload("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidUpload(f"File is too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise InvalidUpload("Could not read the file as a CSV.")
    if len(df) == 0:
        raise InvalidUpload("The CSV has no data rows.")
    if len(df) > MAX_ROWS:
        raise InvalidUpload(f"Too many rows ({len(df)}). Limit is {MAX_ROWS}.")
    return df


def predict_batch(df: pd.DataFrame):
    """Returns (flat results DataFrame, warnings). Raises InvalidUpload for bad data."""
    try:
        results, warnings = predict_dataframe(df)
    except ValueError as exc:
        raise InvalidUpload(str(exc))
    flat = results_to_csv_frame(results)
    if "customerID" in df.columns:
        flat.insert(0, "customerID", df["customerID"].astype(str).values)
    return flat, warnings
