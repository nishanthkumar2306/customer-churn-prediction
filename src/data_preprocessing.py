"""Data loading, cleaning and the reusable scikit-learn preprocessing pipeline.

The SAME functions are used for training and for prediction, so the model
always sees data prepared in exactly the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "telco_clean.csv"

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Columns a CSV MUST contain. Other feature columns may be missing; they are
# filled with the most common training value and a warning is shown.
REQUIRED_COLUMNS = ["tenure", "MonthlyCharges", "Contract"]

_YES_NO_MAP = {"0": "No", "0.0": "No", "no": "No", "false": "No",
               "1": "Yes", "1.0": "Yes", "yes": "Yes", "true": "Yes"}


def load_raw_data(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw Telco CSV. Raises a clear error if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """Clean a dataframe so it matches what the model expects.

    Returns (X, warnings). X contains exactly the FEATURES columns.
    Raises ValueError if required columns are missing or there are no rows.
    Works for both the training data and user-supplied data.
    """
    if df is None or len(df) == 0:
        raise ValueError("The data contains no rows.")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    warnings = []

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required column(s): {', '.join(missing_required)}")

    missing_optional = [c for c in FEATURES if c not in df.columns and c not in REQUIRED_COLUMNS]
    if missing_optional:
        warnings.append(
            "Missing optional column(s) filled with the most common training value: "
            + ", ".join(missing_optional)
        )
        for col in missing_optional:
            df[col] = np.nan

    # Numeric columns: non-numbers (and blanks like " ") become NaN, negatives are invalid
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    # If total charges is unknown, estimate it as tenure x monthly charges
    estimate = df["tenure"] * df["MonthlyCharges"]
    df["TotalCharges"] = df["TotalCharges"].fillna(estimate)

    # Categorical columns: trim spaces, blanks become NaN (imputed later)
    for col in CATEGORICAL_FEATURES:
        s = df[col].astype("string").str.strip()
        s = s.replace("", pd.NA)
        if col == "SeniorCitizen":  # dataset uses 0/1, humans use Yes/No
            s = s.str.lower().map(_YES_NO_MAP)
        df[col] = s.astype(object).where(s.notna(), np.nan)

    n_missing = int(df[FEATURES].isna().any(axis=1).sum())
    if n_missing:
        warnings.append(f"{n_missing} row(s) had missing/invalid values that were filled automatically.")
    return df[FEATURES], warnings


def clean_training_data(raw: pd.DataFrame):
    """Return (X, y, ids) for training. y is 1 for churn, 0 for stay."""
    raw = raw.copy()
    raw = raw.drop_duplicates(subset=ID_COL)
    raw = raw[raw[TARGET].isin(["Yes", "No"])]
    X, _ = prepare_features(raw)
    y = (raw[TARGET] == "Yes").astype(int)
    return X, y, raw[ID_COL]


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numbers, impute + one-hot encode categories."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
