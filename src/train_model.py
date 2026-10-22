"""Train, compare and save the churn model.   Run:  python src/train_model.py"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (PROCESSED_PATH, ROOT, TARGET, build_preprocessor,
                                    clean_training_data, load_raw_data)

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5


def make_models():
    # class_weight="balanced" because only ~27% of customers churn
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    raw = load_raw_data()
    X, y, ids = clean_training_data(raw)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    X.assign(customerID=ids.values, **{TARGET: y.values}).to_csv(PROCESSED_PATH, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  Churn rate: {y.mean():.1%}")

    results, fitted = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in make_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        # Model selection uses cross-validation on TRAIN data only (test set stays untouched)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        metrics["cv_roc_auc_mean"] = float(cv_auc.mean())
        metrics["cv_roc_auc_std"] = float(cv_auc.std())
        results[name], fitted[name] = metrics, pipe
        print(f"\n{name}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc_mean"]:
            print(f"  {k:16s} {metrics[k]:.4f}")
        print("  confusion:", metrics["confusion_matrix"])

    best = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    print(f"\nSelected model: {best} (highest cross-validated ROC-AUC)")

    # Reference values used by the explanation method (see src/explainability.py)
    reference = {}
    for col in X_train.columns:
        if X_train[col].dtype == object:
            freq = X_train[col].value_counts(normalize=True)
            reference[col] = {"values": freq.index.tolist(), "weights": freq.values.tolist()}
        else:
            reference[col] = {"values": X_train[col].quantile([.1, .3, .5, .7, .9]).tolist(),
                              "weights": [0.2] * 5}

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "model_name": best, "reference": reference,
                 "features": list(X.columns)}, MODEL_PATH)
    report = {
        "dataset": "IBM Telco Customer Churn", "n_rows": int(len(X)),
        "churn_rate": float(y.mean()), "test_size": 0.2, "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD, "selected_model": best, "models": results,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved model -> {MODEL_PATH}\nSaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
