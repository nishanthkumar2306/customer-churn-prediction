# Customer Churn Prediction, Risk Analysis & Retention Recommendation System

An end-to-end machine-learning project that predicts whether a telecom customer is likely to leave (churn),
estimates how risky that customer is, shows which factors influenced the model's prediction, suggests
retention actions, and presents everything in an interactive dashboard and a REST API.

## Problem statement
Winning a new customer usually costs more than keeping an existing one. If a business can spot customers
who are likely to leave, it can contact them early with a relevant offer. This project builds that workflow
on a public dataset.

## Objectives
- Train and compare ML models on real churn data using a clean scikit-learn pipeline
- Evaluate with metrics suited to imbalanced data (not accuracy alone)
- Convert probabilities to transparent risk levels
- Explain individual predictions and generate rule-based retention suggestions
- Serve predictions through FastAPI and a Streamlit dashboard (single + CSV batch)

## Features
- Single-customer prediction (probability, risk level, influencing factors, suggestions)
- Business dashboard: KPI cards + charts (churn by contract, tenure, charges, payment method, internet service, senior status, tech support, online security, risk distribution)
- High-risk customer list with filters, sorting and CSV download
- CSV batch upload with validation, results table and CSV download
- Customer analysis page (pick any customer in the dataset)
- Model information page (metrics, confusion matrix, limitations)
- FastAPI endpoints with Pydantic validation
- 33 automated pytest tests

## Tech stack
Python, pandas, scikit-learn (Pipeline + ColumnTransformer), FastAPI + Pydantic, Streamlit, Plotly, pytest.

## Architecture
```
Customer Data -> Preprocessing -> ML Model -> Prediction -> Risk Analysis -> Retention Recommendations -> Dashboard / API
```
The dashboard and API both call the same code in `src/`, so training and prediction use identical preprocessing.
The dashboard calls `src/` directly (it does not need the API to be running).

## Dataset
**IBM Telco Customer Churn**: 7,043 customers, 21 columns. Target: `Churn` (Yes/No, ~26.5% Yes).
Important columns: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `TechSupport`, `PaymentMethod`.
Source and download instructions: [data/README.md](data/README.md).

## Machine learning
- **Cleaning:** `TotalCharges` blanks (new customers) are estimated as tenure x monthly charges; invalid/negative numbers become missing; `SeniorCitizen` 0/1 is converted to No/Yes.
- **Preprocessing (one reusable pipeline):** numbers -> median imputation + scaling; categories -> most-frequent imputation + one-hot encoding (unknown categories are ignored safely).
- **Split:** 80% train / 20% test, stratified, `random_state=42`.
- **Models:** Logistic Regression and Random Forest, both with class weighting because churners are a minority.
- **Selection:** highest 5-fold cross-validated ROC-AUC on the *training* data (the test set is only used for the final report).
- **Decision threshold:** probability >= 0.5 is labelled "Likely to Churn".

### Metrics in plain language
- **Accuracy** – share of all predictions that are right (can look good even for a weak model when most customers stay).
- **Precision** – of customers flagged as churners, how many really churned.
- **Recall** – of customers who really churned, how many we caught.
- **F1** – balance between precision and recall.
- **ROC-AUC** – how well the model ranks churners above non-churners (0.5 = random, 1.0 = perfect).
- **Confusion matrix** – counts of correct and wrong predictions in each class.

### Results (test set, 1,409 customers; produced by `python src/train_model.py`)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression (selected)** | 0.738 | 0.504 | 0.783 | 0.614 | 0.842 | 0.846 |
| Random Forest | 0.768 | 0.546 | 0.741 | 0.629 | 0.842 | 0.844 |

Confusion matrix (Logistic Regression): TN 747, FP 288, FN 81, TP 293.
The two models are nearly tied on ROC-AUC; Logistic Regression was kept because it is simpler and catches slightly more churners (higher recall) at the cost of more false alarms. Full numbers: `models/metrics.json`.

## Risk analysis
| Churn probability | Risk level |
|---|---|
| below 30% | LOW |
| 30% to below 60% | MEDIUM |
| 60% and above | HIGH |

These thresholds are simple, easy-to-explain defaults, not tuned to any business cost.

## Explainability
SHAP was not used. Instead, a simple what-if method (`src/explainability.py`) replaces one feature at a time with typical training values and measures how the predicted probability changes. Features that raise the probability are shown as **important factors influencing the model prediction**.
Limitations: one feature at a time (interactions only partly captured), correlated features (e.g. monthly vs total charges) can give odd-looking effects, and none of this proves causation.

## Retention recommendations
Rule-based (`src/recommendations.py`), e.g. month-to-month contract -> discounted longer contract; monthly charges >= 80 -> plan review; no tech support -> support trial; tenure <= 12 months -> onboarding/loyalty offer. They are suggestions, not guarantees.

## Installation (Windows, VS Code terminal)
Requires Python 3.11 or newer.
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the project
```
python src/train_model.py                         (optional: a trained model is included)
uvicorn backend.main:app --reload --port 8000     (API docs: http://localhost:8000/docs)
streamlit run dashboard/app.py                    (dashboard: http://localhost:8501)
python -m pytest                                  (run tests)
```
Run the API and the dashboard in separate terminals; the dashboard works without the API.

## API
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service and model status |
| GET | `/model/info` | Model name, features, thresholds, evaluation |
| POST | `/predict` | Predict one customer (JSON) |
| POST | `/predict/batch` | Upload a CSV (max 5 MB, 5000 rows) |

Example:
```
POST /predict
{"tenure": 5, "monthly_charges": 85.5, "contract": "Month-to-month"}
```
Only `tenure`, `monthly_charges` and `contract` are required; missing optional fields are filled with the most common training value (predictions are more reliable when more fields are given).

## Safety
No secrets are used or stored; uploads are size/row limited and parsed only as CSV data; file paths are never taken from users; invalid input returns clear errors.

## Limitations
Predictions are probabilistic. Historical data may contain bias. Correlation does not imply causation. Recommendations are suggestions. Performance depends on dataset quality (one public, possibly synthetic-style dataset). Dashboard risk numbers are computed on data the model was trained on. Real retention decisions need business context.

## Screenshots
Add screenshots to a `screenshots/` folder and link them here: Dashboard, Predict Customer, Batch Prediction, Risk Analysis, Customer Analysis, Model Information.

## Future enhancements
Probability calibration and cost-based threshold tuning, SHAP explanations, gradient-boosted models, database storage, Docker deployment, user authentication.

## Author
Name: ______  |  GitHub: ______  |  LinkedIn: ______
