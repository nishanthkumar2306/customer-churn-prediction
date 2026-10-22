# Dataset

**IBM Telco Customer Churn** – 7,043 customers of a fictional telecom company, 21 columns.
Target column: `Churn` (Yes/No). Identifier: `customerID`.

- Source used for this project: https://github.com/IBM/telco-customer-churn-on-icp4d (file `data/Telco-Customer-Churn.csv`)
- Also available on Kaggle: "Telco Customer Churn" by BlastChar.

The file is already at `data/raw/Telco-Customer-Churn.csv`. If it is missing, download it from the
link above and save it with exactly that name in `data/raw/`.

`data/processed/telco_clean.csv` is created by `python src/train_model.py`.
`data/sample_batch.csv` is the first 25 customers (without the `Churn` column) for testing CSV upload.
