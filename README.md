# Fintech Loan Approval Classifier

A machine learning baseline that predicts the status of a loan application (`approved`, `pending`, or `rejected`) from customer profiles, account activity, employment history, and transaction records.

## Overview

The dataset contains 1,000 customer records with nested JSON-like fields (employment history and flagged transactions). `main.py` parses and flattens these fields into tabular features, engineers a few financial ratios, and trains a gradient-boosted tree classifier.

## Dataset

`sampleDataset.csv` (34 columns) combines four logical tables:

| Prefix | Contents |
| --- | --- |
| `Transaction Description` | List of flagged transactions per customer |
| `Customer Profile.*` | Age, occupation, income level, education, marital status, region, employment history |
| `Account Activity.*` | Balances, deposits, withdrawals, transfers, investments |
| `Loan Application Summary.*` | Loan amount, term, interest rate, employment status, and the target `loan_status` |
| `Time Series Data.*` | Monthly transactions and repayments |

## Feature engineering

- **Employment history** (`parse_employment`): average income, total years worked, and number of jobs held.
- **Flagged transactions** (`parse_transactions`): count and total amount of flagged transactions.
- **Income fallback**: missing `income_level` values are filled from average historical employment income.
- **Financial ratios**:
  - `dti_ratio` — loan amount vs. annualized income
  - `liquidity_buffer` — account balance vs. loan amount
  - `net_cash_flow` — deposits minus withdrawals

## Model

Scikit-learn `Pipeline`:

1. `ColumnTransformer`
   - Numeric: mean imputation
   - Categorical: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)
2. `GradientBoostingClassifier`

The target is label-encoded and the train/test split is stratified (80/20, `random_state=42`). The script prints a classification report and macro one-vs-rest ROC AUC.

## Current results

```
              precision    recall  f1-score   support

    approved       0.98      0.92      0.95        89
     pending       0.51      0.51      0.51        57
    rejected       0.51      0.56      0.53        54

    accuracy                           0.70       200
   macro avg       0.66      0.66      0.66       200
weighted avg       0.72      0.70      0.71       200

ROC AUC (macro, OVR): 0.8591
```

`approved` is classified reliably; `pending` and `rejected` are the main improvement targets.

## Getting started

Requires Python 3.10+.

```bash
pip install pandas numpy scikit-learn
python main.py
```

Tested with `pandas==3.0.5`, `scikit-learn==1.8.0`.

## Project structure

```
fintech_algorithm/
├── main.py            # Data parsing, feature engineering, training, evaluation
├── sampleDataset.csv  # Input data (1,000 records)
└── README.md
```

## Possible next steps

- Add class weights or resampling to improve `pending`/`rejected` recall.
- Tune hyperparameters via cross-validation.
- Expand evaluation with per-class ROC AUC and a confusion matrix.
- Persist the trained pipeline (e.g., `joblib`) for reuse.

## License

MIT
