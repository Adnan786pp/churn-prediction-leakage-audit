# Data Leakage Audit — Telco Customer Churn

## Objective
Identify deliberate data leakage in a binary churn-prediction pipeline
and compare flawed and corrected evaluation.

## Bug 1: Pre-split preprocessing
The flawed pipeline fits imputation and scaling on the entire dataset
before splitting. This allows test-set information to influence
preprocessing. The corrected pipeline splits first and fits all
preprocessing through a scikit-learn Pipeline on training data only.

## Bug 2: Target-derived feature
`ChurnRiskScore` is constructed directly from the target `Churn`.
This reveals the answer to the model and creates artificially high
metrics. The feature is removed from valid predictors.

## Bug 3: Duplicate records across splits
Duplicate customer records may occur in both training and testing
sets, making evaluation less representative of unseen customers.
The corrected pipeline removes duplicate customer IDs before splitting.
If customers have multiple legitimate records, use group-based splitting.

## Bug 4: Post-outcome feature
`ChurnedAlready` directly encodes whether churn occurred. It is not
available when predicting future churn, so it is removed.

## Results

| Metric | Leaky model | Fixed model |
|---|---:|---:|
| Accuracy | 1.0000 | 0.8055 |
| Precision | 1.0000 | 0.6572 |
| Recall | 1.0000 | 0.5588 |
| F1-score | 1.0000 | 0.6040 |

The leaky model's perfect scores result from target-derived and
post-outcome features. The fixed model's scores are more credible for
this split, but do not guarantee production performance.

## Corrected workflow
1. Remove duplicate customer records.
2. Separate target and predictors.
3. Remove target-derived and post-outcome features.
4. Split using stratification.
5. Fit imputation, encoding, and scaling on training data only.
6. Train Logistic Regression and evaluate on held-out test data.

## Reproducibility
Place the original `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same
directory as the Python scripts before running them.
