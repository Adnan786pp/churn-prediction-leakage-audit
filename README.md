# Telco Customer Churn — Data Leakage Audit

This capstone demonstrates how data leakage can make a churn
prediction model appear perfect and how to correct the pipeline.

## Files
- `leaky_pipeline.py`: deliberately flawed demonstration
- `fixed_pipeline.py`: corrected machine learning pipeline
- `LEAKAGE_AUDIT.md`: bug explanations and metric comparison
- `requirements.txt`: dependencies

## Dataset
Download the Telco Customer Churn CSV from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this directory.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python leaky_pipeline.py
python fixed_pipeline.py
```

The leaky pipeline intentionally contains invalid practices and must
not be used for real-world predictions. Compare it with the fixed
pipeline to understand the effects of leakage.
