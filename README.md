# 🛡️ Credit-Card Fraud Risk Analyzer

**Live demo:**[ Fraud-Risk-Analyzer ](https://fraud-risk-analyzer.streamlit.app/)

An end-to-end machine-learning project: exploratory analysis → preprocessing → model comparison →
deployed interactive web app.

## What it does

- Scores a single card transaction and returns a **fraud risk score** with a flag/no-flag decision
- Lets you move the **decision threshold** to see the precision/recall trade-off
- **What-if analysis** chart: how the score changes as one feature varies
- **Batch scoring**: upload a CSV, download scored results

## Data & method

- Kaggle credit-card-fraud dataset: 1,000,000 transactions, 7 features, 8.7 % fraud
- Outlier capping, standard scaling, **SMOTE on the training split only**
- Models compared: Logistic Regression, SVM, Random Forest, XGBoost, stacked ensemble
- Evaluated on a 200,000-row hold-out set using fraud-class precision / recall / F1

| Model | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---|---|
| Logistic Regression | 0.838 | 0.700 | 0.763 |
| Random Forest + SMOTE | 0.974 | 0.999 | 0.986 |
| XGBoost + SMOTE | 0.953 | 0.995 | 0.974 |
| Stacked ensemble + SMOTE | 0.974 | 0.999 | 0.986 |

## Run locally

```bash
python -m venv .venv && .venv\Scripts\activate   # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Project structure

```
app/            Streamlit UI, preprocessing, model loading
artifacts/      Saved model, scaler and config
sample_data/    Example CSV for batch scoring
docs/           Detailed guides
```

## Limitations

The dataset appears synthetic and highly separable, so scores are optimistic compared with real-world
fraud detection. SMOTE changes class balance, so outputs are risk scores rather than calibrated
probabilities. This is a portfolio demo, not a production system.
