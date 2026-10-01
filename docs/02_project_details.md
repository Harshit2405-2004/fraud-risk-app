# 02 · Project Details (know your own project cold)

Everything below is taken from your `RiskAnalysis.ipynb`. Interviewers will ask about decisions you
made — this is your cheat-sheet.

---

## 1. The problem

Predict whether a credit-card transaction is **fraudulent (1)** or **legitimate (0)** from seven
behavioural features. Binary classification on a heavily imbalanced dataset.

## 2. The data

- **Source:** Kaggle — `dhanushnarayananr/credit-card-fraud` (licence CC0), file `card_transdata.csv`
- **Size:** 1,000,000 rows × 8 columns, **no missing values**, all columns stored as floats
- **Target balance:** 912,597 legitimate (91.3 %) vs **87,403 fraud (8.7 %)**

| Feature | Type | Meaning |
|---|---|---|
| `distance_from_home` | numeric | Distance between the purchase location and the cardholder's home |
| `distance_from_last_transaction` | numeric | Distance from the previous transaction's location |
| `ratio_to_median_purchase_price` | numeric | This purchase amount ÷ the cardholder's median purchase (1.0 = typical) |
| `repeat_retailer` | 0/1 | Bought from this retailer before? |
| `used_chip` | 0/1 | Chip card used? |
| `used_pin_number` | 0/1 | PIN entered? |
| `online_order` | 0/1 | Online purchase? |
| `fraud` | 0/1 | **Target** |

*(The dataset doesn't state distance units; the app labels them "km" as an assumption — mention that
if asked.)*

## 3. The pipeline, step by step — and *why*

| # | Step in notebook | Why you did it |
|---|---|---|
| 1 | `isnull().sum()`, `info()`, `describe()`, `value_counts()` | Sanity checks; discovered imbalance |
| 2 | Correlation heatmap, count-plots, KDE plots by fraud/non-fraud | EDA: how each feature separates fraud |
| 3 | Box-plots → heavy right-skew/outliers (max distance ≈ 10,600; max ratio ≈ 268) | Outliers can distort scaling and linear/SVM models |
| 4 | **Clip** at 177 / 40 / 13 (≈98th, 98th, 99th percentiles) | Winsorising: keeps the rows, tames extremes |
| 5 | `train_test_split` 80/20, `random_state=42` | 800,000 train / 200,000 test |
| 6 | Baseline Random Forest on unscaled data | Trees don't need scaling; a quick benchmark |
| 7 | `StandardScaler` fitted on **train only**, 3 numeric columns | SVM & logistic regression are scale-sensitive; avoids test leakage |
| 8 | SVC and Logistic Regression | Classic baselines |
| 9 | **SMOTE on train only** → 730,040 + 730,040 rows | Balance classes so the model doesn't ignore fraud |
| 10 | Random Forest (150 trees), XGBoost (depth 5) on resampled data | Stronger models |
| 11 | **Stacking:** RF + XGBoost + SVC → Logistic Regression meta-learner | Combine strengths |

## 4. Results on the 200,000-row hold-out test set

Test set composition: 182,557 legitimate, 17,443 fraud (untouched by SMOTE).

| Model | Accuracy | Fraud precision | Fraud recall | Fraud F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9620 | 0.8375 | 0.6998 | 0.7625 |
| SVM | 0.9963 | 0.9679 | 0.9907 | 0.9792 |
| Random Forest (no SMOTE) | 0.9971 | 0.9763 | 0.9908 | 0.9835 |
| Random Forest + SMOTE | 0.9976 | 0.9738 | 0.9992 | 0.9863 |
| XGBoost + SMOTE | 0.9953 | 0.9529 | 0.9950 | 0.9735 |
| **Stacked ensemble + SMOTE** | **0.9976** | **0.9741** | **0.9990** | **0.9864** |

**How to read them (practice saying this aloud):**

- Logistic Regression is far worse → the fraud boundary is **non-linear**; tree-based models capture it.
- SMOTE barely changed accuracy but **raised fraud recall from 0.991 to 0.999** — fewer missed frauds.
- The stacked model is only marginally better than Random Forest + SMOTE (F1 0.9864 vs 0.9863).
  Being honest about that is a *strength*: the simpler model gives almost the same result.
- Approximate stacked-model confusion matrix (derived from the reported metrics): **~17 frauds missed
  and ~464 false alarms** out of 200,000 transactions.

## 5. Honest limitations (say these before the interviewer finds them)

1. **The data looks synthetic / very separable.** 99.8 % accuracy is far above what real-world fraud
   systems achieve. Real fraud data is noisier, drifts over time, and has a much smaller fraud rate
   (well under 1 %).
2. **Capping thresholds were computed on the whole dataset before the train/test split.** A tiny
   information leak; with 1M rows and fixed round-number caps its effect is negligible, but the
   cleaner approach is to compute quantiles on the training split only.
3. **SMOTE shifts probabilities.** The model was trained on a 50/50 world, so its outputs are
   **risk scores**, not calibrated "there's a 94 % chance this is fraud" values.
4. **No time component.** Random split, not chronological split; real systems validate on future data.
5. **No cost-based threshold.** The 0.5 threshold is a default; a bank would choose it from the cost of
   a missed fraud versus a false alarm. (The app's slider demonstrates this.)
6. **No cross-validation or hyper-parameter search** — a single split with mostly default settings.
7. **Feature meaning is limited** — distances have no stated unit; no card/merchant identifiers.

## 6. Find out which features matter (do this before the interview)

The notebook doesn't compute feature importance, and it's a very likely interview question. Run this
in the notebook (after `model` is trained):

```python
import pandas as pd
names = ['distance_from_home', 'distance_from_last_transaction',
         'ratio_to_median_purchase_price', 'repeat_retailer',
         'used_chip', 'used_pin_number', 'online_order']
imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
print(imp)
imp.plot(kind='barh')
```

Write down the top three features and a one-line reason each might matter. Don't guess in the interview
— use your actual numbers.

## 7. Ideas for "what would you do next?"

- Chronological train/validation/test split and **cross-validation**
- **Probability calibration** (Platt/isotonic) and a **cost-sensitive threshold**
- Compare **PR-AUC** (better than ROC-AUC under imbalance)
- **SHAP** explanations for each prediction in the app
- **Model monitoring / drift detection**, retraining schedule
- Real-time **API** (FastAPI) behind the model, Streamlit as just the front-end
