# 00 · Start Here — From Notebook to Live Streamlit App

You have a trained fraud-detection model in `RiskAnalysis.ipynb`. This guide set turns it into a
**public web app an interviewer can open with one link** and play with.

## What you are building

A **Credit-Card Fraud Risk Analyzer** with three tabs:

| Tab | What the interviewer can do |
|---|---|
| 🔍 Score one transaction | Enter transaction details (or pick an example), see fraud probability, a flag/no-flag decision, and a what-if curve |
| 📄 Batch scoring | Upload a CSV, get every row scored, download results |
| 📊 Model & project | See the model comparison table, methodology, and honest limitations |

A sidebar **decision-threshold slider** lets them see the precision/recall trade-off live.

## ⚠️ One thing to know first

Your notebook **trains** the models but **never saves them** (there is no `joblib.dump`, `pickle`, or
`save_model` anywhere in it). You said you have the trained weights — if you still have the notebook
session open, great: Step 1 below saves them. If you closed it, you will need to re-run the notebook
once (it trains in a reasonable time for the Random Forest; the stacked model takes much longer).

You also need to save **two more things** the app depends on, which are easy to forget:

1. The fitted **`StandardScaler`** (the model was trained on *scaled* numbers; without the same scaler,
   predictions are garbage).
2. The **clipping caps** (177 / 40 / 13) applied to the three skewed columns. These are already built
   into `app/preprocess.py`.

## The roadmap

| Step | What you do | Guide | Time |
|---|---|---|---|
| 1 | Learn the basics & install tools | `01_prerequisites_and_basics.md` | 30–60 min |
| 2 | Understand your own project (for the interview!) | `02_project_details.md` | 20 min |
| 3 | Save model + scaler from the notebook; check file size | `03_export_model.md` | 15 min |
| 4 | Run the app on your own computer | `04_app_walkthrough.md` | 15 min |
| 5 | Push to GitHub and deploy on Streamlit Community Cloud | `05_deploy_streamlit_cloud.md` | 30 min |
| 6 | Prepare your demo and answers | `06_interview_guide.md` | 45 min |
| — | If anything breaks | `07_troubleshooting.md` | — |

## Final folder layout

```
fraud-risk-app/
├── app/
│   ├── app.py            ← the Streamlit user interface
│   ├── preprocess.py     ← capping + scaling (must match the notebook)
│   └── model_utils.py    ← loads model files, returns probabilities
├── artifacts/            ← YOU add these in Step 3
│   ├── model.joblib
│   ├── scaler.joblib
│   └── config.json
├── sample_data/
│   └── sample_transactions.csv
├── docs/                 ← these guides
├── notebook_export_cell.py   ← paste into your notebook
├── requirements.txt
└── README.md             ← what interviewers see first on GitHub
```

## What is already done for you vs. what only you can do

**Done:** the whole app code, preprocessing logic, batch scoring, sample data, export cell, guides.
The app code was tested end-to-end with a stand-in model that has the same schema as yours
(loading, scoring, presets, capping notice, what-if chart — no errors).

**Only you can do:** run the export cell in *your* notebook, drop the files into `artifacts/`,
pin the library versions, push to GitHub, click "Deploy". It has not been run against your actual
weights, so Step 4's local test is important.
