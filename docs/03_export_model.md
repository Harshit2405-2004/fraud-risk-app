# 03 · Export the Model from the Notebook

This is the most important technical step. The app needs **three files** in the `artifacts/` folder.

| File | What it is | Why it's needed |
|---|---|---|
| `model.joblib` | The fitted classifier | Makes the predictions |
| `scaler.joblib` | The fitted `StandardScaler` | Same scaling as training (see Guide 01) |
| `config.json` | Feature order, clipping caps, model name | Keeps the app and notebook in sync |

> Your notebook currently contains **no save code**, so this must be done in a session where the
> models are still trained in memory. If you closed Jupyter, re-run all cells top to bottom first
> (the stacked model is the slowest cell — it trains an SVM on ~1.46M rows).

---

## Step 1 — Paste the export cell

Open `notebook_export_cell.py` (in the project root), copy **everything**, paste it as a **new cell at
the end** of `RiskAnalysis.ipynb`, and run it. It will:

1. save the model and scaler into a new `artifacts/` folder next to the notebook,
2. write `config.json`,
3. **reload the model from disk and confirm it predicts identically** (`True` = good),
4. print each file's size in MB,
5. print the exact library versions to pin in `requirements.txt`.

Then copy the three files into `fraud-risk-app/artifacts/`.

---

## Step 2 — Check the file size (this decides your next move)

GitHub refuses files larger than **100 MB**. Streamlit Cloud's free tier also has limited memory
(roughly 1 GB at the time of writing — confirm on docs.streamlit.io), and loading a huge model uses
that memory.

Your Random Forest is 150 fully-grown trees trained on ~1.46 million rows, and the stacked model
contains a *second* refitted copy of that forest plus an SVM that stores its support vectors. Fully
grown forests like this are **commonly hundreds of MB or more** — I can't see your actual size, so check
the printout.

| Printed size of `model.joblib` | What to do |
|---|---|
| **< 90 MB** | Use it directly. Go to Step 4. |
| **90 MB – 500 MB** | Use Option A or B below |
| **> 500 MB** | Use Option B or C — the stacked model is not realistic for free hosting |

### Option A — Compress harder (cheap, try first)

```python
joblib.dump(FINAL_MODEL, "artifacts/model.joblib", compress=9)
```
Slower to load, but often much smaller. Check the size again.

### Option B — Train a *deployment-sized* Random Forest (recommended if too big)

Your RF + SMOTE already scores F1 0.9863 vs the stack's 0.9864 — practically identical. A smaller forest
usually loses almost nothing. Run in the notebook:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

light_rf = RandomForestClassifier(
    n_estimators=100, max_depth=18, min_samples_leaf=5,
    random_state=42, n_jobs=-1)
light_rf.fit(X_train_resampled, y_train_resampled)
print(classification_report(y_test, light_rf.predict(X_test_final), digits=5))
```

Compare to your 0.98633 fraud F1. If it's within ~0.002, use it: set
`FINAL_MODEL = light_rf` in the export cell, change `MODEL_NAME`, and re-run. Tune `max_depth`
(12–25) and `min_samples_leaf` (3–20) to trade size against accuracy. Update the numbers in the app's
"Model & project" tab only if you want it to show the deployed model's own metrics.

*Tell the interviewer about this choice — "I traded 0.0002 F1 for a model 20× smaller so it's
deployable" is a great engineering story.*

### Option C — XGBoost alone (smallest)

XGBoost models are typically a few MB. **However**, your notebook uses
`objective='binary:logitraw'`, which outputs raw scores rather than probabilities (its fraud F1 was also
lower, 0.9735). If you go this route, retrain with a proper probability objective:

```python
from xgboost import XGBClassifier
xgb_deploy = XGBClassifier(objective='binary:logistic', max_depth=5,
                           n_estimators=300, random_state=42, n_jobs=-1)
xgb_deploy.fit(X_train_resampled, y_train_resampled)
```
Evaluate it, then set `FINAL_MODEL = xgb_deploy`.

### Option D — Keep the big file off GitHub

If you really want the full stacked model, host `model.joblib` on **Hugging Face Hub** (free, allows
large files) and download it when the app starts:

```python
# in model_utils.py, before joblib.load
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="YOUR_USERNAME/fraud-model", filename="model.joblib")
model = joblib.load(path)
```
Add `huggingface_hub` to `requirements.txt`. Note the app must still fit in the host's memory, and the
first start will be slow because of the download. Git LFS is an alternative, but Streamlit Cloud's
handling of large LFS files can be unreliable — Option B is simpler.

---

## Step 3 — Pin the library versions (don't skip!)

A model pickled with scikit-learn *X* can fail — or, worse, **silently misbehave** — when loaded with
scikit-learn *Y*. The export cell prints lines like:

```
scikit-learn==1.x.x
xgboost==2.x.x
numpy==...
```

Replace the matching lines in `requirements.txt` with those exact versions (keep `streamlit`).
Also note the Python version you trained with and choose the same minor version when deploying.

> **Security note you can mention in an interview:** joblib/pickle files can execute code when loaded,
> so only load model files you created or fully trust.

---

## Step 4 — Verify you didn't forget anything

- [ ] `artifacts/model.joblib`, `scaler.joblib`, `config.json` all exist
- [ ] Export cell printed `Reloaded model gives identical predictions: True`
- [ ] `model.joblib` is under 100 MB (otherwise use Option A/B/C/D)
- [ ] `requirements.txt` pins the versions printed by the export cell
- [ ] If you changed `FINAL_MODEL`, you updated `MODEL_NAME`

### A note on batch speed

If you deploy the stacked model, scoring uses the SVM too, which is slow on many rows (cost grows with
the number of stored support vectors). That is why the app limits batch uploads to **20,000 rows**. A
Random Forest or XGBoost scores far faster.

### About probabilities

The app calls `predict_proba`. Stacking and Random Forest give real probabilities (from the
logistic-regression meta-learner or tree votes). Because training used SMOTE, treat them as **risk
scores** — see the limitations in Guide 02.
