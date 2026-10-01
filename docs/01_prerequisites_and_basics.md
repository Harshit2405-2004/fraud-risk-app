# 01 · Prerequisites & Basics

Everything you need to know *before* touching the deployment. Read the concepts even if you've
installed the tools — interviewers love to ask "how does it actually work?"

---

## Part A — Tools to install

| Tool | Why | Get it |
|---|---|---|
| **Python 3.10 – 3.12** | Runs the app. Use the same major version you trained with if possible. | python.org (tick **"Add Python to PATH"** on Windows) |
| **VS Code** (or any editor) | Edit files | code.visualstudio.com |
| **Git** | Version control; needed to push to GitHub | git-scm.com |
| **GitHub account** | Streamlit Cloud deploys from a GitHub repo | github.com |
| **Streamlit Community Cloud account** | Free hosting — sign in with GitHub | share.streamlit.io |

Check your install (Windows PowerShell / Mac/Linux terminal):

```bash
python --version
git --version
pip --version
```

---

## Part B — Terminal basics (Windows-flavoured)

| Command | Meaning |
|---|---|
| `cd folder_name` | go into a folder |
| `cd ..` | go up one folder |
| `dir` (Windows) / `ls` (Mac/Linux) | list files |
| `mkdir name` | create a folder |

### Virtual environments — why and how

A **virtual environment** is a private box of Python packages for one project, so versions from other
projects don't clash. This matters a lot here: a saved scikit-learn model must be loaded with the
**same scikit-learn version** that saved it.

```bash
cd fraud-risk-app
python -m venv .venv

# activate it
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
```

You'll see `(.venv)` at the start of your prompt when it's active.

### requirements.txt

A plain list of packages. Streamlit Cloud reads it and installs them on its servers. If a package is
missing from this file, the deployed app crashes with `ModuleNotFoundError` even though it worked on
your laptop — the #1 deployment bug.

---

## Part C — Machine-learning concepts used in this project

### What are "trained weights" for *this* project?

Neural networks have "weights". Your models (Random Forest, XGBoost, SVM, Logistic Regression) are
scikit-learn / XGBoost objects: the "weights" are the learned trees, support vectors, and coefficients
**stored inside the fitted Python object**. So "saving the weights" means **serializing the whole fitted
object** to a file with `joblib.dump`, and later `joblib.load` brings it back ready to `predict`.

### Training vs inference

| | Training (your notebook) | Inference (the app) |
|---|---|---|
| Data | 1,000,000 labelled rows | One transaction at a time |
| Time | Minutes–hours | Milliseconds |
| Output | A fitted model | A fraud probability |

The app only does **inference**. It never retrains.

### Why the scaler must be saved (train/serve skew)

During training you did `StandardScaler().fit_transform(...)` on three columns. The scaler remembered
each column's training mean and standard deviation. The model learned from *scaled* values. At inference
time, you must apply **the same mean and std** — not recompute them from one row. Saving
`scaler.joblib` guarantees this. Mismatch between training and serving preprocessing is called
**train/serve skew** and silently ruins predictions.

### Why the clipping caps matter

You clipped `distance_from_home` at 177, `distance_from_last_transaction` at 40 and
`ratio_to_median_purchase_price` at 13 *before* scaling. If a user types 5,000 km, the app must clip it
to 177 first, exactly like training. That's what `preprocess.py` does.

### Class imbalance and SMOTE

Only about **8.7 %** of transactions are fraud. A model that always says "not fraud" would score 91 %
accuracy while being useless. So you:

- judge models by **fraud-class precision, recall and F1**, not accuracy;
- used **SMOTE** (Synthetic Minority Over-sampling) to create synthetic fraud examples **in the training
  set only**, so the model sees a balanced 50/50 split. The test set stays untouched and realistic.

### Precision vs recall (the decision threshold)

- **Recall** = of all real frauds, how many did we catch?
- **Precision** = of everything we flagged, how many were truly fraud?

A threshold of 0.5 means "flag if P(fraud) ≥ 50 %". Lowering it catches more fraud (↑ recall) but
annoys more honest customers (↓ precision). The sidebar slider demonstrates exactly this.

### Stacking (your best model)

A **stacked ensemble** trains several different models (Random Forest, XGBoost, SVM), then trains a
simple **meta-model** (Logistic Regression) to combine their predictions. Idea: different models make
different mistakes; the meta-model learns whom to trust.

---

## Part D — Streamlit basics

Streamlit turns a Python script into a web app. Four ideas explain almost everything:

1. **Script reruns top-to-bottom on every interaction.** Move a slider → the whole `app.py` runs again.
2. **Widgets return values.** `x = st.number_input(...)` gives you the current number directly.
3. **Caching avoids repeated work.** `@st.cache_resource` loads the model once and reuses it across
   reruns/users. Without it, the model would reload on every click.
4. **`st.session_state`** remembers values between reruns (used here so the "example" dropdown can fill
   in the input boxes).

Try the smallest possible app to feel it:

```python
import streamlit as st
name = st.text_input("Your name")
st.write(f"Hello {name}!")
```
Save as `hello.py`, run `streamlit run hello.py`.

---

## Part E — Git & GitHub basics

| Term | Meaning |
|---|---|
| **Repository (repo)** | A project folder tracked by Git |
| **Commit** | A saved snapshot with a message |
| **Push** | Upload your commits to GitHub |
| **Branch** | A line of development; yours is `main` |
| `.gitignore` | List of files Git should skip (like `.venv/`) |

**Limit to remember:** GitHub rejects any single file over **100 MB**. Large model files are the usual
cause of failed pushes — Guide 03 explains how to avoid this.

---

## Part F — How the deployment works (big picture)

```
Your laptop ──git push──▶ GitHub repo ──▶ Streamlit Community Cloud
                                              │  1. clones repo
                                              │  2. pip installs requirements.txt
                                              │  3. runs  streamlit run app/app.py
                                              ▼
                                   https://your-app.streamlit.app   ← link for the interviewer
```

Updating the app later = edit locally → `git push` → the live app redeploys automatically.

---

### ✅ Checklist before moving on

- [ ] `python --version`, `git --version` both work
- [ ] GitHub account created
- [ ] You can explain train/serve skew, SMOTE, and precision vs recall in your own words
