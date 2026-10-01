# 07 · Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: ... model.joblib` or red "Could not load the model files" | `artifacts/` empty or wrong location | Put the 3 files in `fraud-risk-app/artifacts/` (next to `app/`, not inside it) |
| `ModuleNotFoundError: xgboost` / `imblearn` etc. on Streamlit Cloud | Package missing from `requirements.txt` | A stacked model that contains an XGBoost model needs `xgboost` installed to unpickle. Add every library the model uses. (`imbalanced-learn` is **not** needed — SMOTE isn't part of the saved model.) |
| `InconsistentVersionWarning` or odd errors on load | scikit-learn version differs from training | Pin the exact versions printed by the export cell; choose a matching Python version in Advanced settings |
| `AttributeError: Can't get attribute ...` when loading | Same cause — a version mismatch | Same fix |
| Both "safe" and "suspicious" examples give the same probability | Scaler not applied, wrong scaler, or wrong column order | Re-run the "prove the app matches the notebook" test (Guide 04); confirm `scaler.joblib` came from the same session as the model |
| Predictions all near 0 or near 1 | Same as above, or the wrong `FINAL_MODEL` was exported | Reload and test in the notebook first |
| `git push` rejected: file larger than 100 MB | Model too big | Guide 03 Options A–D; if already committed, `git reset --soft HEAD~1` first |
| App is killed / "resource limits exceeded" / keeps restarting | Model too large for free-tier memory | Use a smaller Random Forest (Option B) or XGBoost (Option C) |
| App is very slow on batch upload | SVM inside stacked model | Smaller upload, or deploy RF/XGBoost instead |
| "Main file path" error on deploy | Wrong path | It must be `app/app.py` |
| Works locally, fails on cloud | Windows vs Linux paths, missing requirement, version differences | Read the logs (bottom-right "Manage app"); fix and push again |
| Inputs don't update when picking an example | Old Streamlit version | `pip install -U streamlit` and use `streamlit>=1.40` |
| `st.toggle` not found | Streamlit older than 1.26 | Upgrade Streamlit |
| App asleep when interviewer opens it | Free-tier inactivity sleep | Open it yourself shortly before; mention wake-up time in your CV |
| Page shows old version after push | Redeploy still running | Wait for the build; use "Reboot app" in Manage app |

## Debugging habits that save hours

1. **Reproduce locally** with a fresh virtual environment built only from `requirements.txt` — if it fails
   there, it will fail in the cloud.
2. **Read the first error in the logs**, not the last; later errors are usually side-effects.
3. Add a temporary `st.write(df)` or `st.write(X)` to inspect what the model actually receives.
4. Change one thing at a time and commit often.

## Still stuck?

Copy the **full** error text (not a screenshot of part of it), your `requirements.txt`, and the printed
file sizes from the export cell, and ask for help with those three things.
