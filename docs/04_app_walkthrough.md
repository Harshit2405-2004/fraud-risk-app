# 04 · The App, Explained (and How to Run It Locally)

## Run it on your computer first

```bash
cd fraud-risk-app
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app/app.py
```

Your browser opens at `http://localhost:8501`. **Always get it working locally before deploying** —
debugging is far easier here.

If the `artifacts/` folder is empty you'll see a red "Could not load the model files" message. That's
expected until you complete Guide 03.

---

## How the three code files fit together

```
User input / CSV
      │
      ▼
 app.py  ──────────▶ builds a DataFrame of the 7 raw features
      │
      ▼
 model_utils.predict_proba()
      │   calls
      ▼
 preprocess.transform():   clip  →  scale (saved scaler)  →  append 0/1 columns
      │
      ▼
 model.predict_proba()  →  P(fraud)  →  banner, metric, chart
```

### `preprocess.py` — the "must match the notebook" file

| Piece | Purpose |
|---|---|
| `NUMERIC_FEATURES`, `BINARY_FEATURES` | Column groups. **Order matters** — the model expects 3 scaled numeric columns first, then 4 binary ones, exactly as built in your notebook |
| `DEFAULT_CAPS` | 177 / 40 / 13 — the clipping thresholds from the notebook |
| `apply_caps()` | `clip(upper=cap)` — equivalent to your `np.where(x >= cap, cap, x)` |
| `transform()` | clip → `scaler.transform` on numeric columns → concatenate with binary columns |
| `which_were_capped()` | Lets the UI tell the user "your value was capped" |

### `model_utils.py` — loading and predicting

- `load_artifacts()` reads `model.joblib`, `scaler.joblib`, `config.json` from `artifacts/`.
- `predict_proba()` returns P(fraud) per row. If a model has no `predict_proba` (e.g. XGBoost with
  `logitraw`), it converts a raw score with a sigmoid as a fallback.

### `app.py` — the interface, section by section

| Section | What it does | Streamlit idea |
|---|---|---|
| `st.set_page_config` | Title, icon, wide layout | must be the first Streamlit call |
| `@st.cache_resource get_artifacts()` | Loads the model once, shared by all visitors | caching |
| `try/except … st.stop()` | Friendly error instead of a crash if files are missing | graceful failure |
| `PRESETS` + `apply_preset()` | The "Start from an example" dropdown fills in inputs | `session_state` + `on_change` callback |
| Sidebar slider | Decision threshold (default 0.50) | widget in `st.sidebar` |
| Tab 1 | Inputs on the left, result on the right, what-if chart below | `st.columns`, `st.metric`, `st.progress`, `st.line_chart` |
| Tab 2 | CSV upload → validation → scoring → download | `st.file_uploader`, `st.download_button` |
| Tab 3 | Project summary, results table, limitations | `st.dataframe`, `st.markdown` |

### Safeguards built in

- Missing columns / empty cells / over 20,000 rows are rejected with clear messages
- Extreme inputs are capped *and* the user is told
- The risk banner has three levels: flagged (≥ threshold), elevated (≥ half the threshold), legitimate
- The "Limitations" box is deliberate — it signals honesty and judgement

---

## Test checklist (do this locally before you push)

1. **Everyday chip & PIN purchase** preset → should show low probability, green banner.
2. **Suspicious: far away, online, 7x usual spend** preset → should show high probability, red banner.
   *If both presets give similar scores, something is wrong with preprocessing or the scaler — stop and
   debug.*
3. Type `5000` into "Distance from home" → blue "capped to 177" notice appears.
4. Drag the threshold slider → the banner changes at the right point.
5. What-if chart → pick `ratio_to_median_purchase_price`; probability should generally rise with the ratio.
6. Batch tab → download the sample CSV, upload it back → 4 rows scored; precision/recall appear because
   it contains a `fraud` column.
7. Upload a CSV missing a column → red error message, no crash.

### Best extra test: prove the app matches the notebook

In the notebook, take 5 rows of the **raw** (unscaled, uncapped) data, save to CSV and run them through
the app's batch tab. Then compare the app's `fraud_probability` with
`FINAL_MODEL.predict_proba(X_test_final[...])[:, 1]` for the same rows. They should match to several
decimals. This single check catches almost every preprocessing mistake.

---

## Making it yours (optional upgrades that impress)

| Upgrade | How |
|---|---|
| Show top feature importances | Add a bar chart in tab 3 using `model.feature_importances_` (works for RF) |
| Explain each prediction | Add the `shap` package (more dependencies, heavier app) |
| Replace "km" labels | Edit the labels in `app.py` if you decide on another unit |
| Different theme | Create `.streamlit/config.toml` with `[theme]` colours |
| Show real metrics for the *deployed* model | Edit the `results` table in tab 3 |

Make changes small, test locally, then push.
