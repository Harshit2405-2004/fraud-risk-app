"""Credit-Card Fraud Risk Analyzer - Streamlit app."""
import io

import numpy as np
import pandas as pd
import streamlit as st

from model_utils import load_artifacts, predict_proba
from preprocess import (BINARY_FEATURES, DEFAULT_CAPS, FEATURES,
                        NUMERIC_FEATURES, which_were_capped)

st.set_page_config(page_title="Fraud Risk Analyzer", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    model, scaler, config = get_artifacts()
except Exception as exc:  # shows a friendly message instead of a stack trace
    st.error("Could not load the model files from the `artifacts/` folder.")
    st.exception(exc)
    st.stop()

CAPS = config["caps"]

# ------------------------------------------------------------------ presets
PRESETS = {
    "Custom (set your own values)": None,
    "Everyday chip & PIN purchase near home": dict(
        distance_from_home=4.0, distance_from_last_transaction=0.5,
        ratio_to_median_purchase_price=0.9, repeat_retailer=True,
        used_chip=True, used_pin_number=True, online_order=False),
    "Routine online order from a known shop": dict(
        distance_from_home=8.0, distance_from_last_transaction=1.0,
        ratio_to_median_purchase_price=1.1, repeat_retailer=True,
        used_chip=False, used_pin_number=False, online_order=True),
    "Suspicious: far away, online, 7x usual spend": dict(
        distance_from_home=150.0, distance_from_last_transaction=25.0,
        ratio_to_median_purchase_price=7.0, repeat_retailer=False,
        used_chip=False, used_pin_number=False, online_order=True),
}
DEFAULTS = PRESETS["Everyday chip & PIN purchase near home"]

for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def apply_preset():
    preset = PRESETS[st.session_state["preset"]]
    if preset:
        for key, val in preset.items():
            st.session_state[key] = val


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.write("Scores a card transaction for fraud risk using a machine-learning "
             "model trained on 1,000,000 transactions.")
    st.caption(f"Model: **{config['model_name']}**")
    st.divider()
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="Transactions with P(fraud) at or above this value are flagged. "
             "Lower = catch more fraud but more false alarms.")
    st.caption("Tip: lower the threshold when missing fraud is costly; "
               "raise it when false alarms are costly.")

st.title("🛡️ Credit-Card Fraud Risk Analyzer")
tab_single, tab_batch, tab_about = st.tabs(
    ["🔍 Score one transaction", "📄 Batch scoring (CSV)", "📊 Model & project"])


def risk_banner(p: float, thr: float):
    if p >= thr:
        st.error(f"🚨 FLAGGED as likely fraud  (P = {p:.1%}, threshold {thr:.0%})")
    elif p >= thr * 0.5:
        st.warning(f"⚠️ Elevated risk, below flag threshold  (P = {p:.1%})")
    else:
        st.success(f"✅ Looks legitimate  (P = {p:.1%})")


# ------------------------------------------------------------------ tab 1
with tab_single:
    st.selectbox("Start from an example", list(PRESETS), key="preset",
                 on_change=apply_preset)
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Transaction details")
        distance_from_home = st.number_input(
            "Distance from home (km)", 0.0, 12000.0, step=1.0, key="distance_from_home",
            help="How far from the cardholder's home the purchase happened.")
        distance_from_last_transaction = st.number_input(
            "Distance from last transaction (km)", 0.0, 12000.0, step=0.5,
            key="distance_from_last_transaction")
        ratio_to_median_purchase_price = st.number_input(
            "Purchase amount ÷ cardholder's median purchase", 0.0, 300.0, step=0.1,
            key="ratio_to_median_purchase_price",
            help="1.0 = a typical purchase. 6.0 = six times the usual amount.")
        c1, c2 = st.columns(2)
        repeat_retailer = c1.toggle("Repeat retailer", key="repeat_retailer",
                                    help="Has this card bought from this retailer before?")
        used_chip = c2.toggle("Chip used", key="used_chip")
        used_pin_number = c1.toggle("PIN used", key="used_pin_number")
        online_order = c2.toggle("Online order", key="online_order")

    row = dict(
        distance_from_home=distance_from_home,
        distance_from_last_transaction=distance_from_last_transaction,
        ratio_to_median_purchase_price=ratio_to_median_purchase_price,
        repeat_retailer=float(repeat_retailer), used_chip=float(used_chip),
        used_pin_number=float(used_pin_number), online_order=float(online_order),
    )
    prob = float(predict_proba(model, scaler, config, pd.DataFrame([row]))[0])

    with right:
        st.subheader("Result")
        st.metric("Fraud probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        risk_banner(prob, threshold)
        capped = which_were_capped(row, CAPS)
        if capped:
            st.info("Very large values are capped to match training data: "
                    + ", ".join(f"`{c}` → {CAPS[c]:g}" for c in capped))

    st.divider()
    st.subheader("What-if analysis")
    st.caption("Vary one numeric feature while holding the others fixed to see how the "
               "model's risk score responds.")
    sweep = st.selectbox("Feature to vary", NUMERIC_FEATURES)
    grid = np.linspace(0, CAPS[sweep], 60)
    sweep_df = pd.DataFrame([{**row, sweep: v} for v in grid])
    curve = pd.DataFrame({"P(fraud)": predict_proba(model, scaler, config, sweep_df)},
                         index=pd.Index(grid, name=sweep))
    st.line_chart(curve)

# ------------------------------------------------------------------ tab 2
with tab_batch:
    st.subheader("Score many transactions at once")
    st.write("Upload a CSV with these columns (an optional `fraud` column with 0/1 "
             "labels lets the app compute accuracy):")
    st.code(", ".join(FEATURES))
    sample = pd.DataFrame([DEFAULTS | {"distance_from_home": 4.0},
                           PRESETS["Suspicious: far away, online, 7x usual spend"]])
    sample[BINARY_FEATURES] = sample[BINARY_FEATURES].astype(float)
    st.download_button("Download a sample CSV", sample.to_csv(index=False),
                       "sample_transactions.csv", "text/csv")

    upload = st.file_uploader("Upload CSV", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
            missing = [c for c in FEATURES if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif data.shape[0] > 20_000:
                st.error("Please upload at most 20,000 rows.")
            elif data[FEATURES].isnull().any().any():
                st.error("File contains empty cells in the feature columns.")
            else:
                data["fraud_probability"] = predict_proba(model, scaler, config, data)
                data["flagged"] = (data["fraud_probability"] >= threshold).astype(int)
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows scored", f"{len(data):,}")
                m2.metric("Flagged", f"{int(data['flagged'].sum()):,}")
                m3.metric("Flag rate", f"{data['flagged'].mean():.1%}")

                if "fraud" in data.columns:
                    from sklearn.metrics import precision_score, recall_score
                    y = data["fraud"].astype(int)
                    st.write("**Labels found - performance at current threshold**")
                    a, b = st.columns(2)
                    a.metric("Precision", f"{precision_score(y, data['flagged'], zero_division=0):.3f}")
                    b.metric("Recall", f"{recall_score(y, data['flagged'], zero_division=0):.3f}")
                    st.dataframe(pd.crosstab(y.rename("actual"), data["flagged"].rename("flagged")))

                st.dataframe(data.sort_values("fraud_probability", ascending=False).head(500),
                             )
                buf = io.StringIO()
                data.to_csv(buf, index=False)
                st.download_button("Download scored CSV", buf.getvalue(),
                                   "scored_transactions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process that file: {exc}")

# ------------------------------------------------------------------ tab 3
with tab_about:
    st.subheader("Project summary")
    st.markdown(
        "- **Data:** 1,000,000 card transactions, 7 features, ~8.7% fraud (class-imbalanced)\n"
        "- **Preprocessing:** outlier capping at ~98th/99th percentile, standard scaling, "
        "SMOTE on the *training split only*\n"
        "- **Models compared:** Logistic Regression, SVM, Random Forest, XGBoost, "
        "and a stacked ensemble (RF + XGBoost + SVM with a logistic-regression meta-learner)\n"
        "- **Evaluation:** 20% hold-out set (200,000 transactions), focus on fraud-class "
        "precision and recall rather than accuracy")

    results = pd.DataFrame(
        [["Logistic Regression", 0.96198, 0.83753, 0.69982, 0.76251],
         ["SVM (RBF)", 0.99632, 0.96790, 0.99066, 0.97915],
         ["Random Forest (no SMOTE)", 0.99710, 0.97627, 0.99077, 0.98347],
         ["Random Forest + SMOTE", 0.99759, 0.97380, 0.99920, 0.98633],
         ["XGBoost + SMOTE", 0.99528, 0.95294, 0.99495, 0.97350],
         ["Stacked ensemble + SMOTE", 0.99760, 0.97406, 0.99903, 0.98639]],
        columns=["Model", "Accuracy", "Fraud precision", "Fraud recall", "Fraud F1"])
    st.dataframe(results.style.format({c: "{:.4f}" for c in results.columns[1:]}),
                 hide_index=True)
    st.caption("Metrics measured on the hold-out test set in the training notebook.")

    st.warning(
        "**Limitations.** The dataset appears synthetic and is easy to separate (hence the very "
        "high scores). SMOTE changes class balance, so the probabilities are *risk scores*, not "
        "calibrated real-world fraud rates. This is a portfolio demo, not a production fraud system.")
