# ============================================================
# PASTE THIS AS A NEW CELL AT THE VERY END OF RiskAnalysis.ipynb
# and run it in the SAME session where the models are trained
# (it needs: stacking_clf, model, xgb_model, scaler, X_test_final, y_test)
# ============================================================
import json
import os
import joblib
import numpy as np

os.makedirs("artifacts", exist_ok=True)

# ---- 1. Choose which trained model the app will use -----------------------
# stacking_clf  -> best/most impressive, but can be VERY large (see docs/03)
# model         -> Random Forest + SMOTE, nearly identical scores
FINAL_MODEL = stacking_clf
MODEL_NAME = "Stacked ensemble (RF + XGBoost + SVM -> Logistic Regression)"

# ---- 2. Save the model, the fitted scaler, and the config -----------------
joblib.dump(FINAL_MODEL, "artifacts/model.joblib", compress=3)
joblib.dump(scaler, "artifacts/scaler.joblib")

config = {
    "model_name": MODEL_NAME,
    "features": ["distance_from_home", "distance_from_last_transaction",
                 "ratio_to_median_purchase_price", "repeat_retailer",
                 "used_chip", "used_pin_number", "online_order"],
    "caps": {"distance_from_home": 177.0,
             "distance_from_last_transaction": 40.0,
             "ratio_to_median_purchase_price": 13.0},
}
with open("artifacts/config.json", "w") as f:
    json.dump(config, f, indent=2)

# ---- 3. Sanity check: reload from disk and compare predictions ------------
reloaded = joblib.load("artifacts/model.joblib")
sample = X_test_final[:2000]
same = np.array_equal(FINAL_MODEL.predict(sample), reloaded.predict(sample))
print("Reloaded model gives identical predictions:", same)

# ---- 4. File sizes (GitHub blocks files > 100 MB) --------------------------
for name in os.listdir("artifacts"):
    mb = os.path.getsize(os.path.join("artifacts", name)) / 1e6
    print(f"{name:15s} {mb:8.1f} MB")

# ---- 5. Library versions: copy these into requirements.txt -----------------
import sklearn, xgboost, pandas
print("\nPin these in requirements.txt:")
print(f"scikit-learn=={sklearn.__version__}")
print(f"xgboost=={xgboost.__version__}")
print(f"numpy=={np.__version__}")
print(f"pandas=={pandas.__version__}")
print(f"joblib=={joblib.__version__}")
