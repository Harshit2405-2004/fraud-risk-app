"""Inference-time preprocessing. MUST mirror the notebook exactly.

Notebook pipeline (in order):
  1. Clip 3 skewed numeric columns at fixed caps (177 / 40 / 13)
  2. StandardScaler fitted on TRAIN data, numeric columns only
  3. Concatenate [3 scaled numeric] + [4 binary columns] in that order
"""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
]
BINARY_FEATURES = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Same values as the notebook (approx. 98th / 98th / 99th percentiles)
DEFAULT_CAPS = {
    "distance_from_home": 177.0,
    "distance_from_last_transaction": 40.0,
    "ratio_to_median_purchase_price": 13.0,
}


def apply_caps(df: pd.DataFrame, caps: dict = None) -> pd.DataFrame:
    """Clip numeric features at the training-time caps (same as np.where(x>=cap, cap, x))."""
    caps = caps or DEFAULT_CAPS
    out = df.copy()
    for col, cap in caps.items():
        out[col] = out[col].clip(upper=cap)
    return out


def transform(df: pd.DataFrame, scaler, caps: dict = None) -> np.ndarray:
    """Raw feature DataFrame -> model-ready matrix."""
    df = apply_caps(df[FEATURES].astype(float), caps)
    scaled = scaler.transform(df[NUMERIC_FEATURES])
    binary = df[BINARY_FEATURES].to_numpy(dtype=float)
    return np.concatenate([scaled, binary], axis=1)


def which_were_capped(row: dict, caps: dict = None) -> list:
    caps = caps or DEFAULT_CAPS
    return [c for c, cap in caps.items() if row[c] > cap]
