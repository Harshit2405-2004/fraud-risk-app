"""Loads saved artifacts and produces fraud probabilities."""
import json
from pathlib import Path

import joblib
import numpy as np

from preprocess import DEFAULT_CAPS, FEATURES, transform

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_artifacts(artifact_dir: Path = ARTIFACT_DIR):
    model = joblib.load(artifact_dir / "model.joblib")
    scaler = joblib.load(artifact_dir / "scaler.joblib")
    cfg_path = artifact_dir / "config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    config.setdefault("features", FEATURES)
    config.setdefault("caps", DEFAULT_CAPS)
    config.setdefault("model_name", type(model).__name__)
    return model, scaler, config


def predict_proba(model, scaler, config, df) -> np.ndarray:
    """Return P(fraud) for each row of a raw-feature DataFrame."""
    X = transform(df, scaler, config["caps"])
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Fallback for models exposing only a raw score (e.g. XGB with binary:logitraw)
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
    else:
        score = model.predict(X, output_margin=True)
    return 1.0 / (1.0 + np.exp(-score))
