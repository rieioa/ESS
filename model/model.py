from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
FALLBACK_MODEL_DIR = Path("/tmp/geonwook_model")


def get_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def fit_model(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    model.fit(X, np.log(y))
    return model


def predict(model: Pipeline, X: np.ndarray) -> np.ndarray:
    return np.exp(model.predict(X))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def save_model(model: Pipeline, name: str = "ridge_alpha1_log_target", model_dir: Path = MODEL_DIR) -> Path:
    path = model_dir / f"{name}.pkl"
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        return path
    except PermissionError:
        FALLBACK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        fallback_path = FALLBACK_MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, fallback_path)
        return fallback_path
