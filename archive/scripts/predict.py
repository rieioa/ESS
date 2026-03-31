"""
predict.py — Inference logic only.

Responsibilities:
    - Generate predictions from a fitted model
    - Accept numpy arrays or DataFrames (model handles internally)

No training, evaluation, or data loading here.
"""

import os
import joblib
import numpy as np
import pandas as pd


def load_model(name: str, model_dir: str = None):
    """
    Loads a saved model from disk.

    Args:
        name      : Filename stem used when saving, e.g. 'xgboost_final'.
        model_dir : Directory to load from. Defaults to ../saved_models/
                    relative to this file.

    Returns:
        Fitted model object.
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    path = os.path.join(model_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at: {path}")
    return joblib.load(path)


def predict(model, X) -> np.ndarray:
    """
    Generate predictions from a fitted model.

    Args:
        model : Fitted sklearn-compatible estimator.
        X     : Feature array or DataFrame (numpy or pandas).

    Returns:
        Predicted cycle life values as a numpy array.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    return model.predict(X)
