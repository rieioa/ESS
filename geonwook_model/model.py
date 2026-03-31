"""
model.py — Model definitions plus minimal training/eval utilities.

This module intentionally absorbs the old train/predict helpers so that
the experiment path only depends on feature.py and model.py.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
MODEL_DIR = Path(__file__).parent.parent / 'saved_models'


def get_linear() -> Pipeline:
    """OLS with StandardScaler. Best for small n when features are already invariant."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])


def get_elastic_net() -> Pipeline:
    """
    Elastic Net with StandardScaler.
    Scaling is required — features live on very different numeric scales.
    alpha=0.1, l1_ratio=0.5 gives a balanced Lasso/Ridge mix.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(
            alpha        = 0.1,
            l1_ratio     = 0.5,   # 0 = Ridge, 1 = Lasso
            max_iter     = 10_000,
            random_state = RANDOM_SEED,
        )),
    ])


def get_xgboost():
    """
    XGBRegressor tuned conservatively for small battery datasets (~40 cells).
    Shallow trees + regularization reduce overfitting on low-sample regimes.

    n_estimators is set high (1000) as an upper bound.
    fit_xgboost() applies early stopping to find the
    optimal number of rounds automatically using a held-out eval set.
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators     = 1000,  # upper bound — early stopping cuts this short
        max_depth        = 3,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 1.0,
        reg_alpha        = 0.1,   # L1
        reg_lambda       = 1.0,   # L2
        random_state     = RANDOM_SEED,
        verbosity        = 0,
    )


def get_model(name: str):
    """
    Model factory — returns an unfitted model by name.

    Args:
        name: 'elastic_net' or 'xgboost'

    Returns:
        Unfitted sklearn-compatible estimator.

    Raises:
        ValueError if name is not recognised.
    """
    registry = {
        'linear':      get_linear,
        'elastic_net': get_elastic_net,
        'xgboost':     get_xgboost,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name]()


def save_model(model, name: str, model_dir: Path = MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f'{name}.pkl'
    joblib.dump(model, path)
    print(f'  Model saved -> {path}')
    return path


def load_model(name: str, model_dir: Path = MODEL_DIR):
    path = model_dir / f'{name}.pkl'
    if not path.exists():
        raise FileNotFoundError(f'No saved model found at: {path}')
    return joblib.load(path)


def predict(model, X: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.values
    return model.predict(X)


def fit_model(model, X: np.ndarray, y: np.ndarray):
    model.fit(X, y)
    return model


def fit_xgboost(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 20,
):
    model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f'  XGBoost early stopping: best round = {model.best_iteration + 1} / {model.n_estimators}')
    return model


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    y_pred = predict(model, X)
    return {'mape': mape(y, y_pred), 'rmse': rmse(y, y_pred)}


def cross_val_mape(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    scores: list[float] = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred = fold_model.predict(X[val_idx])
        scores.append(mape(y[val_idx], y_pred))

    return float(np.mean(scores))


def cell_holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    rng.shuffle(unique_cells)

    n_val = max(1, int(len(unique_cells) * val_ratio))
    val_cells = set(unique_cells[-n_val:])

    val_mask = np.array([cid in val_cells for cid in cell_ids])
    train_mask = ~val_mask

    return (
        X[train_mask],
        X[val_mask],
        y[train_mask],
        y[val_mask],
    )
