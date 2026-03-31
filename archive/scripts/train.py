"""
train.py — All training logic.

Responsibilities:
    - Feature selection (dynamic, no hardcoded names)
    - GroupKFold cross-validation (cell-level, no leakage)
    - Cell-level hold-out split
    - Model fitting

No evaluation or reporting logic here — see test.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

RANDOM_SEED  = 42
MODEL_DIR    = os.path.join(os.path.dirname(__file__), '..', 'saved_models')


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, name: str, model_dir: str = MODEL_DIR) -> str:
    """
    Saves a fitted model to disk using joblib.

    Args:
        model     : Fitted sklearn-compatible estimator.
        name      : Filename stem, e.g. 'xgboost_final' → saved as 'xgboost_final.pkl'.
        model_dir : Directory to save into (created if it doesn't exist).

    Returns:
        Full path of the saved file.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Model saved → {path}")
    return path


# ── Feature selection ─────────────────────────────────────────────────────────

def get_features(X: pd.DataFrame, selected_features: list = None) -> list:
    """
    Returns the list of feature columns to use from X.

    Args:
        X                 : Feature DataFrame returned by the dataloader.
        selected_features : Optional list of column names to restrict to.
                            Pass None to use every column in X.

    Returns:
        List of column name strings.

    Example:
        features = get_features(X)                          # all columns
        features = get_features(X, ['dQ_min', 'log_dQ_var'])  # subset
    """
    if selected_features is None:
        return X.columns.tolist()

    missing = [f for f in selected_features if f not in X.columns]
    if missing:
        raise ValueError(f"Requested features not found in X: {missing}")

    return selected_features


# ── Model fitting ─────────────────────────────────────────────────────────────

def train_model(model, X: np.ndarray, y: np.ndarray):
    """
    Fits a model in-place and returns it.

    Args:
        model : Unfitted sklearn-compatible estimator.
        X     : Training feature array.
        y     : Training target array.

    Returns:
        Fitted model.
    """
    model.fit(X, y)
    return model


# ── XGBoost training with early stopping ─────────────────────────────────────

def train_xgboost(
    model,
    X_train:               np.ndarray,
    y_train:               np.ndarray,
    X_val:                 np.ndarray,
    y_val:                 np.ndarray,
    early_stopping_rounds: int = 20,
    eval_metric:           str = 'rmse',
):
    """
    Fits an XGBRegressor with early stopping.

    Training halts when the eval metric on (X_val, y_val) has not improved
    for `early_stopping_rounds` consecutive rounds. The model is restored
    to the best checkpoint automatically by XGBoost.

    Use this instead of train_model() for XGBoost so that n_estimators
    (set high in model.py) is tuned automatically per training run.

    Args:
        model                 : Unfitted XGBRegressor from get_xgboost().
        X_train, y_train      : Training data.
        X_val,   y_val        : Validation data used only for early stopping
                                (not exposed to the model as training signal).
        early_stopping_rounds : Stop if no improvement for this many rounds.
        eval_metric           : XGBoost metric to monitor ('rmse', 'mae', etc.).

    Returns:
        Fitted XGBRegressor (stopped at best round).
    """
    model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose  = False,
    )
    print(f"  XGBoost early stopping: best round = {model.best_iteration + 1} "
          f"/ {model.n_estimators}")
    return model


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_val_mape(
    model,
    X:        np.ndarray,
    y:        np.ndarray,
    groups:   np.ndarray,
    n_splits: int = 5,
) -> float:
    """
    GroupKFold cross-validation returning mean MAPE (%) across folds.

    Splitting is done at the cell level (groups = cell_ids) so that no
    battery cell ever appears in both the train and validation fold.
    This prevents leakage from the high inter-cycle correlation within
    the same cell.

    Args:
        model    : Unfitted sklearn-compatible estimator.
        X        : Feature array (numpy), one row per cell.
        y        : Target array (cycle_life).
        groups   : Cell ID array aligned with X rows.
        n_splits : Number of CV folds.

    Returns:
        Mean MAPE (%) across all folds.
    """
    gkf    = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        m = clone(model)          # fresh copy per fold — no state bleed
        m.fit(X[train_idx], y[train_idx])
        y_pred  = m.predict(X[val_idx])
        fold_mape = float(np.mean(np.abs((y[val_idx] - y_pred) / y[val_idx])) * 100)
        scores.append(fold_mape)

    return float(np.mean(scores))


# ── Cell-level hold-out split ─────────────────────────────────────────────────

def cell_holdout_split(
    X:        np.ndarray,
    y:        np.ndarray,
    cell_ids: np.ndarray,
    val_ratio: float = 0.2,
    seed:      int   = RANDOM_SEED,
):
    """
    Splits data into train / valid sets strictly by cell ID.

    Shuffles unique cell IDs, assigns the last `val_ratio` fraction to
    validation. Guarantees no cell appears in both splits.

    Args:
        X         : Feature array.
        y         : Target array.
        cell_ids  : Cell ID array aligned with X rows.
        val_ratio : Fraction of cells to reserve for validation.
        seed      : Random seed for reproducibility.

    Returns:
        X_train, X_valid, y_train, y_valid
    """
    rng          = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    rng.shuffle(unique_cells)

    n_val     = max(1, int(len(unique_cells) * val_ratio))
    val_cells = set(unique_cells[-n_val:])

    val_mask   = np.array([cid in val_cells for cid in cell_ids])
    train_mask = ~val_mask

    return (
        X[train_mask], X[val_mask],
        y[train_mask], y[val_mask],
    )
