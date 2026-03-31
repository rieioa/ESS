"""
test.py — Main experiment runner.

Responsibilities:
    - Load features.csv and split into train (Batch 1) / test (Batch 2)
    - Select features dynamically via get_features()
    - Run CV and hold-out validation on Batch 1
    - Evaluate final model on Batch 2 (test)
    - Print standardised performance table

Imports from:
    model.py   → get_model()
    train.py   → get_features(), train_model(), train_xgboost(), cross_val_mape(), cell_holdout_split()
    predict.py → predict()
"""

import os
import sys
import numpy as np
import pandas as pd

# Allow running as: python scripts/test.py from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from model   import get_model
from train   import get_features, train_model, train_xgboost, save_model, cross_val_mape, cell_holdout_split
from predict import predict

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TARGET      = 9.1    # Paper baseline MAPE (%)

FEATURES_CSV = os.path.join(os.path.dirname(__file__), '..', 'preprocessed_data', 'features.csv')

BATCH1_NAME = '2017-05-12'
BATCH2_NAME = '2018-02-12'

# Features to use — pass None to use all columns, or specify a subset.
SELECTED_FEATURES = ['delta_q_min', 'log_delta_q_var']


# ── Evaluation metrics ────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Returns {'mape': float, 'rmse': float} for a fitted model."""
    y_pred = predict(model, X)
    return {'mape': mape(y, y_pred), 'rmse': rmse(y, y_pred)}


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(
    model_name:    str,
    train_cv_mape: float,
    valid_mape:    float,
    test_mape:     float,
) -> None:
    """
    Prints the standardised performance table.

    Gap sign convention (MAPE — lower is better):
      Gap(Train-Valid) < 0  →  train fits better than valid  →  overfitting
      Gap(Valid-Test)  < 0  →  valid fits better than test   →  generalization drop
    """
    gap_tv = train_cv_mape - valid_mape
    gap_vt = valid_mape    - test_mape
    gap_tg = test_mape     - TARGET

    note_tv = "(+) Overfitting suspected" if gap_tv < 0 else ""
    note_vt = "(+) Generalization drop"   if gap_vt < 0 else ""
    note_tg = "Target = 9.1%"

    w = 60
    print(f"\n{'=' * w}")
    print(f"  Model: {model_name}")
    print(f"{'=' * w}")
    print(f"{'구분':<26}| {'MAPE (%)':>9} | 비고")
    print(f"{'-' * 26}|{'-' * 11}|{'-' * 28}")
    print(f"{'Train (Batch 1 CV)':<26}| {train_cv_mape:>9.2f} |")
    print(f"{'Valid (Batch 1 Hold-out)':<26}| {valid_mape:>9.2f} |")
    print(f"{'Test (Batch 2)':<26}| {test_mape:>9.2f} |")
    print(f"{'-' * 26}|{'-' * 11}|{'-' * 28}")
    print(f"{'Gap (Train-Valid)':<26}| {gap_tv:>9.2f} | {note_tv}")
    print(f"{'Gap (Valid-Test)':<26}| {gap_vt:>9.2f} | {note_vt}")
    print(f"{'Gap (Target-Test)':<26}| {gap_tg:>9.2f} | {note_tg}")
    print(f"{'=' * w}\n")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads features.csv and splits into Batch 1 (train) and Batch 2 (test).

    Returns:
        df1 : Batch 1 DataFrame (cell_uid index, feature cols + cycle_life)
        df2 : Batch 2 DataFrame (same schema)
    """
    df = pd.read_csv(FEATURES_CSV)
    df1 = df[df['batch_name'] == BATCH1_NAME].copy()
    df2 = df[df['batch_name'] == BATCH2_NAME].copy()
    print(f"  Batch 1: {len(df1)} cells, Batch 2: {len(df2)} cells\n")
    return df1, df2


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("Loading features.csv...")
    df1, df2 = load_dataset()

    # 2. Select features (dynamic — edit SELECTED_FEATURES at top to experiment)
    feature_cols = get_features(
        df1.drop(columns=['cell_uid', 'batch_name', 'cycle_life', 'charging_policy'], errors='ignore'),
        selected_features=SELECTED_FEATURES,
    )
    print(f"Features used: {feature_cols}\n")

    # 3. Prepare arrays
    X1       = df1[feature_cols].values
    y1       = df1['cycle_life'].values
    cell_ids = df1['cell_uid'].values

    X2 = df2[feature_cols].values
    y2 = df2['cycle_life'].values

    # 4. Hold-out split of Batch 1 (cell-level, no leakage)
    X_train, X_valid, y_train, y_valid = cell_holdout_split(
        X1, y1, cell_ids, val_ratio=0.2, seed=RANDOM_SEED
    )

    # 5. Train and evaluate each model
    for model_name in ['elastic_net', 'xgboost']:
        print(f"Running: {model_name} ...")

        # 5a. Cross-validation on full Batch 1 → Train score
        cv_mape = cross_val_mape(
            get_model(model_name), X1, y1, groups=cell_ids, n_splits=5
        )

        # 5b. Hold-out validation → Valid score
        model_valid = get_model(model_name)
        if model_name == 'xgboost':
            train_xgboost(model_valid, X_train, y_train, X_valid, y_valid)
        else:
            train_model(model_valid, X_train, y_train)
        valid_metrics = evaluate(model_valid, X_valid, y_valid)

        # 5c. Final model trained on full Batch 1 → Test score on Batch 2
        model_final = get_model(model_name)
        if model_name == 'xgboost':
            train_xgboost(model_final, X1, y1, X2, y2)
        else:
            train_model(model_final, X1, y1)
        test_metrics = evaluate(model_final, X2, y2)

        # 5d. Save the final model
        save_model(model_final, name=f"{model_name}_final")

        # 5e. Report
        label = 'Elastic Net' if model_name == 'elastic_net' else 'XGBoost'
        print_results_table(
            model_name    = label,
            train_cv_mape = cv_mape,
            valid_mape    = valid_metrics['mape'],
            test_mape     = test_metrics['mape'],
        )

        print(f"  RMSE — Valid: {valid_metrics['rmse']:.1f}  |  "
              f"Test: {test_metrics['rmse']:.1f}\n")


if __name__ == '__main__':
    main()
