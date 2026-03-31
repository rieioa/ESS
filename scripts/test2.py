"""
test2.py — Best experiment reproduction script.

Reproduces the best-performing configuration found via feature/hyperparameter search:
  - Features : ['delta_q_var', 'low_min', 'v5_log_var', 'v2_v3_ratio', 'q10_v3_var']
  - Target   : log(cycle_life)  → inverted with exp() for MAPE
  - Models   : Linear Regression, Elastic Net (alpha=0.0001, l1_ratio=0.1), XGBoost

Prerequisites (run once in order before this script):
  1. python scripts/feature.py      # extracts features.csv from preprocessed data
  2. python scripts/_feature.py     # recomputes Batch 2 cycle_life at 0.88 Ah cutoff
  3. python scripts/test2.py        # this file

Results reference (2026-03-31):
  Linear Regression  — Train CV: 11.35%  Valid: 13.26%  Test: 17.83%
  Elastic Net        — Train CV: 11.06%  Valid: 13.51%  Test: 17.90%
  XGBoost            — Train CV: 10.12%  Valid: 20.47%  Test: 24.23%
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))

# ── Constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TARGET      = 9.1    # Paper baseline MAPE (%)

FEATURES_CSV = os.path.join(os.path.dirname(__file__), '..', 'preprocessed_data', 'features.csv')

BATCH1_NAME = '2017-05-12'
BATCH2_NAME = '2018-02-20'
BATCH3_NAME = '2018-04-12'

SELECTED_FEATURES = ['delta_q_var', 'low_min', 'v5_log_var', 'v2_v3_ratio', 'q10_v3_var']
LOG_TARGET        = True


# ── Model definitions ─────────────────────────────────────────────────────────

def make_linear_regression():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])


def make_elastic_net():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  ElasticNet(
            alpha        = 0.0001,
            l1_ratio     = 0.1,
            max_iter     = 10_000,
            random_state = RANDOM_SEED,
        )),
    ])


def make_xgboost():
    return XGBRegressor(
        n_estimators     = 1000,
        max_depth        = 3,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 1.0,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = RANDOM_SEED,
        verbosity        = 0,
    )


MODEL_REGISTRY = {
    'Linear Regression': make_linear_regression,
    'Elastic Net':       make_elastic_net,
    'XGBoost':           make_xgboost,
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ── Training helpers ──────────────────────────────────────────────────────────

def cell_holdout_split(X, y, cell_ids, val_ratio=0.2, seed=RANDOM_SEED):
    rng          = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    rng.shuffle(unique_cells)
    n_val        = max(1, int(len(unique_cells) * val_ratio))
    val_cells    = set(unique_cells[-n_val:])
    val_mask     = np.array([cid in val_cells for cid in cell_ids])
    train_mask   = ~val_mask
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


def cross_val_mape(model, X, y, groups, n_splits=5):
    gkf    = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])
        y_true = y[val_idx]
        if LOG_TARGET:
            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
        scores.append(float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100))
    return float(np.mean(scores))


def train_xgboost(model, X_train, y_train, X_val, y_val, early_stopping_rounds=20):
    model.set_params(early_stopping_rounds=early_stopping_rounds)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  XGBoost early stopping: best round = {model.best_iteration + 1} / {model.n_estimators}")
    return model


def eval_with_inv(model, X, y_raw):
    yp = model.predict(X)
    if LOG_TARGET:
        yp = np.exp(yp)
    return {'mape': mape(y_raw, yp), 'rmse': rmse(y_raw, yp)}


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(model_name, train_cv_mape, valid_mape, test2_mape, test3_mape):
    gap_tv  = train_cv_mape - valid_mape
    gap_vt2 = valid_mape    - test2_mape
    gap_vt3 = valid_mape    - test3_mape
    gap_tg2 = test2_mape    - TARGET
    gap_tg3 = test3_mape    - TARGET

    note_tv  = "(+) Overfitting suspected" if gap_tv  < 0 else ""
    note_vt2 = "(+) Generalization drop"   if gap_vt2 < 0 else ""
    note_vt3 = "(+) Generalization drop"   if gap_vt3 < 0 else ""

    w = 60
    print(f"\n{'=' * w}")
    print(f"  Model: {model_name}")
    print(f"{'=' * w}")
    print(f"{'구분':<26}| {'MAPE (%)':>9} | 비고")
    print(f"{'-' * 26}|{'-' * 11}|{'-' * 28}")
    print(f"{'Train (Batch 1 CV)':<26}| {train_cv_mape:>9.2f} |")
    print(f"{'Valid (Batch 1 Hold-out)':<26}| {valid_mape:>9.2f} |")
    print(f"{'Test  (Batch 2)':<26}| {test2_mape:>9.2f} |")
    print(f"{'Test  (Batch 3)':<26}| {test3_mape:>9.2f} |")
    print(f"{'-' * 26}|{'-' * 11}|{'-' * 28}")
    print(f"{'Gap (Train-Valid)':<26}| {gap_tv:>9.2f} | {note_tv}")
    print(f"{'Gap (Valid-Batch2)':<26}| {gap_vt2:>9.2f} | {note_vt2}")
    print(f"{'Gap (Valid-Batch3)':<26}| {gap_vt3:>9.2f} | {note_vt3}")
    print(f"{'Gap (Target-Batch2)':<26}| {gap_tg2:>9.2f} | Target = 9.1%")
    print(f"{'Gap (Target-Batch3)':<26}| {gap_tg3:>9.2f} | Target = 9.1%")
    print(f"{'=' * w}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading features.csv...")
    df = pd.read_csv(FEATURES_CSV)
    df1 = df[df['batch_name'] == BATCH1_NAME].copy()
    df2 = df[df['batch_name'] == BATCH2_NAME].copy()
    df3 = df[df['batch_name'] == BATCH3_NAME].copy()
    print(f"  Batch 1: {len(df1)} cells, Batch 2: {len(df2)} cells, Batch 3: {len(df3)} cells\n")

    print(f"Features: {SELECTED_FEATURES}\n")

    X1       = df1[SELECTED_FEATURES].values
    y1_raw   = df1['cycle_life'].values
    cell_ids = df1['cell_uid'].values
    X2       = df2[SELECTED_FEATURES].values
    y2_raw   = df2['cycle_life'].values
    X3       = df3[SELECTED_FEATURES].values
    y3_raw   = df3['cycle_life'].values

    y1 = np.log(y1_raw) if LOG_TARGET else y1_raw
    y2 = np.log(y2_raw) if LOG_TARGET else y2_raw

    X_train, X_valid, y_train, y_valid = cell_holdout_split(X1, y1, cell_ids)
    _, _, y_train_raw, y_valid_raw     = cell_holdout_split(X1, y1_raw, cell_ids)

    for label, factory in MODEL_REGISTRY.items():
        print(f"Running: {label} ...")

        cv_mape_score = cross_val_mape(factory(), X1, y1, groups=cell_ids)

        model_valid = factory()
        if label == 'XGBoost':
            train_xgboost(model_valid, X_train, y_train, X_valid, y_valid)
        else:
            model_valid.fit(X_train, y_train)
        valid_metrics = eval_with_inv(model_valid, X_valid, y_valid_raw)

        # Final model trained on full Batch 1, evaluated on Batch 2 and Batch 3
        model_final = factory()
        if label == 'XGBoost':
            train_xgboost(model_final, X1, y1, X2, y2)
        else:
            model_final.fit(X1, y1)
        test2_metrics = eval_with_inv(model_final, X2, y2_raw)
        test3_metrics = eval_with_inv(model_final, X3, y3_raw)

        print_results_table(
            model_name    = label,
            train_cv_mape = cv_mape_score,
            valid_mape    = valid_metrics['mape'],
            test2_mape    = test2_metrics['mape'],
            test3_mape    = test3_metrics['mape'],
        )
        print(f"  RMSE — Valid: {valid_metrics['rmse']:.1f}  |  "
              f"Batch2: {test2_metrics['rmse']:.1f}  |  Batch3: {test3_metrics['rmse']:.1f}\n")


if __name__ == '__main__':
    main()