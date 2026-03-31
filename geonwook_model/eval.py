"""
eval.py — End-to-end experiment runner with minimal local dependencies.

Imports only:
    feature.py → dataset loading, feature selection
    model.py   → model factory, training, evaluation, persistence
"""

from __future__ import annotations

from pathlib import Path

from feature import get_features, load_feature_dataset
from model import (
    cell_holdout_split,
    cross_val_mape,
    evaluate,
    fit_model,
    fit_xgboost,
    save_model,
    get_model,
)

RANDOM_SEED = 42
TARGET = 9.1

FEATURES_CSV = Path(__file__).parent.parent / 'preprocessed_data' / 'features.csv'
TRAIN_BATCH = '2017-05-12'
TEST_BATCH = '2018-02-20'
EXTRA_BATCHES = ['2018-04-12']
SELECTED_FEATURES = ['delta_q_min', 'log_delta_q_var', 'qdlin_010_var', 'delta_q_var']


def print_results_table(
    model_name: str,
    train_cv_mape: float,
    valid_mape: float,
    test_mape: float,
) -> None:
    gap_tv = train_cv_mape - valid_mape
    gap_vt = valid_mape - test_mape
    gap_tg = test_mape - TARGET

    note_tv = '(+) Overfitting suspected' if gap_tv < 0 else ''
    note_vt = '(+) Generalization drop' if gap_vt < 0 else ''
    note_tg = 'Target = 9.1%'

    w = 60
    print(f"\n{'=' * w}")
    print(f'  Model: {model_name}')
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


def main() -> None:
    print('Loading features.csv...')
    datasets = load_feature_dataset(
        features_csv=FEATURES_CSV,
        train_batch=TRAIN_BATCH,
        test_batch=TEST_BATCH,
        extra_batches=EXTRA_BATCHES,
    )
    df_train = datasets['train']
    df_test = datasets['test']
    print(
        f"  Train: {len(df_train)} cells, Test: {len(df_test)} cells, "
        + ', '.join(f'{batch}: {len(datasets[batch])} cells' for batch in EXTRA_BATCHES)
    )

    feature_cols = get_features(
        df_train.drop(columns=['cell_uid', 'batch_name', 'cycle_life', 'charging_policy'], errors='ignore'),
        selected_features=SELECTED_FEATURES,
    )
    print(f'Features used: {feature_cols}\n')

    X_train_full = df_train[feature_cols].values
    y_train_full = df_train['cycle_life'].values
    cell_ids = df_train['cell_uid'].values

    X_test = df_test[feature_cols].values
    y_test = df_test['cycle_life'].values

    X_train, X_valid, y_train, y_valid = cell_holdout_split(
        X_train_full, y_train_full, cell_ids, val_ratio=0.2, seed=RANDOM_SEED
    )

    for model_name in ['linear', 'elastic_net', 'xgboost']:
        print(f'Running: {model_name} ...')

        cv_mape = cross_val_mape(
            get_model(model_name), X_train_full, y_train_full, groups=cell_ids, n_splits=5
        )

        model_valid = get_model(model_name)
        if model_name == 'xgboost':
            fit_xgboost(model_valid, X_train, y_train, X_valid, y_valid)
        else:
            fit_model(model_valid, X_train, y_train)
        valid_metrics = evaluate(model_valid, X_valid, y_valid)

        model_final = get_model(model_name)
        if model_name == 'xgboost':
            fit_xgboost(model_final, X_train_full, y_train_full, X_test, y_test)
        else:
            fit_model(model_final, X_train_full, y_train_full)
        test_metrics = evaluate(model_final, X_test, y_test)

        save_model(model_final, name=f'{model_name}_final')

        label_map = {'linear': 'Linear', 'elastic_net': 'Elastic Net', 'xgboost': 'XGBoost'}
        print_results_table(
            model_name=label_map[model_name],
            train_cv_mape=cv_mape,
            valid_mape=valid_metrics['mape'],
            test_mape=test_metrics['mape'],
        )

        for batch_name in EXTRA_BATCHES:
            df_extra = datasets[batch_name]
            extra_metrics = evaluate(
                model_final,
                df_extra[feature_cols].values,
                df_extra['cycle_life'].values,
            )
            print(
                f'  {batch_name} — MAPE: {extra_metrics["mape"]:.2f}%  '
                f'RMSE: {extra_metrics["rmse"]:.1f}'
            )

        print(f'  RMSE — Valid: {valid_metrics["rmse"]:.1f}  |  Test: {test_metrics["rmse"]:.1f}\n')


if __name__ == '__main__':
    main()
