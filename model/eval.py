from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

try:
    from .feature import FEATURE_COLUMNS, build_datasets
    from .model import fit_model, get_model, mape, predict, save_model
except ImportError:  # pragma: no cover
    from feature import FEATURE_COLUMNS, build_datasets
    from model import fit_model, get_model, mape, predict, save_model

RANDOM_SEED = 42
TARGET = 9.1


def cross_val_mape(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> float:
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], np.log(y[train_idx]))
        y_pred = np.exp(m.predict(X[val_idx]))
        scores.append(float(np.mean(np.abs((y[val_idx] - y_pred) / y[val_idx])) * 100))
    return float(np.mean(scores))


def cell_holdout_split(X, y, cell_ids, val_ratio=0.2):
    rng = np.random.default_rng(RANDOM_SEED)
    unique_cells = np.unique(cell_ids)
    rng.shuffle(unique_cells)
    n_val = max(1, int(len(unique_cells) * val_ratio))
    val_cells = set(unique_cells[-n_val:])
    val_mask = np.array([c in val_cells for c in cell_ids])
    return X[~val_mask], X[val_mask], y[~val_mask], y[val_mask]


def print_results(
    model_name: str,
    train_cv_mape: float,
    valid_mape: float,
    test2_mape: float,
    test3_mape: float,
) -> None:
    gap_tv = train_cv_mape - valid_mape
    gap_vt = valid_mape - test2_mape
    gap_tg2 = test2_mape - TARGET
    gap_b23 = test2_mape - test3_mape
    gap_tg3 = test3_mape - TARGET

    note_tv  = "(+) 과적합 의심"           if gap_tv < 0 else ""
    note_vt  = "(+) 배치간 일반화 저하 의심" if gap_vt < 0 else ""

    w = 62
    print(f"\n{'=' * w}")
    print(f"  Model: {model_name}")
    print(f"{'=' * w}")
    print(f"{'구분':<28}| {'MAPE (%)':>9} | 비고")
    print(f"{'-' * 28}|{'-' * 11}|{'-' * 28}")
    print(f"{'Train (Batch 1 CV)':<28}| {train_cv_mape:>9.2f} |")
    print(f"{'Valid (Batch 1 Hold-out)':<28}| {valid_mape:>9.2f} |")
    print(f"{'Test (Batch 2)':<28}| {test2_mape:>9.2f} |")
    print(f"{'-' * 28}|{'-' * 11}|{'-' * 28}")
    print(f"{'  Gap (Train-Valid)':<28}| {gap_tv:>9.2f} | {note_tv}")
    print(f"{'  Gap (Valid-Test)':<28}| {gap_vt:>9.2f} | {note_vt}")
    print(f"{'  Gap (Target-Test)':<28}| {gap_tg2:>9.2f} | Target = {TARGET}%")
    print(f"{'-' * 28}|{'-' * 11}|{'-' * 28}")
    print(f"{'Test (Batch 3)':<28}| {test3_mape:>9.2f} |")
    print(f"{'-' * 28}|{'-' * 11}|{'-' * 28}")
    print(f"{'  Gap (Batch2-Batch3)':<28}| {gap_b23:>9.2f} | Test 성능 간 비교")
    print(f"{'  Gap (Target-Test)':<28}| {gap_tg3:>9.2f} | Target = {TARGET}%")
    print(f"{'=' * w}\n")


def main() -> None:
    train_df, test2_df, test3_df = build_datasets()

    X_train  = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train  = train_df["cycle_life"].to_numpy(dtype=float)
    cell_ids = train_df["cell_uid"].to_numpy()
    X_test2  = test2_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test2  = test2_df["cycle_life"].to_numpy(dtype=float)
    X_test3  = test3_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test3  = test3_df["cycle_life"].to_numpy(dtype=float)

    model = get_model(alpha=1.0)

    # CV
    cv_mape = cross_val_mape(model, X_train, y_train, groups=cell_ids, n_splits=5)

    # Hold-out
    X_tr, X_val, y_tr, y_val = cell_holdout_split(X_train, y_train, cell_ids)
    m_valid = clone(model)
    fit_model(m_valid, X_tr, y_tr)
    valid_mape = mape(y_val, predict(m_valid, X_val))

    # Final model on full B1
    m_final = clone(model)
    fit_model(m_final, X_train, y_train)
    test2_mape = mape(y_test2, predict(m_final, X_test2))
    test3_mape = mape(y_test3, predict(m_final, X_test3))

    save_model(m_final)

    print(f"  train_rows={len(train_df)}  test2_rows={len(test2_df)}  test3_rows={len(test3_df)}")
    print(f"  features={FEATURE_COLUMNS}")
    print_results(
        model_name    = "Ridge(alpha=1.0, log_target=True)",
        train_cv_mape = cv_mape,
        valid_mape    = valid_mape,
        test2_mape    = test2_mape,
        test3_mape    = test3_mape,
    )


if __name__ == "__main__":
    main()
