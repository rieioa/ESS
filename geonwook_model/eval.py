from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from geonwook_model.feature import BATCH1, BATCH2, BATCH3, FEATURE_COLUMNS, build_dataset
from geonwook_model.model import LOG_TARGET, get_model


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def fit_predict(model, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    if LOG_TARGET:
        model.fit(X_train, np.log(y_train))
        return np.exp(model.predict(X_eval))
    model.fit(X_train, y_train)
    return model.predict(X_eval)


def evaluate(train_batch: str = BATCH1, test_batches: tuple[str, ...] = (BATCH2, BATCH3)) -> None:
    train_df = build_dataset(train_batch)
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)

    model = get_model()
    train_pred = fit_predict(model, X_train, y_train, X_train)

    print("=== geonwook_model evaluation ===")
    print(f"train_batch={train_batch} rows={len(train_df)}")
    print(f"features={FEATURE_COLUMNS}")
    print(f"log_target={LOG_TARGET}")
    print(f"train_mape={mape(y_train, train_pred):.6f}")

    for batch_name in test_batches:
        test_df = build_dataset(batch_name)
        X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        y_test = test_df["cycle_life"].to_numpy(dtype=float)
        test_pred = fit_predict(get_model(), X_train, y_train, X_test)
        print(
            f"test_batch={batch_name} rows={len(test_df)} "
            f"mape={mape(y_test, test_pred):.6f}"
        )


if __name__ == "__main__":
    evaluate()
