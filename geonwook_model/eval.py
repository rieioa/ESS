from __future__ import annotations

try:
    from .feature import FEATURE_COLUMNS, build_datasets
    from .model import fit_model, get_model, mape, predict, save_model
except ImportError:  # pragma: no cover
    from feature import FEATURE_COLUMNS, build_datasets
    from model import fit_model, get_model, mape, predict, save_model


def main() -> None:
    train_df, test2_df, test3_df = build_datasets()

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    X_test2 = test2_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test2 = test2_df["cycle_life"].to_numpy(dtype=float)
    X_test3 = test3_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test3 = test3_df["cycle_life"].to_numpy(dtype=float)

    model = fit_model(get_model(alpha=1.0), X_train, y_train)
    pred2 = predict(model, X_test2)
    pred3 = predict(model, X_test3)
    model_path = save_model(model)

    print("=== geonwook_model evaluation ===")
    print(f"features={'|'.join(FEATURE_COLUMNS)}")
    print("model=Ridge(alpha=1.0, log_target=True)")
    print(f"train_rows={len(train_df)}")
    print(f"test2_rows={len(test2_df)}")
    print(f"test3_rows={len(test3_df)}")
    print(f"test2_mape={mape(y_test2, pred2):.6f}")
    print(f"test3_mape={mape(y_test3, pred3):.6f}")
    print(f"saved_model={model_path}")


if __name__ == "__main__":
    main()
