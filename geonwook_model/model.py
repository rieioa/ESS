from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["DeltaQ_var", "charge_time_avg", "temp_integral"]
RIDGE_ALPHA = 1.0
LOG_TARGET = True


def get_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=RIDGE_ALPHA, random_state=42)),
        ]
    )
