"""
model.py — Model definitions only.

Responsibilities:
    - Define and return unfitted model instances
    - No training, evaluation, or feature logic here

Available models:
    - elastic_net : Pipeline(StandardScaler → ElasticNet)
    - xgboost     : XGBRegressor
"""

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

RANDOM_SEED = 42


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


def get_xgboost() -> XGBRegressor:
    """
    XGBRegressor tuned conservatively for small battery datasets (~40 cells).
    Shallow trees + regularization reduce overfitting on low-sample regimes.

    n_estimators is set high (1000) as an upper bound.
    train_xgboost() in train.py applies early stopping to find the
    optimal number of rounds automatically using a held-out eval set.
    """
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
