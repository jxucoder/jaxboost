"""
Metrics for bounded regression (proportion/rate prediction in [0, 1]).
"""

from collections.abc import Callable

import numpy as np

from jaxboost.metric.base import Metric


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def bounded_mse_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create MSE metric for bounded regression.

    By default, applies sigmoid to transform logits to [0, 1].

    Args:
        transform: Optional function to transform raw predictions to [0, 1].
                   If None, sigmoid is applied.

    Returns:
        Metric object

    Example:
        >>> # For bounded regression with sigmoid link
        >>> model = xgb.train(
        ...     {'disable_default_eval_metric': 1},
        ...     dtrain, obj=soft_ce.xgb_objective,
        ...     custom_metric=bounded_mse_metric().xgb_metric
        ... )
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return _sigmoid(predt)

    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    return Metric(
        name="bounded_mse",
        fn=_mse,
        transform=_transform,
        higher_is_better=False,
    )


def out_of_bounds_metric(lower: float = 0.0, upper: float = 1.0) -> Metric:
    """
    Create metric to measure proportion of predictions outside valid bounds.

    Useful for comparing bounded vs unbounded regression approaches.

    Args:
        lower: Lower bound (default 0.0)
        upper: Upper bound (default 1.0)

    Returns:
        Metric object

    Example:
        >>> # Check how many predictions fall outside [0, 1]
        >>> model = xgb.train(
        ...     {'disable_default_eval_metric': 1},
        ...     dtrain, obj=mse.xgb_objective,
        ...     custom_metric=out_of_bounds_metric().xgb_metric
        ... )
    """

    def _oob(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_pred < lower) | (y_pred > upper))

    return Metric(
        name="oob_rate",
        fn=_oob,
        transform=None,  # Don't transform - we want to see raw predictions
        higher_is_better=False,
    )
