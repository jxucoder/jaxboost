"""
Regression metrics.
"""

from collections.abc import Callable

import numpy as np

from jaxboost.metric.base import Metric


def mse_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create Mean Squared Error metric.

    Args:
        transform: Optional function to transform raw predictions

    Returns:
        Metric object

    Example:
        >>> model = xgb.train(
        ...     {'disable_default_eval_metric': 1},
        ...     dtrain, obj=my_objective.xgb_objective,
        ...     custom_metric=mse_metric().xgb_metric
        ... )
    """

    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    return Metric(
        name="mse",
        fn=_mse,
        transform=transform,
        higher_is_better=False,
    )


def rmse_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create Root Mean Squared Error metric.

    Args:
        transform: Optional function to transform raw predictions

    Returns:
        Metric object
    """

    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    return Metric(
        name="rmse",
        fn=_rmse,
        transform=transform,
        higher_is_better=False,
    )


def mae_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create Mean Absolute Error metric.

    Args:
        transform: Optional function to transform raw predictions

    Returns:
        Metric object
    """

    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    return Metric(
        name="mae",
        fn=_mae,
        transform=transform,
        higher_is_better=False,
    )


def r2_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create R² (coefficient of determination) metric.

    R² = 1 - SS_res / SS_tot

    Args:
        transform: Optional function to transform raw predictions

    Returns:
        Metric object
    """

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot < 1e-10:
            return 0.0
        return 1 - ss_res / ss_tot

    return Metric(
        name="r2",
        fn=_r2,
        transform=transform,
        higher_is_better=True,
    )
