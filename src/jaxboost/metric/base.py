"""
Base utilities for creating XGBoost/LightGBM compatible metrics.

XGBoost metric interface:
    def metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]

LightGBM metric interface:
    def metric(preds: np.ndarray, eval_data: lgb.Dataset) -> Tuple[str, float, bool]
"""

from collections.abc import Callable
from typing import Any

import numpy as np


class Metric:
    """
    Base class for XGBoost/LightGBM evaluation metrics.

    Provides both XGBoost and LightGBM compatible interfaces.

    Args:
        name: Metric name displayed during training
        fn: Metric function (y_true, y_pred) -> float
        transform: Optional prediction transform (e.g., sigmoid for binary)
        higher_is_better: Whether higher metric values are better

    Example:
        >>> metric = Metric(
        ...     name='accuracy',
        ...     fn=lambda y, p: (y == p).mean(),
        ...     transform=lambda p: (p > 0.5).astype(int),
        ...     higher_is_better=True
        ... )
        >>>
        >>> # Use with XGBoost
        >>> model = xgb.train(params, dtrain, custom_metric=metric.xgb_metric)
        >>>
        >>> # Use with LightGBM
        >>> model = lgb.train(params, train_data, feval=metric.lgb_metric)
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[np.ndarray, np.ndarray], float],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        higher_is_better: bool = True,
    ):
        self.name = name
        self.fn = fn
        self.transform = transform
        self.higher_is_better = higher_is_better

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric value."""
        if self.transform is not None:
            y_pred = self.transform(y_pred)
        return float(self.fn(y_true, y_pred))

    def xgb_metric(self, predt: np.ndarray, dtrain: Any) -> tuple[str, float]:
        """
        XGBoost-compatible metric function.

        Args:
            predt: Raw predictions from model
            dtrain: XGBoost DMatrix with labels

        Returns:
            (metric_name, metric_value)
        """
        y_true = dtrain.get_label()
        y_pred = self.transform(predt) if self.transform else predt
        value = self.fn(y_true, y_pred)
        return self.name, float(value)

    def lgb_metric(self, preds: np.ndarray, eval_data: Any) -> tuple[str, float, bool]:
        """
        LightGBM-compatible metric function.

        Args:
            preds: Raw predictions from model
            eval_data: LightGBM Dataset with labels

        Returns:
            (metric_name, metric_value, is_higher_better)
        """
        y_true = eval_data.get_label()
        y_pred = self.transform(preds) if self.transform else preds
        value = self.fn(y_true, y_pred)
        return self.name, float(value), self.higher_is_better

    def __repr__(self) -> str:
        return f"Metric(name='{self.name}', higher_is_better={self.higher_is_better})"


def make_metric(
    name: str,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    higher_is_better: bool = True,
) -> Callable:
    """
    Decorator to create XGBoost/LightGBM compatible metrics.

    Args:
        name: Metric name shown during training
        transform: Optional function to transform raw predictions
        higher_is_better: Whether higher values are better

    Example:
        >>> @make_metric('my_accuracy', transform=lambda p: (p > 0.5).astype(int))
        ... def my_accuracy(y_true, y_pred):
        ...     return (y_true == y_pred).mean()
        >>>
        >>> # Use with XGBoost
        >>> model = xgb.train(params, dtrain, custom_metric=my_accuracy.xgb_metric)

    Returns:
        Decorated function that has .xgb_metric and .lgb_metric attributes
    """

    def decorator(fn: Callable[[np.ndarray, np.ndarray], float]) -> Metric:
        return Metric(
            name=name,
            fn=fn,
            transform=transform,
            higher_is_better=higher_is_better,
        )

    return decorator
