"""
Ordinal regression metrics.

These metrics are designed for ordered categorical outcomes where
the distance between predictions matters.
"""

from collections.abc import Callable

import numpy as np

from jaxboost.metric.base import Metric


def _compute_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Compute Quadratic Weighted Kappa for ordinal classification.

    QWK measures agreement between two raters, penalizing disagreements
    quadratically by their distance. Perfect agreement = 1, random = 0.

    Args:
        y_true: True ordinal labels (0 to n_classes-1)
        y_pred: Predicted ordinal labels (0 to n_classes-1)
        n_classes: Number of ordinal classes

    Returns:
        QWK score in [-1, 1], higher is better
    """
    # Weight matrix: w_{i,j} = (i - j)^2 / (n_classes - 1)^2
    weights = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = (i - j) ** 2

    # Normalize weights
    if n_classes > 1:
        weights = weights / ((n_classes - 1) ** 2)

    # Confusion matrix
    conf_mat = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf_mat[t, p] += 1

    # Normalize confusion matrix
    n_samples = conf_mat.sum()
    if n_samples == 0:
        return 0.0
    conf_mat = conf_mat / n_samples

    # Expected matrix under random agreement
    hist_true = conf_mat.sum(axis=1)
    hist_pred = conf_mat.sum(axis=0)
    expected = np.outer(hist_true, hist_pred)

    # QWK = 1 - (weighted_error / expected_error)
    numerator = np.sum(weights * conf_mat)
    denominator = np.sum(weights * expected)

    if denominator < 1e-10:
        return 0.0

    return 1.0 - numerator / denominator


def qwk_metric(
    n_classes: int, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create Quadratic Weighted Kappa metric for ordinal regression.

    QWK penalizes predictions quadratically by distance from truth.
    Perfect agreement = 1, random agreement ≈ 0, worse than random < 0.

    Args:
        n_classes: Number of ordinal classes
        transform: Optional function to convert raw predictions to class labels.
                   If None, predictions are rounded and clipped.

    Returns:
        Metric object with .xgb_metric and .lgb_metric methods

    Example:
        >>> # With ordinal objective
        >>> ordinal = ordinal_logit(n_classes=6)
        >>> qwk = qwk_metric(n_classes=6, transform=ordinal.predict)
        >>>
        >>> model = xgb.train(
        ...     {'disable_default_eval_metric': 1},
        ...     dtrain, obj=ordinal.xgb_objective,
        ...     custom_metric=qwk.xgb_metric
        ... )
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        # Default: round and clip
        return np.clip(np.round(predt), 0, n_classes - 1).astype(int)

    def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return _compute_qwk(y_true, y_pred, n_classes)

    return Metric(
        name="qwk",
        fn=_qwk,
        transform=_transform,
        higher_is_better=True,
    )


def ordinal_mae_metric(
    n_classes: int, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create Mean Absolute Error metric for ordinal regression.

    Measures average distance between predicted and true class.

    Args:
        n_classes: Number of ordinal classes
        transform: Optional function to convert raw predictions to class labels

    Returns:
        Metric object

    Example:
        >>> mae = ordinal_mae_metric(n_classes=6)
        >>> model = xgb.train(params, dtrain, custom_metric=mae.xgb_metric)
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return np.clip(np.round(predt), 0, n_classes - 1).astype(int)

    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    return Metric(
        name="ordinal_mae",
        fn=_mae,
        transform=_transform,
        higher_is_better=False,
    )


def ordinal_accuracy_metric(
    n_classes: int, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create exact accuracy metric for ordinal regression.

    Measures proportion of exactly correct predictions.

    Args:
        n_classes: Number of ordinal classes
        transform: Optional function to convert raw predictions to class labels

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return np.clip(np.round(predt), 0, n_classes - 1).astype(int)

    def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true.astype(int) == y_pred.astype(int))

    return Metric(
        name="ordinal_acc",
        fn=_acc,
        transform=_transform,
        higher_is_better=True,
    )


def adjacent_accuracy_metric(
    n_classes: int, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create adjacent accuracy metric (within ±1) for ordinal regression.

    Measures proportion of predictions within 1 class of truth.
    Useful when exact prediction is difficult but close is acceptable.

    Args:
        n_classes: Number of ordinal classes
        transform: Optional function to convert raw predictions to class labels

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return np.clip(np.round(predt), 0, n_classes - 1).astype(int)

    def _adj_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred) <= 1)

    return Metric(
        name="adj_acc",
        fn=_adj_acc,
        transform=_transform,
        higher_is_better=True,
    )
