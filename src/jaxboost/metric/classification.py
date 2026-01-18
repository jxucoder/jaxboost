"""
Classification metrics for binary and multi-class problems.
"""

from collections.abc import Callable

import numpy as np

from jaxboost.metric.base import Metric


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve.

    Uses the Mann-Whitney U statistic formulation for efficiency.
    """
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Get unique thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate(([0], distinct_value_indices + 1, [len(y_true)]))

    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true)[threshold_idxs[:-1]]
    fps = np.cumsum(1 - y_true)[threshold_idxs[:-1]]

    tps = np.concatenate(([0], tps))
    fps = np.concatenate(([0], fps))

    # Avoid division by zero
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Compute AUC using trapezoidal rule
    return np.trapezoid(tpr, fpr)


def auc_metric(transform: Callable[[np.ndarray], np.ndarray] | None = None) -> Metric:
    """
    Create Area Under ROC Curve metric for binary classification.

    Args:
        transform: Optional function to convert raw predictions to probabilities.
                   If None, sigmoid is applied.

    Returns:
        Metric object with .xgb_metric and .lgb_metric methods

    Example:
        >>> from jaxboost.metric import auc_metric
        >>>
        >>> model = xgb.train(
        ...     {'disable_default_eval_metric': 1},
        ...     dtrain, obj=focal_loss.xgb_objective,
        ...     custom_metric=auc_metric().xgb_metric
        ... )
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return _sigmoid(predt)

    return Metric(
        name="auc",
        fn=_compute_auc,
        transform=_transform,
        higher_is_better=True,
    )


def log_loss_metric(
    transform: Callable[[np.ndarray], np.ndarray] | None = None, eps: float = 1e-7
) -> Metric:
    """
    Create Log Loss (binary cross-entropy) metric.

    Args:
        transform: Optional function to convert raw predictions to probabilities
        eps: Small value for numerical stability

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        if transform is not None:
            return transform(predt)
        return _sigmoid(predt)

    def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return Metric(
        name="logloss",
        fn=_log_loss,
        transform=_transform,
        higher_is_better=False,
    )


def accuracy_metric(
    threshold: float = 0.5, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create accuracy metric for binary classification.

    Args:
        threshold: Classification threshold (default 0.5)
        transform: Optional function to convert raw predictions to probabilities

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        probs = transform(predt) if transform is not None else _sigmoid(predt)
        return (probs >= threshold).astype(int)

    def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true.astype(int) == y_pred.astype(int))

    return Metric(
        name="accuracy",
        fn=_acc,
        transform=_transform,
        higher_is_better=True,
    )


def f1_metric(
    threshold: float = 0.5, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create F1 score metric for binary classification.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        threshold: Classification threshold
        transform: Optional function to convert raw predictions to probabilities

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        probs = transform(predt) if transform is not None else _sigmoid(predt)
        return (probs >= threshold).astype(int)

    def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return Metric(
        name="f1",
        fn=_f1,
        transform=_transform,
        higher_is_better=True,
    )


def precision_metric(
    threshold: float = 0.5, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create precision metric for binary classification.

    Precision = TP / (TP + FP)

    Args:
        threshold: Classification threshold
        transform: Optional function to convert raw predictions to probabilities

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        probs = transform(predt) if transform is not None else _sigmoid(predt)
        return (probs >= threshold).astype(int)

    def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return Metric(
        name="precision",
        fn=_precision,
        transform=_transform,
        higher_is_better=True,
    )


def recall_metric(
    threshold: float = 0.5, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> Metric:
    """
    Create recall metric for binary classification.

    Recall = TP / (TP + FN)

    Args:
        threshold: Classification threshold
        transform: Optional function to convert raw predictions to probabilities

    Returns:
        Metric object
    """

    def _transform(predt: np.ndarray) -> np.ndarray:
        probs = transform(predt) if transform is not None else _sigmoid(predt)
        return (probs >= threshold).astype(int)

    def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return Metric(
        name="recall",
        fn=_recall,
        transform=_transform,
        higher_is_better=True,
    )
