"""
XGBoost/LightGBM evaluation metrics using JAXBoost.

This module provides evaluation metrics that work with custom objectives.
When using custom objectives, XGBoost's default metrics may not be meaningful.
Use these metrics to properly monitor training.

Quick Start:
    >>> from jaxboost.metric import qwk_metric, mae_metric
    >>> from jaxboost.objective import ordinal_logit
    >>>
    >>> # Create ordinal objective with built-in metric
    >>> ordinal = ordinal_logit(n_classes=6)
    >>> ordinal.init_thresholds_from_data(y_train)
    >>>
    >>> # Train with custom metric
    >>> model = xgb.train(
    ...     {'disable_default_eval_metric': 1, ...},
    ...     dtrain,
    ...     obj=ordinal.xgb_objective,
    ...     custom_metric=ordinal.qwk_metric,
    ...     evals=[(dtest, 'test')]
    ... )

Important:
    When using custom objectives, always set `'disable_default_eval_metric': 1`
    in XGBoost params to suppress meaningless default metrics.

    For LightGBM, set `'metric': 'None'` instead.
"""

from jaxboost.metric.base import (
    Metric,
    make_metric,
)
from jaxboost.metric.bounded import (
    bounded_mse_metric,
    out_of_bounds_metric,
)
from jaxboost.metric.classification import (
    accuracy_metric,
    auc_metric,
    f1_metric,
    log_loss_metric,
    precision_metric,
    recall_metric,
)
from jaxboost.metric.ordinal import (
    adjacent_accuracy_metric,
    ordinal_accuracy_metric,
    ordinal_mae_metric,
    qwk_metric,
)
from jaxboost.metric.regression import (
    mae_metric,
    mse_metric,
    r2_metric,
    rmse_metric,
)

__all__ = [
    # Base
    "Metric",
    "make_metric",
    # Ordinal
    "qwk_metric",
    "ordinal_mae_metric",
    "ordinal_accuracy_metric",
    "adjacent_accuracy_metric",
    # Classification
    "auc_metric",
    "log_loss_metric",
    "accuracy_metric",
    "f1_metric",
    "precision_metric",
    "recall_metric",
    # Regression
    "mse_metric",
    "rmse_metric",
    "mae_metric",
    "r2_metric",
    # Bounded
    "bounded_mse_metric",
    "out_of_bounds_metric",
]
