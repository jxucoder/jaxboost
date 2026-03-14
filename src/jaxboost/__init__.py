"""
jaxboost: JAX autodiff for XGBoost/LightGBM objectives.

Write a loss function, get gradients and Hessians automatically via JAX.
Works with XGBoost and LightGBM.

Quick Start:
    >>> import xgboost as xgb
    >>> from jaxboost import auto_objective, focal_loss, huber, quantile
    >>>
    >>> # Use built-in objectives
    >>> model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)
    >>> model = xgb.train(params, dtrain, obj=huber.xgb_objective)
    >>> model = xgb.train(params, dtrain, obj=quantile(0.9).xgb_objective)
    >>>
    >>> # Custom objective - just write the loss, autodiff handles the rest
    >>> @auto_objective
    ... def my_loss(y_pred, y_true):
    ...     return (y_pred - y_true) ** 2
    >>>
    >>> model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)

Available Objectives:
    - Binary: focal_loss, binary_crossentropy, hinge_loss
    - Regression: mse, huber, quantile, tweedie, asymmetric, log_cosh, poisson, gamma
    - Multi-class: softmax_cross_entropy, focal_multiclass, label_smoothing
    - Ordinal: ordinal_logit, ordinal_probit, qwk_ordinal, sord_objective, slace_objective
    - Survival: aft, weibull_aft
    - Multi-task: multi_task_regression, multi_task_classification
"""

# Metrics module
from jaxboost import metric
from jaxboost._version import __version__
from jaxboost.objective import (
    # Core decorator
    AutoObjective,
    # Multi-task
    MaskedMultiTaskObjective,
    # Multi-class/multi-output variants
    MultiClassObjective,
    MultiOutputObjective,
    # Ordinal regression
    OLLObjective,
    OrdinalObjective,
    QWKOrdinalObjective,
    SLACEObjective,
    SORDObjective,
    SquaredCDFObjective,
    aft,
    asymmetric,
    auto_objective,
    binary_crossentropy,
    class_balanced,
    focal_loss,
    focal_multiclass,
    gamma,
    gaussian_nll,
    hinge_loss,
    huber,
    hybrid_ordinal,
    label_smoothing,
    laplace_nll,
    log_cosh,
    mae_smooth,
    masked_multi_task_objective,
    mse,
    multi_output_objective,
    multi_task_classification,
    multi_task_huber,
    multi_task_quantile,
    multi_task_regression,
    multiclass_objective,
    oll_objective,
    ordinal_logit,
    ordinal_probit,
    ordinal_regression,
    poisson,
    pseudo_huber,
    quantile,
    qwk_ordinal,
    slace_objective,
    softmax_cross_entropy,
    sord_objective,
    squared_cdf_ordinal,
    tweedie,
    weibull_aft,
    weighted_binary_crossentropy,
)

__all__ = [
    "__version__",
    "metric",
    # Core
    "AutoObjective",
    "auto_objective",
    "MultiClassObjective",
    "multiclass_objective",
    "MultiOutputObjective",
    "multi_output_objective",
    "MaskedMultiTaskObjective",
    "masked_multi_task_objective",
    # Binary classification
    "focal_loss",
    "binary_crossentropy",
    "weighted_binary_crossentropy",
    "hinge_loss",
    # Regression
    "mse",
    "huber",
    "quantile",
    "tweedie",
    "asymmetric",
    "log_cosh",
    "pseudo_huber",
    "mae_smooth",
    "poisson",
    "gamma",
    # Multi-class
    "softmax_cross_entropy",
    "focal_multiclass",
    "label_smoothing",
    "class_balanced",
    # Ordinal regression
    "OrdinalObjective",
    "QWKOrdinalObjective",
    "SquaredCDFObjective",
    "SORDObjective",
    "OLLObjective",
    "SLACEObjective",
    "ordinal_regression",
    "ordinal_probit",
    "ordinal_logit",
    "qwk_ordinal",
    "hybrid_ordinal",
    "squared_cdf_ordinal",
    "sord_objective",
    "oll_objective",
    "slace_objective",
    # Survival
    "aft",
    "weibull_aft",
    # Multi-task
    "multi_task_regression",
    "multi_task_classification",
    "multi_task_huber",
    "multi_task_quantile",
    # Multi-output
    "gaussian_nll",
    "laplace_nll",
]
