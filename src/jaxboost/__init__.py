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
    - Regression: mse, huber, quantile, tweedie, asymmetric, log_cosh
    - Multi-class: softmax_cross_entropy, focal_multiclass, label_smoothing
    - Survival: cox_partial_likelihood, aft, weibull_aft
    - Multi-task: multi_task_regression, multi_task_classification
"""

from jaxboost._version import __version__

# =============================================================================
# Core: Auto-Objective
# =============================================================================
# =============================================================================
# Built-in Objectives: Binary Classification
# =============================================================================
# =============================================================================
# Built-in Objectives: Regression
# =============================================================================
# =============================================================================
# Built-in Objectives: Multi-class Classification
# =============================================================================
# =============================================================================
# Built-in Objectives: Survival Analysis
# =============================================================================
# =============================================================================
# Built-in Objectives: Multi-task Learning
# =============================================================================
# =============================================================================
# Built-in Objectives: Multi-output (Uncertainty)
# =============================================================================
from jaxboost.objective import (
    # Core decorator
    AutoObjective,
    # Multi-task
    MaskedMultiTaskObjective,
    # Multi-class/multi-output variants
    MultiClassObjective,
    MultiOutputObjective,
    aft,
    asymmetric,
    auto_objective,
    binary_crossentropy,
    class_balanced,
    focal_loss,
    focal_multiclass,
    gaussian_nll,
    hinge_loss,
    huber,
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
    pseudo_huber,
    quantile,
    softmax_cross_entropy,
    tweedie,
    weibull_aft,
    weighted_binary_crossentropy,
)

__all__ = [
    "__version__",
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
    # Multi-class
    "softmax_cross_entropy",
    "focal_multiclass",
    "label_smoothing",
    "class_balanced",
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
