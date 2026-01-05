"""
jaxboost: JAX autodiff for XGBoost/LightGBM objectives.

Write a loss function, get gradients and Hessians automatically via JAX.
Works with XGBoost, LightGBM, and CatBoost.

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
from jaxboost.objective import (
    # Core decorator
    AutoObjective,
    auto_objective,
    # Multi-class/multi-output variants
    MultiClassObjective,
    multiclass_objective,
    MultiOutputObjective,
    multi_output_objective,
    # Multi-task
    MaskedMultiTaskObjective,
    masked_multi_task_objective,
)

# =============================================================================
# Built-in Objectives: Binary Classification
# =============================================================================
from jaxboost.objective import (
    focal_loss,
    binary_crossentropy,
    weighted_binary_crossentropy,
    hinge_loss,
)

# =============================================================================
# Built-in Objectives: Regression
# =============================================================================
from jaxboost.objective import (
    mse,
    huber,
    quantile,
    tweedie,
    asymmetric,
    log_cosh,
    pseudo_huber,
    mae_smooth,
)

# =============================================================================
# Built-in Objectives: Multi-class Classification
# =============================================================================
from jaxboost.objective import (
    softmax_cross_entropy,
    focal_multiclass,
    label_smoothing,
    class_balanced,
)

# =============================================================================
# Built-in Objectives: Survival Analysis
# =============================================================================
from jaxboost.objective import (
    cox_partial_likelihood,
    aft,
    weibull_aft,
    interval_regression,
)

# =============================================================================
# Built-in Objectives: Multi-task Learning
# =============================================================================
from jaxboost.objective import (
    multi_task_regression,
    multi_task_classification,
    multi_task_huber,
    multi_task_quantile,
)

# =============================================================================
# Built-in Objectives: Multi-output (Uncertainty)
# =============================================================================
from jaxboost.objective import (
    gaussian_nll,
    laplace_nll,
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
    "cox_partial_likelihood",
    "aft",
    "weibull_aft",
    "interval_regression",
    # Multi-task
    "multi_task_regression",
    "multi_task_classification",
    "multi_task_huber",
    "multi_task_quantile",
    # Multi-output
    "gaussian_nll",
    "laplace_nll",
]
