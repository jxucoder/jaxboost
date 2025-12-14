"""
XGBoost/LightGBM objective function generator using JAX automatic differentiation.

This module provides tools to automatically generate gradient and Hessian
functions for custom loss functions, eliminating the need for manual derivation.

Quick Start:
    >>> from jaxboost.objective import auto_objective, focal_loss
    >>>
    >>> # Use built-in objective
    >>> model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)
    >>>
    >>> # Create custom objective
    >>> @auto_objective
    ... def my_loss(y_pred, y_true, alpha=0.5):
    ...     return alpha * (y_pred - y_true) ** 2
    >>>
    >>> model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
"""

# Core classes and decorators
from jaxboost.objective.auto import (
    AutoObjective,
    auto_objective,
)
from jaxboost.objective.multiclass import (
    MultiClassObjective,
    multiclass_objective,
)
from jaxboost.objective.multi_output import (
    MultiOutputObjective,
    multi_output_objective,
)

# Binary classification objectives
from jaxboost.objective.binary import (
    binary_crossentropy,
    focal_loss,
    hinge_loss,
    weighted_binary_crossentropy,
)

# Regression objectives
from jaxboost.objective.regression import (
    asymmetric,
    huber,
    log_cosh,
    mae_smooth,
    mse,
    pseudo_huber,
    quantile,
    tweedie,
)

# Multi-class classification objectives
from jaxboost.objective.multiclass import (
    class_balanced,
    focal_multiclass,
    label_smoothing,
    softmax_cross_entropy,
)

# Survival analysis objectives
from jaxboost.objective.survival import (
    aft,
    cox_partial_likelihood,
    interval_regression,
    weibull_aft,
)

# Multi-output objectives
from jaxboost.objective.multi_output import (
    gaussian_nll,
    laplace_nll,
)

__all__ = [
    # Core
    "AutoObjective",
    "auto_objective",
    "MultiClassObjective",
    "multiclass_objective",
    "MultiOutputObjective",
    "multi_output_objective",
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
    "interval_regression",
    "cox_partial_likelihood",
    "weibull_aft",
    # Multi-output
    "gaussian_nll",
    "laplace_nll",
]

