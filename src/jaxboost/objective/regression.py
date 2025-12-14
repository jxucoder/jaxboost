"""
Regression objective functions.

All objectives are pre-wrapped with AutoObjective and ready to use
with XGBoost/LightGBM.
"""

import jax
import jax.numpy as jnp

from jaxboost.objective.auto import AutoObjective


@AutoObjective
def mse(
    y_pred: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """
    Mean Squared Error Loss.

    Standard squared error loss. Included for testing and as a baseline.

    Args:
        y_pred: Predicted value
        y_true: True value

    Example:
        >>> model = xgb.train(params, dtrain, obj=mse.xgb_objective)
    """
    return (y_pred - y_true) ** 2


@AutoObjective
def huber(
    y_pred: jax.Array,
    y_true: jax.Array,
    delta: float = 1.0,
) -> jax.Array:
    """
    Huber Loss for robust regression.

    Combines MSE for small errors and MAE for large errors, making it
    robust to outliers while maintaining smoothness near zero.

    Args:
        y_pred: Predicted value
        y_true: True value
        delta: Threshold where loss transitions from quadratic to linear. Default: 1.0

    Example:
        >>> model = xgb.train(params, dtrain, obj=huber.xgb_objective)
        >>> # With custom delta:
        >>> model = xgb.train(params, dtrain, obj=huber.get_xgb_objective(delta=0.5))
    """
    error = y_pred - y_true
    abs_error = jnp.abs(error)
    return jnp.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta),
    )


@AutoObjective
def quantile(
    y_pred: jax.Array,
    y_true: jax.Array,
    q: float = 0.5,
    alpha: float = 0.01,
) -> jax.Array:
    """
    Smooth Quantile Loss (Pinball Loss) for quantile regression.

    Asymmetric loss that penalizes under-prediction and over-prediction
    differently, allowing prediction of specific quantiles.

    Uses a smooth approximation to ensure non-zero Hessians.

    Args:
        y_pred: Predicted value
        y_true: True value
        q: Target quantile in (0, 1). Default: 0.5 (median)
           - q=0.1: 10th percentile (conservative)
           - q=0.5: median
           - q=0.9: 90th percentile (aggressive)
        alpha: Smoothing parameter for regularization. Default: 0.01

    Example:
        >>> # Predict the 90th percentile
        >>> q90 = quantile.with_params(q=0.9)
        >>> model = xgb.train(params, dtrain, obj=q90.xgb_objective)
    """
    error = y_true - y_pred
    abs_error = jnp.abs(error)

    # Standard quantile loss + small quadratic regularization
    loss = jnp.where(
        error >= 0,
        q * abs_error,
        (1 - q) * abs_error,
    )

    # Add small quadratic term for Hessian
    loss = loss + alpha * error**2

    return loss


@AutoObjective
def tweedie(
    y_pred: jax.Array,
    y_true: jax.Array,
    p: float = 1.5,
) -> jax.Array:
    """
    Tweedie Loss for zero-inflated and positive continuous data.

    Common in insurance claims, rainfall prediction, and other scenarios
    with many zeros and positive continuous values.

    Args:
        y_pred: Raw prediction (will be exponentiated to ensure positivity)
        y_true: True value (must be non-negative)
        p: Tweedie power parameter. Default: 1.5
           - p=1: Poisson-like
           - p=2: Gamma-like
           - 1<p<2: Compound Poisson-Gamma (most common for insurance)

    Example:
        >>> model = xgb.train(params, dtrain, obj=tweedie.xgb_objective)
    """
    # Ensure positive predictions via exp
    mu = jnp.exp(y_pred)
    mu = jnp.clip(mu, 1e-10, 1e10)

    # Tweedie deviance
    return -y_true * jnp.power(mu, 1 - p) / (1 - p) + jnp.power(mu, 2 - p) / (2 - p)


@AutoObjective
def asymmetric(
    y_pred: jax.Array,
    y_true: jax.Array,
    alpha: float = 0.7,
) -> jax.Array:
    """
    Asymmetric Loss for different penalties on under/over-prediction.

    Useful when the cost of under-prediction differs from over-prediction,
    e.g., inventory management, demand forecasting.

    Args:
        y_pred: Predicted value
        y_true: True value
        alpha: Asymmetry parameter in (0, 1). Default: 0.7
               - alpha > 0.5: Penalize under-prediction more
               - alpha < 0.5: Penalize over-prediction more

    Example:
        >>> # Penalize under-prediction heavily (for safety stock)
        >>> obj = asymmetric.with_params(alpha=0.9)
        >>> model = xgb.train(params, dtrain, obj=obj.xgb_objective)
    """
    error = y_true - y_pred
    return jnp.where(
        error >= 0,
        alpha * error**2,
        (1 - alpha) * error**2,
    )


@AutoObjective
def log_cosh(
    y_pred: jax.Array,
    y_true: jax.Array,
) -> jax.Array:
    """
    Log-Cosh Loss for smooth robust regression.

    Similar to Huber loss but smoother everywhere. Twice differentiable,
    which can lead to better optimization behavior.

    Args:
        y_pred: Predicted value
        y_true: True value

    Example:
        >>> model = xgb.train(params, dtrain, obj=log_cosh.xgb_objective)
    """
    error = y_pred - y_true
    return jnp.log(jnp.cosh(error))


@AutoObjective
def pseudo_huber(
    y_pred: jax.Array,
    y_true: jax.Array,
    delta: float = 1.0,
) -> jax.Array:
    """
    Pseudo-Huber Loss.

    A smooth approximation to the Huber loss that is differentiable everywhere.

    Args:
        y_pred: Predicted value
        y_true: True value
        delta: Scale parameter controlling the transition. Default: 1.0

    Example:
        >>> model = xgb.train(params, dtrain, obj=pseudo_huber.xgb_objective)
    """
    error = y_pred - y_true
    return delta**2 * (jnp.sqrt(1 + (error / delta) ** 2) - 1)


@AutoObjective
def mae_smooth(
    y_pred: jax.Array,
    y_true: jax.Array,
    beta: float = 0.1,
) -> jax.Array:
    """
    Smooth Mean Absolute Error Loss.

    A smooth approximation to MAE that has non-zero Hessian.
    Uses sqrt(error^2 + beta^2) - beta as approximation.

    Args:
        y_pred: Predicted value
        y_true: True value
        beta: Smoothing parameter. Smaller = closer to true MAE. Default: 0.1

    Example:
        >>> model = xgb.train(params, dtrain, obj=mae_smooth.xgb_objective)
    """
    error = y_pred - y_true
    return jnp.sqrt(error**2 + beta**2) - beta

