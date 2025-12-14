"""
Survival analysis objective functions.

These objectives handle censored data for survival analysis and
interval regression tasks.
"""

import jax
import jax.numpy as jnp

from jaxboost.objective.auto import AutoObjective


@AutoObjective
def aft(
    y_pred: jax.Array,
    y_true: jax.Array,
    label_lower_bound: jax.Array | None = None,
    label_upper_bound: jax.Array | None = None,
    sigma: float = 1.0,
) -> jax.Array:
    """
    Accelerated Failure Time (AFT) Loss for survival analysis.

    Models log(T) = y_pred + sigma * epsilon, where epsilon follows
    a normal distribution.

    Handles censored data:
    - Uncensored: lower == upper (exact event time)
    - Right-censored: upper == inf (event hasn't occurred yet)
    - Interval-censored: lower < upper (event in time range)

    Args:
        y_pred: Predicted log survival time
        y_true: Label (used as lower bound if bounds not provided)
        label_lower_bound: Lower bound of survival time
        label_upper_bound: Upper bound of survival time
        sigma: Scale parameter (default 1.0)

    Example:
        >>> # Survival data: some patients censored (still alive)
        >>> lower_bounds = event_times
        >>> upper_bounds = np.where(is_censored, np.inf, event_times)
        >>> dtrain.set_float_info('label_lower_bound', lower_bounds)
        >>> dtrain.set_float_info('label_upper_bound', upper_bounds)
        >>> model = xgb.train(params, dtrain, obj=aft.xgb_objective)
    """
    # Use y_true as default bounds if not provided
    lower = y_true if label_lower_bound is None else label_lower_bound
    upper = y_true if label_upper_bound is None else label_upper_bound

    # Log transform (AFT works in log-time space)
    log_lower = jnp.log(jnp.maximum(lower, 1e-10))

    # Standardized residual for lower bound
    z_lower = (log_lower - y_pred) / sigma

    # Check censoring type
    is_uncensored = jnp.abs(upper - lower) < 1e-7
    is_right_censored = upper > 1e10  # inf check

    # Uncensored: -log(pdf) = 0.5*z^2 + log(sigma) + const
    uncensored_loss = 0.5 * z_lower**2 + jnp.log(sigma)

    # Right-censored: -log(survival) = -log(1 - CDF(z))
    cdf_lower = jax.scipy.stats.norm.cdf(z_lower)
    cdf_lower_clipped = jnp.clip(cdf_lower, 0.0, 1.0 - 1e-7)
    right_censored_loss = -jnp.log1p(-cdf_lower_clipped)

    # Interval-censored: -log(CDF(upper) - CDF(lower))
    log_upper = jnp.log(jnp.maximum(upper, 1e-10))
    log_upper_clipped = jnp.clip(log_upper, -100.0, 100.0)
    z_upper = (log_upper_clipped - y_pred) / sigma
    cdf_upper = jax.scipy.stats.norm.cdf(z_upper)
    interval_prob = jnp.maximum(cdf_upper - cdf_lower, 1e-10)
    interval_loss = -jnp.log(interval_prob)

    # Select appropriate loss
    loss = jnp.where(
        is_uncensored,
        uncensored_loss,
        jnp.where(is_right_censored, right_censored_loss, interval_loss),
    )

    return loss


@AutoObjective
def interval_regression(
    y_pred: jax.Array,
    y_true: jax.Array,
    label_lower_bound: jax.Array | None = None,
    label_upper_bound: jax.Array | None = None,
    sigma: float = 1.0,
) -> jax.Array:
    """
    Interval Regression Loss for censored/interval data.

    Handles three cases:
    - Uncensored: lower == upper (exact observation)
    - Right-censored: upper == inf (only lower bound known)
    - Left-censored: lower == -inf (only upper bound known)
    - Interval-censored: lower < upper (observation in range)

    Uses a log-normal likelihood approach.

    Args:
        y_pred: Log of predicted value
        y_true: Label (used as lower bound if bounds not provided)
        label_lower_bound: Lower bound of interval (optional)
        label_upper_bound: Upper bound of interval (optional)
        sigma: Scale parameter for the distribution

    Example:
        >>> dtrain = xgb.DMatrix(X, label=y)
        >>> dtrain.set_float_info('label_lower_bound', lower_bounds)
        >>> dtrain.set_float_info('label_upper_bound', upper_bounds)
        >>> model = xgb.train(params, dtrain, obj=interval_regression.xgb_objective)
    """
    # Use y_true as default bounds if not provided
    lower = y_true if label_lower_bound is None else label_lower_bound
    upper = y_true if label_upper_bound is None else label_upper_bound

    # Standardized residuals
    z_lower = (jnp.log(jnp.maximum(lower, 1e-10)) - y_pred) / sigma
    z_upper = (jnp.log(jnp.maximum(upper, 1e-10)) - y_pred) / sigma

    # Check if uncensored (lower == upper)
    is_uncensored = jnp.abs(upper - lower) < 1e-10

    # Uncensored: negative log of normal PDF
    uncensored_loss = 0.5 * z_lower**2 + jnp.log(sigma)

    # Censored: negative log of CDF difference
    cdf_upper = jax.scipy.stats.norm.cdf(z_upper)
    cdf_lower = jax.scipy.stats.norm.cdf(z_lower)
    interval_prob = jnp.maximum(cdf_upper - cdf_lower, 1e-10)
    censored_loss = -jnp.log(interval_prob)

    return jnp.where(is_uncensored, uncensored_loss, censored_loss)


@AutoObjective
def cox_partial_likelihood(
    y_pred: jax.Array,
    y_true: jax.Array,
    is_event: jax.Array | None = None,
) -> jax.Array:
    """
    Cox Proportional Hazards partial likelihood loss (per-sample approximation).

    Note: This is a simplified per-sample approximation. For true Cox PH,
    the loss involves the risk set at each event time, which requires
    special handling not fully captured in this per-sample formulation.

    Args:
        y_pred: Log hazard ratio prediction
        y_true: Observed time
        is_event: Whether the observation is an event (1) or censored (0)

    Example:
        >>> model = xgb.train(params, dtrain, obj=cox_partial_likelihood.xgb_objective)
    """
    # Default: all observations are events
    event = jnp.ones_like(y_true) if is_event is None else is_event

    # Simplified per-sample loss: event * (log_sum_exp - y_pred)
    # This is an approximation; true Cox requires ranking
    loss = event * (jnp.exp(y_pred) - y_pred)

    return loss


@AutoObjective
def weibull_aft(
    y_pred: jax.Array,
    y_true: jax.Array,
    label_lower_bound: jax.Array | None = None,
    label_upper_bound: jax.Array | None = None,
    k: float = 1.0,
) -> jax.Array:
    """
    Weibull AFT (Accelerated Failure Time) Loss.

    Uses Weibull distribution for survival times instead of log-normal.

    Args:
        y_pred: Predicted log scale parameter (lambda)
        y_true: Label (used as lower bound if bounds not provided)
        label_lower_bound: Lower bound of survival time
        label_upper_bound: Upper bound of survival time
        k: Shape parameter of Weibull distribution (default 1.0 = exponential)

    Example:
        >>> model = xgb.train(params, dtrain, obj=weibull_aft.xgb_objective)
    """
    # Use y_true as default bounds if not provided
    lower = y_true if label_lower_bound is None else label_lower_bound
    upper = y_true if label_upper_bound is None else label_upper_bound

    # Scale parameter from prediction
    lambda_ = jnp.exp(y_pred)
    lambda_ = jnp.clip(lambda_, 1e-10, 1e10)

    # Check censoring type
    is_uncensored = jnp.abs(upper - lower) < 1e-7
    is_right_censored = upper > 1e10

    # Weibull survival function: S(t) = exp(-(t/lambda)^k)
    # Weibull PDF: f(t) = (k/lambda) * (t/lambda)^(k-1) * exp(-(t/lambda)^k)

    t_lower = jnp.maximum(lower, 1e-10)

    # Uncensored: -log(pdf)
    z_lower = (t_lower / lambda_) ** k
    log_pdf = jnp.log(k) - jnp.log(lambda_) + (k - 1) * (
        jnp.log(t_lower) - jnp.log(lambda_)
    ) - z_lower
    uncensored_loss = -log_pdf

    # Right-censored: -log(survival) = (t/lambda)^k
    right_censored_loss = z_lower

    # Interval-censored: -log(S(lower) - S(upper))
    t_upper = jnp.maximum(upper, 1e-10)
    t_upper = jnp.clip(t_upper, 0, 1e10)
    z_upper = (t_upper / lambda_) ** k
    survival_lower = jnp.exp(-z_lower)
    survival_upper = jnp.exp(-z_upper)
    interval_prob = jnp.maximum(survival_lower - survival_upper, 1e-10)
    interval_loss = -jnp.log(interval_prob)

    loss = jnp.where(
        is_uncensored,
        uncensored_loss,
        jnp.where(is_right_censored, right_censored_loss, interval_loss),
    )

    return loss

