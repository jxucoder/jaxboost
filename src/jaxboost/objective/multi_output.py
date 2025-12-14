"""
Multi-output objective functions.

These objectives handle cases where each sample has multiple predictions,
such as multi-target regression or parametric models.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class MultiOutputObjective:
    """
    Objective for multi-output/multi-task learning.

    Handles cases where each sample has multiple predictions, such as:
    - Multi-target regression
    - Parametric models learning multiple parameters
    - Uncertainty estimation (mean + variance)

    Args:
        loss_fn: A loss function that takes (y_pred, y_true, **kwargs) where:
                 - y_pred: shape (n_outputs,) for a single sample
                 - y_true: shape (n_outputs,) or scalar for a single sample
                 Returns a scalar loss.
        n_outputs: Number of outputs per sample

    Example:
        >>> @MultiOutputObjective(n_outputs=2)
        ... def parametric_loss(params, y_true, t=None):
        ...     # params = [A, B] for model y = A + B*t
        ...     A, B = params[0], params[1]
        ...     y_pred = A + B * t
        ...     return (y_pred - y_true) ** 2
        >>>
        >>> model = xgb.train(params, dtrain, obj=parametric_loss.xgb_objective)
    """

    def __init__(
        self,
        loss_fn: Callable[..., jax.Array] | None = None,
        n_outputs: int = 2,
    ) -> None:
        self.n_outputs = n_outputs
        self._default_kwargs: dict[str, Any] = {}

        if loss_fn is not None:
            self._init_with_fn(loss_fn)
        else:
            self.loss_fn = None
            self._name = "multi_output_objective"

    def _init_with_fn(self, loss_fn: Callable[..., jax.Array]) -> None:
        """Initialize with a loss function."""
        self.loss_fn = loss_fn
        self._name = getattr(loss_fn, "__name__", "multi_output_objective")

        # Gradient function: returns gradient w.r.t. each output
        self._grad_fn = jax.grad(self._loss_wrapper, argnums=0)

        # Hessian: diagonal of Hessian matrix
        def diag_hess(
            y_pred: jax.Array,
            y_true: jax.Array,
            scalar_kwargs: dict[str, Any],
            array_kwargs: dict[str, Any],
        ) -> jax.Array:
            """Compute diagonal of Hessian."""

            def grad_i(i: int) -> jax.Array:
                def loss_i(yi: jax.Array) -> jax.Array:
                    y_pred_new = y_pred.at[i].set(yi)
                    return self._loss_wrapper(
                        y_pred_new, y_true, scalar_kwargs, array_kwargs
                    )

                return jax.grad(jax.grad(loss_i))(y_pred[i])

            return jax.vmap(grad_i)(jnp.arange(self.n_outputs))

        self._hess_fn = diag_hess

    def __call__(self, loss_fn: Callable[..., jax.Array]) -> MultiOutputObjective:
        """Allow use as a decorator with arguments."""
        new_instance = MultiOutputObjective(loss_fn=loss_fn, n_outputs=self.n_outputs)
        new_instance._default_kwargs = self._default_kwargs.copy()
        return new_instance

    def _loss_wrapper(
        self,
        y_pred: jax.Array,
        y_true: jax.Array,
        scalar_kwargs: dict[str, Any],
        array_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Wrapper for loss function."""
        return self.loss_fn(y_pred, y_true, **scalar_kwargs, **array_kwargs)

    def _split_kwargs(
        self, kwargs: dict[str, Any], n_samples: int
    ) -> tuple[dict[str, Any], dict[str, jax.Array], frozenset[str]]:
        """Split kwargs into scalar and array kwargs."""
        scalar_kwargs: dict[str, Any] = {}
        array_kwargs: dict[str, jax.Array] = {}

        for k, v in kwargs.items():
            if (
                isinstance(v, (np.ndarray, jax.Array))
                and hasattr(v, "__len__")
                and len(v) == n_samples
            ):
                array_kwargs[k] = jnp.asarray(v, dtype=jnp.float32)
            else:
                scalar_kwargs[k] = v

        return scalar_kwargs, array_kwargs, frozenset(array_kwargs.keys())

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradient for multi-output predictions.

        Args:
            y_pred: Predictions, shape (n_samples, n_outputs) or flattened
            y_true: True labels
            **kwargs: Additional arguments

        Returns:
            Gradients, shape (n_samples * n_outputs,) for XGBoost compatibility
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}

        # Reshape if flattened
        y_pred_2d = np.asarray(y_pred).reshape(-1, self.n_outputs)
        n_samples = y_pred_2d.shape[0]

        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.float32)

        scalar_kwargs, array_kwargs, _ = self._split_kwargs(merged_kwargs, n_samples)

        # Compute gradients for each sample
        def grad_single(
            y_pred_i: jax.Array, y_true_i: jax.Array, array_kwargs_i: dict[str, Any]
        ) -> jax.Array:
            return self._grad_fn(y_pred_i, y_true_i, scalar_kwargs, array_kwargs_i)

        if array_kwargs:
            grads = jax.vmap(
                lambda yp, yt, *arr_vals: grad_single(
                    yp, yt, dict(zip(array_kwargs.keys(), arr_vals))
                )
            )(y_pred_jax, y_true_jax, *array_kwargs.values())
        else:
            grads = jax.vmap(lambda yp, yt: grad_single(yp, yt, {}))(
                y_pred_jax, y_true_jax
            )

        # Flatten for XGBoost
        return np.asarray(grads, dtype=np.float64).flatten()

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute diagonal Hessian for multi-output predictions.

        Args:
            y_pred: Predictions, shape (n_samples, n_outputs) or flattened
            y_true: True labels
            **kwargs: Additional arguments

        Returns:
            Diagonal Hessians, shape (n_samples * n_outputs,) for XGBoost
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}

        y_pred_2d = np.asarray(y_pred).reshape(-1, self.n_outputs)
        n_samples = y_pred_2d.shape[0]

        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.float32)

        scalar_kwargs, array_kwargs, _ = self._split_kwargs(merged_kwargs, n_samples)

        def hess_single(
            y_pred_i: jax.Array, y_true_i: jax.Array, array_kwargs_i: dict[str, Any]
        ) -> jax.Array:
            return self._hess_fn(y_pred_i, y_true_i, scalar_kwargs, array_kwargs_i)

        if array_kwargs:
            hess = jax.vmap(
                lambda yp, yt, *arr_vals: hess_single(
                    yp, yt, dict(zip(array_kwargs.keys(), arr_vals))
                )
            )(y_pred_jax, y_true_jax, *array_kwargs.values())
        else:
            hess = jax.vmap(lambda yp, yt: hess_single(yp, yt, {}))(
                y_pred_jax, y_true_jax
            )

        return np.asarray(hess, dtype=np.float64).flatten()

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Compute both gradient and Hessian.

        Args:
            y_pred: Predictions, shape (n_samples * n_outputs,)
            y_true: True labels
            sample_weight: Optional sample weights, shape (n_samples,)
            **kwargs: Additional arguments

        Returns:
            Tuple of (gradients, hessians), each shape (n_samples * n_outputs,)
        """
        grad = self.gradient(y_pred, y_true, **kwargs)
        hess = self.hessian(y_pred, y_true, **kwargs)

        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.asarray(sample_weight, dtype=np.float64)
            weight_expanded = np.repeat(weight, self.n_outputs)
            grad = grad * weight_expanded
            hess = hess * weight_expanded

        return grad, hess

    def get_xgb_objective(
        self, **kwargs: Any
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get an XGBoost-compatible objective function for multi-output.

        Note: Requires XGBoost with multi-output support.
        Set `multi_strategy='multi_output_tree'` and `num_target=n_outputs` in params.

        Args:
            **kwargs: Parameters to pass to the loss function

        Returns:
            XGBoost objective function
        """

        def objective(
            y_pred: NDArray[np.floating[Any]], dtrain: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()
            return self.grad_hess(
                y_pred, y_true, sample_weight=sample_weight, **kwargs
            )

        objective.__name__ = f"{self._name}_xgb_objective"
        return objective

    @property
    def xgb_objective(self) -> Callable[..., Any]:
        """XGBoost-compatible objective function."""
        return self.get_xgb_objective(**self._default_kwargs)

    def with_params(self, **kwargs: Any) -> MultiOutputObjective:
        """Create a new instance with default parameters set."""
        new_instance = MultiOutputObjective(
            loss_fn=self.loss_fn, n_outputs=self.n_outputs
        )
        new_instance._default_kwargs = {**self._default_kwargs, **kwargs}
        return new_instance

    def __repr__(self) -> str:
        params = f", params={self._default_kwargs}" if self._default_kwargs else ""
        return (
            f"MultiOutputObjective({self._name}, n_outputs={self.n_outputs}{params})"
        )


def multi_output_objective(
    n_outputs: int = 2,
) -> Callable[[Callable[..., jax.Array]], MultiOutputObjective]:
    """
    Decorator factory for multi-output objective functions.

    Args:
        n_outputs: Number of outputs per sample

    Example:
        >>> @multi_output_objective(n_outputs=2)
        ... def parametric_loss(params, y_true, t=None):
        ...     A, B = params[0], params[1]
        ...     y_pred = A + B * t
        ...     return (y_pred - y_true) ** 2
    """

    def decorator(func: Callable[..., jax.Array]) -> MultiOutputObjective:
        return MultiOutputObjective(loss_fn=func, n_outputs=n_outputs)

    return decorator


# =============================================================================
# Built-in Multi-output Objectives
# =============================================================================


def gaussian_nll(n_outputs: int = 2) -> MultiOutputObjective:
    """
    Gaussian Negative Log-Likelihood for uncertainty estimation.

    Predicts both mean and log-variance, enabling uncertainty quantification.

    Args:
        n_outputs: Should be 2 (mean, log_variance)

    Returns:
        MultiOutputObjective instance

    Example:
        >>> nll = gaussian_nll()
        >>> params = {'multi_strategy': 'multi_output_tree', 'num_target': 2}
        >>> model = xgb.train(params, dtrain, obj=nll.xgb_objective)
    """
    if n_outputs != 2:
        raise ValueError("gaussian_nll requires n_outputs=2 (mean, log_variance)")

    @multi_output_objective(n_outputs=2)
    def gnll(params: jax.Array, y_true: jax.Array) -> jax.Array:
        """Gaussian NLL for a single sample."""
        mean = params[0]
        log_var = params[1]

        # Clip log_var for numerical stability
        log_var = jnp.clip(log_var, -10.0, 10.0)
        var = jnp.exp(log_var)

        # NLL = 0.5 * (log(var) + (y - mean)^2 / var)
        nll = 0.5 * (log_var + (y_true - mean) ** 2 / var)
        return nll

    return gnll


def laplace_nll(n_outputs: int = 2) -> MultiOutputObjective:
    """
    Laplace Negative Log-Likelihood for robust uncertainty estimation.

    Similar to Gaussian NLL but uses Laplace distribution, which is more
    robust to outliers.

    Args:
        n_outputs: Should be 2 (location, log_scale)

    Returns:
        MultiOutputObjective instance

    Example:
        >>> nll = laplace_nll()
        >>> model = xgb.train(params, dtrain, obj=nll.xgb_objective)
    """
    if n_outputs != 2:
        raise ValueError("laplace_nll requires n_outputs=2 (location, log_scale)")

    @multi_output_objective(n_outputs=2)
    def lnll(params: jax.Array, y_true: jax.Array) -> jax.Array:
        """Laplace NLL for a single sample."""
        loc = params[0]
        log_scale = params[1]

        log_scale = jnp.clip(log_scale, -10.0, 10.0)
        scale = jnp.exp(log_scale)

        # NLL = log(scale) + |y - loc| / scale
        nll = log_scale + jnp.abs(y_true - loc) / scale
        return nll

    return lnll

