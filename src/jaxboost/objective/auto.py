"""
Core AutoObjective class for automatic gradient/Hessian computation.

This module provides the foundation for generating XGBoost and LightGBM
compatible objective functions using JAX automatic differentiation.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class LossFunction(Protocol):
    """Protocol for loss functions."""

    def __call__(
        self, y_pred: jax.Array, y_true: jax.Array, **kwargs: Any
    ) -> jax.Array:
        """Compute loss for a single sample."""
        ...


class AutoObjective:
    """
    Automatically generate XGBoost/LightGBM objective functions.

    Uses JAX automatic differentiation to compute gradients and Hessians,
    eliminating the need for manual derivation.

    Args:
        loss_fn: A loss function that takes (y_pred, y_true, **kwargs) and
                 returns a scalar loss. Should operate on single samples.

    Example:
        >>> @AutoObjective
        ... def my_loss(y_pred, y_true, alpha=0.5):
        ...     return alpha * (y_pred - y_true) ** 2
        >>>
        >>> # Use with XGBoost
        >>> model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
        >>>
        >>> # Use with custom parameters
        >>> model = xgb.train(params, dtrain, obj=my_loss.get_xgb_objective(alpha=0.7))
    """

    def __init__(self, loss_fn: LossFunction) -> None:
        self.loss_fn = loss_fn
        self._name = getattr(loss_fn, "__name__", "custom_objective")
        self._default_kwargs: dict[str, Any] = {}

        # Pre-compile gradient and Hessian functions
        self._grad_fn = jax.grad(self._loss_wrapper, argnums=0)
        self._hess_fn = jax.grad(
            lambda *args, **kw: self._grad_fn(*args, **kw), argnums=0
        )

        # Cache for JIT-compiled vmap functions (keyed by array kwargs pattern)
        self._vmap_cache: dict[frozenset[str], tuple[Any, Any]] = {}

    def _loss_wrapper(
        self,
        y_pred: jax.Array,
        y_true: jax.Array,
        scalar_kwargs: dict[str, Any],
        array_kwargs: dict[str, Any],
    ) -> jax.Array:
        """Wrapper to handle kwargs as dict arguments for JAX transformations."""
        return self.loss_fn(y_pred, y_true, **scalar_kwargs, **array_kwargs)

    def _get_vmap_fns(self, array_kwarg_keys: frozenset[str]) -> tuple[Any, Any]:
        """Get or create JIT-compiled vmap functions for given array kwargs pattern."""
        if array_kwarg_keys not in self._vmap_cache:
            # Create in_axes: (0, 0, None for scalars, 0 for each array kwarg)
            array_in_axes = {k: 0 for k in array_kwarg_keys}
            in_axes = (0, 0, None, array_in_axes if array_in_axes else None)

            vmap_grad = jax.jit(jax.vmap(self._grad_fn, in_axes=in_axes))
            vmap_hess = jax.jit(jax.vmap(self._hess_fn, in_axes=in_axes))
            self._vmap_cache[array_kwarg_keys] = (vmap_grad, vmap_hess)

        return self._vmap_cache[array_kwarg_keys]

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

    def __call__(
        self, y_pred: jax.Array, y_true: jax.Array, **kwargs: Any
    ) -> jax.Array:
        """Compute the loss value for a batch."""
        n_samples = len(y_pred)
        scalar_kwargs, array_kwargs, array_keys = self._split_kwargs(kwargs, n_samples)
        in_axes = (0, 0, None, {k: 0 for k in array_keys} if array_keys else None)
        return jax.vmap(self._loss_wrapper, in_axes=in_axes)(
            y_pred, y_true, scalar_kwargs, array_kwargs
        )

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradient of loss w.r.t. y_pred for each sample.

        Args:
            y_pred: Predictions, shape (n_samples,)
            y_true: True labels, shape (n_samples,)
            **kwargs: Additional arguments passed to the loss function

        Returns:
            Gradients, shape (n_samples,)
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}
        y_pred_jax = jnp.asarray(y_pred, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.float32)

        n_samples = len(y_pred)
        scalar_kwargs, array_kwargs, array_keys = self._split_kwargs(
            merged_kwargs, n_samples
        )
        vmap_grad, _ = self._get_vmap_fns(array_keys)

        grads = vmap_grad(y_pred_jax, y_true_jax, scalar_kwargs, array_kwargs)
        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute Hessian (second derivative) of loss w.r.t. y_pred for each sample.

        Args:
            y_pred: Predictions, shape (n_samples,)
            y_true: True labels, shape (n_samples,)
            **kwargs: Additional arguments passed to the loss function

        Returns:
            Hessians (diagonal), shape (n_samples,)
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}
        y_pred_jax = jnp.asarray(y_pred, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.float32)

        n_samples = len(y_pred)
        scalar_kwargs, array_kwargs, array_keys = self._split_kwargs(
            merged_kwargs, n_samples
        )
        _, vmap_hess = self._get_vmap_fns(array_keys)

        hess = vmap_hess(y_pred_jax, y_true_jax, scalar_kwargs, array_kwargs)
        return np.asarray(hess, dtype=np.float64)

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Compute both gradient and Hessian efficiently.

        Args:
            y_pred: Predictions, shape (n_samples,)
            y_true: True labels, shape (n_samples,)
            sample_weight: Optional sample weights, shape (n_samples,)
            **kwargs: Additional arguments passed to the loss function

        Returns:
            Tuple of (gradients, hessians), each shape (n_samples,)
        """
        grad = self.gradient(y_pred, y_true, **kwargs)
        hess = self.hessian(y_pred, y_true, **kwargs)

        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.asarray(sample_weight, dtype=np.float64)
            grad = grad * weight
            hess = hess * weight

        return grad, hess

    def get_xgb_objective(
        self, **kwargs: Any
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get an XGBoost-compatible objective function with custom parameters.

        Automatically handles:
        - sample_weight: Sample weights from DMatrix
        - label_lower_bound: Lower bound labels for interval/survival regression
        - label_upper_bound: Upper bound labels for interval/survival regression

        Args:
            **kwargs: Parameters to pass to the loss function

        Returns:
            XGBoost objective function: (y_pred, dtrain) -> (grad, hess)
        """

        def objective(
            y_pred: NDArray[np.floating[Any]], dtrain: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()

            # Get label bounds for interval/survival regression
            extra_kwargs = dict(kwargs)

            if hasattr(dtrain, "get_float_info"):
                try:
                    lower_bound = dtrain.get_float_info("label_lower_bound")
                    if len(lower_bound) > 0:
                        extra_kwargs["label_lower_bound"] = lower_bound
                except Exception:
                    pass
                try:
                    upper_bound = dtrain.get_float_info("label_upper_bound")
                    if len(upper_bound) > 0:
                        extra_kwargs["label_upper_bound"] = upper_bound
                except Exception:
                    pass

            return self.grad_hess(
                y_pred, y_true, sample_weight=sample_weight, **extra_kwargs
            )

        objective.__name__ = f"{self._name}_xgb_objective"
        return objective

    @property
    def xgb_objective(
        self,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        XGBoost-compatible objective function using default parameters.

        Returns:
            XGBoost objective function: (y_pred, dtrain) -> (grad, hess)

        Example:
            >>> model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
        """
        return self.get_xgb_objective(**self._default_kwargs)

    def get_lgb_objective(
        self, **kwargs: Any
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get a LightGBM-compatible objective function with custom parameters.

        Automatically handles sample weights if set in the Dataset.

        Args:
            **kwargs: Parameters to pass to the loss function

        Returns:
            LightGBM objective function: (y_pred, dataset) -> (grad, hess)
        """

        def objective(
            y_pred: NDArray[np.floating[Any]], dataset: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dataset.get_label()
            sample_weight = dataset.get_weight()
            return self.grad_hess(
                y_pred, y_true, sample_weight=sample_weight, **kwargs
            )

        objective.__name__ = f"{self._name}_lgb_objective"
        return objective

    @property
    def lgb_objective(
        self,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        LightGBM-compatible objective function using default parameters.

        Returns:
            LightGBM objective function: (y_pred, dataset) -> (grad, hess)

        Example:
            >>> model = lgb.train(params, dtrain, fobj=my_loss.lgb_objective)
        """
        return self.get_lgb_objective(**self._default_kwargs)

    def with_params(self, **kwargs: Any) -> AutoObjective:
        """
        Create a new AutoObjective with default parameters set.

        Args:
            **kwargs: Default parameters for the loss function

        Returns:
            New AutoObjective instance with default parameters

        Example:
            >>> focal = focal_loss.with_params(gamma=3.0, alpha=0.75)
            >>> model = xgb.train(params, dtrain, obj=focal.xgb_objective)
        """
        new_instance = AutoObjective(self.loss_fn)
        new_instance._default_kwargs = {**self._default_kwargs, **kwargs}
        return new_instance

    def __repr__(self) -> str:
        params = f", params={self._default_kwargs}" if self._default_kwargs else ""
        return f"AutoObjective({self._name}{params})"


def auto_objective(func: LossFunction) -> AutoObjective:
    """
    Decorator to create an AutoObjective from a loss function.

    This is an alias for AutoObjective() that can be used as a decorator.

    Example:
        >>> @auto_objective
        ... def my_custom_loss(y_pred, y_true):
        ...     return (y_pred - y_true) ** 2
        >>>
        >>> model = xgb.train(params, dtrain, obj=my_custom_loss.xgb_objective)
    """
    return AutoObjective(func)

