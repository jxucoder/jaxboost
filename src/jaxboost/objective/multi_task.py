"""
Multi-task learning objective with missing label support.

In multi-task learning, it's common for samples to have labels for only 
some tasks. XGBoost's native multi-output doesn't support this well.

This module provides objectives that:
1. Accept a mask indicating which labels are valid
2. Compute gradients only for valid labels
3. Set gradients to 0 for missing labels (no parameter update)

Example (single-node):
    >>> import numpy as np
    >>> import xgboost as xgb
    >>> from jaxboost.objective import multi_task_regression
    >>> 
    >>> # Data with missing labels (NaN)
    >>> y_true = np.array([[1.0, np.nan, 0.5],
    ...                    [np.nan, 2.0, 1.0]])
    >>> mask = ~np.isnan(y_true)  # True where valid
    >>> y_filled = np.nan_to_num(y_true, nan=0.0)  # Fill NaN for XGBoost
    >>> 
    >>> # Create DMatrix
    >>> dtrain = xgb.DMatrix(X, label=y_filled.flatten())
    >>> 
    >>> # Create objective with mask
    >>> obj = multi_task_regression(n_tasks=3)
    >>> 
    >>> # Pass mask via get_xgb_objective
    >>> params = {'multi_strategy': 'multi_output_tree', 'num_target': 3}
    >>> model = xgb.train(params, dtrain, obj=obj.get_xgb_objective(mask=mask))

Example (distributed / Ray XGBoost):
    >>> # Store mask in DMatrix so it gets partitioned with the data
    >>> dtrain.set_float_info("label_mask", mask.astype(np.float32).flatten())
    >>> 
    >>> # Use mask_key to read mask from each worker's local DMatrix
    >>> model = xgb.train(params, dtrain, obj=obj.get_xgb_objective(mask_key="label_mask"))
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class MaskedMultiTaskObjective:
    """
    Multi-task objective with support for missing labels.

    Key features:
    - Handles arbitrary label missingness patterns
    - Gradients are 0 for missing labels (no update)
    - Supports per-task loss functions
    - Automatic gradient/Hessian computation via JAX

    Args:
        n_tasks: Number of tasks (outputs per sample)
        task_loss_fn: Loss function for each task. Signature:
                      (y_pred, y_true) -> scalar loss
                      Default is squared error.
        task_weights: Optional weights for each task, shape (n_tasks,)

    Example:
        >>> # Basic usage
        >>> obj = MaskedMultiTaskObjective(n_tasks=3)
        >>> 
        >>> # Custom per-task loss
        >>> @MaskedMultiTaskObjective(n_tasks=3)
        ... def my_mtl_loss(y_pred, y_true):
        ...     return (y_pred - y_true) ** 2
        >>> 
        >>> # Different weights per task
        >>> obj = MaskedMultiTaskObjective(n_tasks=3, task_weights=[1.0, 2.0, 0.5])
    """

    def __init__(
        self,
        task_loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        n_tasks: int = 2,
        task_weights: list[float] | NDArray[np.floating[Any]] | None = None,
    ) -> None:
        self.n_tasks = n_tasks
        self.task_weights = (
            jnp.asarray(task_weights, dtype=jnp.float32)
            if task_weights is not None
            else jnp.ones(n_tasks, dtype=jnp.float32)
        )
        self._default_kwargs: dict[str, Any] = {}

        if task_loss_fn is not None:
            self._init_with_fn(task_loss_fn)
        else:
            # Default: squared error
            self._init_with_fn(lambda y_pred, y_true: (y_pred - y_true) ** 2)

    def _init_with_fn(
        self, task_loss_fn: Callable[[jax.Array, jax.Array], jax.Array]
    ) -> None:
        """Initialize gradient/Hessian functions."""
        self.task_loss_fn = task_loss_fn
        self._name = getattr(task_loss_fn, "__name__", "masked_multi_task")

        # Gradient w.r.t. single prediction
        self._grad_fn = jax.grad(task_loss_fn, argnums=0)
        # Hessian (second derivative)
        self._hess_fn = jax.grad(lambda yp, yt: self._grad_fn(yp, yt), argnums=0)

    def __call__(
        self, task_loss_fn: Callable[[jax.Array, jax.Array], jax.Array]
    ) -> MaskedMultiTaskObjective:
        """Allow use as decorator."""
        new_instance = MaskedMultiTaskObjective(
            task_loss_fn=task_loss_fn,
            n_tasks=self.n_tasks,
            task_weights=self.task_weights,
        )
        new_instance._default_kwargs = self._default_kwargs.copy()
        return new_instance

    def _compute_grad_hess_single(
        self,
        y_pred: jax.Array,  # (n_tasks,)
        y_true: jax.Array,  # (n_tasks,)
        mask: jax.Array,  # (n_tasks,) - 1 for valid, 0 for missing
    ) -> tuple[jax.Array, jax.Array]:
        """
        Compute gradient and Hessian for a single sample.

        For missing labels (mask=0), gradient and Hessian are set to 0,
        so they don't contribute to parameter updates.
        """

        def task_grad_hess(y_pred_k: jax.Array, y_true_k: jax.Array) -> tuple:
            g = self._grad_fn(y_pred_k, y_true_k)
            h = self._hess_fn(y_pred_k, y_true_k)
            return g, h

        # Compute for all tasks
        grads, hess = jax.vmap(task_grad_hess)(y_pred, y_true)

        # Apply mask: 0 gradient/hessian for missing labels
        grads = grads * mask * self.task_weights
        hess = hess * mask * self.task_weights

        # Ensure Hessian is positive (for XGBoost stability)
        hess = jnp.maximum(hess, 1e-6)

        return grads, hess

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        mask: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradients with missing label support.

        Args:
            y_pred: Predictions, shape (n_samples * n_tasks,) or (n_samples, n_tasks)
            y_true: Labels, shape (n_samples * n_tasks,) or (n_samples, n_tasks)
            mask: Label mask, shape (n_samples * n_tasks,) or (n_samples, n_tasks)
                  1 = valid label, 0 = missing label. Default: all valid.
            **kwargs: Additional arguments (unused)

        Returns:
            Gradients, shape (n_samples * n_tasks,)
        """
        # Reshape to 2D
        y_pred_2d = np.asarray(y_pred, dtype=np.float64).reshape(-1, self.n_tasks)
        y_true_2d = np.asarray(y_true, dtype=np.float64).reshape(-1, self.n_tasks)
        n_samples = y_pred_2d.shape[0]

        if mask is None:
            mask_2d = np.ones((n_samples, self.n_tasks), dtype=np.float32)
        else:
            mask_2d = np.asarray(mask, dtype=np.float32).reshape(-1, self.n_tasks)

        # Convert to JAX arrays
        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true_2d, dtype=jnp.float32)
        mask_jax = jnp.asarray(mask_2d, dtype=jnp.float32)

        # Compute gradients for all samples (vectorized)
        grads, _ = jax.vmap(self._compute_grad_hess_single)(
            y_pred_jax, y_true_jax, mask_jax
        )

        return np.asarray(grads, dtype=np.float64).flatten()

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        mask: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute Hessians with missing label support.

        Args:
            y_pred: Predictions, shape (n_samples * n_tasks,)
            y_true: Labels, shape (n_samples * n_tasks,)
            mask: Label mask, shape (n_samples * n_tasks,)
            **kwargs: Additional arguments

        Returns:
            Diagonal Hessians, shape (n_samples * n_tasks,)
        """
        y_pred_2d = np.asarray(y_pred, dtype=np.float64).reshape(-1, self.n_tasks)
        y_true_2d = np.asarray(y_true, dtype=np.float64).reshape(-1, self.n_tasks)
        n_samples = y_pred_2d.shape[0]

        if mask is None:
            mask_2d = np.ones((n_samples, self.n_tasks), dtype=np.float32)
        else:
            mask_2d = np.asarray(mask, dtype=np.float32).reshape(-1, self.n_tasks)

        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true_2d, dtype=jnp.float32)
        mask_jax = jnp.asarray(mask_2d, dtype=jnp.float32)

        _, hess = jax.vmap(self._compute_grad_hess_single)(
            y_pred_jax, y_true_jax, mask_jax
        )

        return np.asarray(hess, dtype=np.float64).flatten()

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        mask: NDArray[np.floating[Any]] | None = None,
        sample_weight: NDArray[np.floating[Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Compute both gradient and Hessian efficiently.

        Args:
            y_pred: Predictions, shape (n_samples * n_tasks,)
            y_true: Labels, shape (n_samples * n_tasks,)
            mask: Label mask, shape (n_samples * n_tasks,)
            sample_weight: Optional sample weights, shape (n_samples,)
            **kwargs: Additional arguments

        Returns:
            Tuple of (gradients, hessians)
        """
        y_pred_2d = np.asarray(y_pred, dtype=np.float64).reshape(-1, self.n_tasks)
        y_true_2d = np.asarray(y_true, dtype=np.float64).reshape(-1, self.n_tasks)
        n_samples = y_pred_2d.shape[0]

        if mask is None:
            mask_2d = np.ones((n_samples, self.n_tasks), dtype=np.float32)
        else:
            mask_2d = np.asarray(mask, dtype=np.float32).reshape(-1, self.n_tasks)

        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true_2d, dtype=jnp.float32)
        mask_jax = jnp.asarray(mask_2d, dtype=jnp.float32)

        # JIT-compiled batch computation
        @jax.jit
        def batch_grad_hess(y_pred, y_true, mask):
            return jax.vmap(self._compute_grad_hess_single)(y_pred, y_true, mask)

        grads, hess = batch_grad_hess(y_pred_jax, y_true_jax, mask_jax)

        grads = np.asarray(grads, dtype=np.float64).flatten()
        hess = np.asarray(hess, dtype=np.float64).flatten()

        # Apply sample weights if provided
        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.asarray(sample_weight, dtype=np.float64)
            weight_expanded = np.repeat(weight, self.n_tasks)
            grads = grads * weight_expanded
            hess = hess * weight_expanded

        return grads, hess

    def get_xgb_objective(
        self,
        mask: NDArray[np.floating[Any]] | None = None,
        mask_key: str | None = None,
        **kwargs: Any,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get an XGBoost-compatible objective function.

        Args:
            mask: Label mask, shape (n_samples * n_tasks,) or (n_samples, n_tasks).
                  1 = valid label, 0 = missing label. If None, all labels valid.
            mask_key: Key to retrieve mask from DMatrix via get_float_info().
                      Use this for distributed training (e.g., Ray XGBoost) where
                      data is partitioned across workers. The mask will be read
                      per-worker, ensuring correct alignment with local data.
            **kwargs: Additional parameters for the loss function

        Returns:
            XGBoost objective function: (y_pred, dtrain) -> (grad, hess)

        Example:
            >>> # Single-node: pass mask directly
            >>> y_true = np.array([[1.0, np.nan, 0.5],
            ...                    [np.nan, 2.0, 1.0]])
            >>> mask = ~np.isnan(y_true)
            >>> y_true_filled = np.nan_to_num(y_true, nan=0.0)
            >>> dtrain = xgb.DMatrix(X, label=y_true_filled.flatten())
            >>> obj = MaskedMultiTaskObjective(n_tasks=3)
            >>> model = xgb.train(params, dtrain, obj=obj.get_xgb_objective(mask=mask))

            >>> # Distributed (Ray XGBoost): store mask in DMatrix
            >>> dtrain.set_float_info("label_mask", mask.astype(np.float32).flatten())
            >>> model = xgb.train(params, dtrain, obj=obj.get_xgb_objective(mask_key="label_mask"))
        """
        captured_mask = mask
        n_tasks = self.n_tasks

        def objective(
            y_pred: NDArray[np.floating[Any]], dtrain: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()

            if mask_key is not None:
                current_mask = dtrain.get_float_info(mask_key).reshape(-1, n_tasks)
            else:
                current_mask = captured_mask

            return self.grad_hess(
                y_pred, y_true, mask=current_mask, sample_weight=sample_weight, **kwargs
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
        XGBoost-compatible objective function (no mask).
        
        For missing label support, use get_xgb_objective(mask=...) instead.
        """
        return self.get_xgb_objective(mask=None, **self._default_kwargs)

    def get_lgb_objective(
        self,
        mask: NDArray[np.floating[Any]] | None = None,
        mask_key: str | None = None,
        **kwargs: Any,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get a LightGBM-compatible objective function.

        Args:
            mask: Label mask for missing labels.
            mask_key: Key to retrieve mask from Dataset. Use for distributed training.
            **kwargs: Additional parameters.

        Note: LightGBM multi-output support is more limited than XGBoost.
        """
        captured_mask = mask
        n_tasks = self.n_tasks

        def objective(
            y_pred: NDArray[np.floating[Any]], dataset: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dataset.get_label()
            sample_weight = dataset.get_weight()

            if mask_key is not None and hasattr(dataset, "get_data"):
                current_mask = dataset.get_data(mask_key).reshape(-1, n_tasks)
            else:
                current_mask = captured_mask

            return self.grad_hess(
                y_pred, y_true, mask=current_mask, sample_weight=sample_weight, **kwargs
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
        """LightGBM-compatible objective function (no mask)."""
        return self.get_lgb_objective(mask=None, **self._default_kwargs)

    def with_params(self, **kwargs: Any) -> MaskedMultiTaskObjective:
        """Create a new instance with default parameters set."""
        new_instance = MaskedMultiTaskObjective(
            task_loss_fn=self.task_loss_fn,
            n_tasks=self.n_tasks,
            task_weights=self.task_weights,
        )
        new_instance._default_kwargs = {**self._default_kwargs, **kwargs}
        return new_instance

    def __repr__(self) -> str:
        params = f", params={self._default_kwargs}" if self._default_kwargs else ""
        return f"MaskedMultiTaskObjective({self._name}, n_tasks={self.n_tasks}{params})"


def masked_multi_task_objective(
    n_tasks: int = 2,
    task_weights: list[float] | None = None,
) -> Callable[
    [Callable[[jax.Array, jax.Array], jax.Array]], MaskedMultiTaskObjective
]:
    """
    Decorator factory for masked multi-task objectives.

    Args:
        n_tasks: Number of tasks
        task_weights: Optional weights per task

    Example:
        >>> @masked_multi_task_objective(n_tasks=3, task_weights=[1.0, 2.0, 0.5])
        ... def my_mtl_loss(y_pred, y_true):
        ...     # Custom loss per task
        ...     return jnp.abs(y_pred - y_true)  # MAE instead of MSE
    """

    def decorator(
        func: Callable[[jax.Array, jax.Array], jax.Array]
    ) -> MaskedMultiTaskObjective:
        return MaskedMultiTaskObjective(
            task_loss_fn=func, n_tasks=n_tasks, task_weights=task_weights
        )

    return decorator


# =============================================================================
# Specialized Multi-Task Objectives
# =============================================================================


def multi_task_regression(n_tasks: int) -> MaskedMultiTaskObjective:
    """
    Standard multi-task regression with MSE loss.

    Supports missing labels via masking.

    Args:
        n_tasks: Number of regression tasks

    Example:
        >>> obj = multi_task_regression(n_tasks=5)
        >>> # Labels shape: (n_samples, 5), some can be NaN
        >>> mask = ~np.isnan(y_true)
        >>> dtrain.set_float_info('label_mask', mask.flatten())
    """
    return MaskedMultiTaskObjective(
        task_loss_fn=lambda y_pred, y_true: (y_pred - y_true) ** 2,
        n_tasks=n_tasks,
    )


def multi_task_classification(n_tasks: int) -> MaskedMultiTaskObjective:
    """
    Multi-task binary classification with log loss.

    Each task is an independent binary classification.
    Supports missing labels via masking.

    Args:
        n_tasks: Number of binary classification tasks

    Example:
        >>> obj = multi_task_classification(n_tasks=3)
        >>> # Each task: predict 0 or 1
    """

    def binary_logloss(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
        """Binary cross-entropy loss."""
        # y_pred is raw score (logit), y_true is 0 or 1
        # BCE = -y*log(sigmoid(s)) - (1-y)*log(1-sigmoid(s))
        #     = max(s, 0) - s*y + log(1 + exp(-|s|))
        return (
            jnp.maximum(y_pred, 0)
            - y_pred * y_true
            + jnp.log1p(jnp.exp(-jnp.abs(y_pred)))
        )

    return MaskedMultiTaskObjective(
        task_loss_fn=binary_logloss,
        n_tasks=n_tasks,
    )


def multi_task_huber(n_tasks: int, delta: float = 1.0) -> MaskedMultiTaskObjective:
    """
    Multi-task regression with Huber loss (robust to outliers).

    Args:
        n_tasks: Number of tasks
        delta: Threshold for switching between L1 and L2

    Example:
        >>> obj = multi_task_huber(n_tasks=3, delta=1.5)
    """

    def huber_loss(y_pred: jax.Array, y_true: jax.Array) -> jax.Array:
        error = y_pred - y_true
        abs_error = jnp.abs(error)
        quadratic = 0.5 * error**2
        linear = delta * abs_error - 0.5 * delta**2
        return jnp.where(abs_error <= delta, quadratic, linear)

    return MaskedMultiTaskObjective(
        task_loss_fn=huber_loss,
        n_tasks=n_tasks,
    )


def multi_task_quantile(
    n_tasks: int, quantiles: list[float] | None = None
) -> MaskedMultiTaskObjective:
    """
    Multi-task quantile regression.

    Each task predicts a different quantile. Useful for prediction intervals.

    Args:
        n_tasks: Number of quantiles to predict
        quantiles: List of quantile values (default: evenly spaced)

    Example:
        >>> # Predict 10th, 50th, 90th percentiles
        >>> obj = multi_task_quantile(n_tasks=3, quantiles=[0.1, 0.5, 0.9])
    """
    if quantiles is None:
        quantiles = list(np.linspace(0.1, 0.9, n_tasks))
    elif len(quantiles) != n_tasks:
        raise ValueError(f"quantiles length ({len(quantiles)}) must match n_tasks ({n_tasks})")

    quantiles_arr = jnp.asarray(quantiles, dtype=jnp.float32)

    # We need a way to pass the quantile to each task
    # Since task_loss_fn operates per-task, we'll use a trick:
    # Store quantiles and use task index

    class QuantileMTL(MaskedMultiTaskObjective):
        """Quantile MTL with per-task quantiles."""

        def __init__(self) -> None:
            super().__init__(n_tasks=n_tasks)
            self.quantiles = quantiles_arr
            self._name = "multi_task_quantile"

        def _compute_grad_hess_single(
            self,
            y_pred: jax.Array,
            y_true: jax.Array,
            mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            """Quantile loss with per-task quantiles."""
            error = y_true - y_pred

            # Gradient of quantile loss
            # Loss = q * max(error, 0) + (1-q) * max(-error, 0)
            # Grad w.r.t. y_pred:
            #   if error > 0: -q
            #   if error < 0: 1-q
            grads = jnp.where(error > 0, -self.quantiles, 1 - self.quantiles)

            # Hessian is 0 for quantile loss (piecewise linear)
            # Use small constant for XGBoost stability
            hess = jnp.ones_like(grads) * 1.0

            # Apply mask
            grads = grads * mask
            hess = hess * mask

            return grads, hess

    return QuantileMTL()

