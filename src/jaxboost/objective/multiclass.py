"""
Multi-class classification objective functions.

These objectives handle the specific requirements of multi-class classification
with XGBoost/LightGBM where predictions are logits of shape (n_samples, n_classes).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class MultiClassObjective:
    """
    Objective function wrapper for multi-class classification.

    Handles the specific requirements of multi-class classification with XGBoost/LightGBM:
    - Predictions are logits of shape (n_samples, n_classes)
    - Labels are integer class indices
    - Computes gradients w.r.t. each class logit

    Args:
        loss_fn: A loss function that takes (logits, label, **kwargs) where:
                 - logits: shape (n_classes,) raw scores for each class
                 - label: integer class index (0 to n_classes-1)
                 Returns a scalar loss.
        n_classes: Number of classes

    Example:
        >>> @MultiClassObjective(n_classes=3)
        ... def my_multiclass_loss(logits, label):
        ...     probs = jax.nn.softmax(logits)
        ...     return -jnp.log(probs[label] + 1e-10)
        >>>
        >>> params = {'num_class': 3}
        >>> model = xgb.train(params, dtrain, obj=my_multiclass_loss.xgb_objective)
    """

    def __init__(
        self,
        loss_fn: Callable[..., jax.Array] | None = None,
        n_classes: int = 3,
    ) -> None:
        self.n_classes = n_classes
        self._default_kwargs: dict[str, Any] = {}

        if loss_fn is not None:
            self._init_with_fn(loss_fn)
        else:
            self.loss_fn = None
            self._name = "multiclass_objective"

    def _init_with_fn(self, loss_fn: Callable[..., jax.Array]) -> None:
        """Initialize with a loss function."""
        self.loss_fn = loss_fn
        self._name = getattr(loss_fn, "__name__", "multiclass_objective")

        # Gradient function: returns gradient w.r.t. each class logit
        self._grad_fn = jax.grad(self._loss_wrapper, argnums=0)

        # Hessian diagonal
        def diag_hess(
            logits: jax.Array, label: jax.Array, kwargs: dict[str, Any]
        ) -> jax.Array:
            """Compute diagonal of Hessian."""

            def loss_i(i: int) -> Callable[[jax.Array], jax.Array]:
                def fn(li: jax.Array) -> jax.Array:
                    logits_new = logits.at[i].set(li)
                    return self._loss_wrapper(logits_new, label, kwargs)

                return fn

            def hess_ii(i: int) -> jax.Array:
                return jax.grad(jax.grad(loss_i(i)))(logits[i])

            return jax.vmap(hess_ii)(jnp.arange(self.n_classes))

        self._hess_fn = diag_hess

    def __call__(self, loss_fn: Callable[..., jax.Array]) -> MultiClassObjective:
        """Allow use as a decorator with arguments."""
        new_instance = MultiClassObjective(loss_fn=loss_fn, n_classes=self.n_classes)
        new_instance._default_kwargs = self._default_kwargs.copy()
        return new_instance

    def _loss_wrapper(
        self, logits: jax.Array, label: jax.Array, kwargs: dict[str, Any]
    ) -> jax.Array:
        """Wrapper for loss function."""
        return self.loss_fn(logits, label, **kwargs)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradient for multi-class predictions.

        Args:
            y_pred: Predictions, shape (n_samples, n_classes) or flattened
            y_true: True labels, shape (n_samples,) integer class indices
            **kwargs: Additional arguments

        Returns:
            Gradients, shape (n_samples, n_classes)
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}

        # Reshape predictions if needed
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_2d = y_pred_arr.reshape(-1, self.n_classes)
        else:
            y_pred_2d = y_pred_arr

        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.int32)

        # Compute gradients for each sample
        def grad_single(logits: jax.Array, label: jax.Array) -> jax.Array:
            return self._grad_fn(logits, label, merged_kwargs)

        grads = jax.vmap(grad_single)(y_pred_jax, y_true_jax)

        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        **kwargs: Any,
    ) -> NDArray[np.floating[Any]]:
        """
        Compute diagonal Hessian for multi-class predictions.

        Args:
            y_pred: Predictions, shape (n_samples, n_classes) or flattened
            y_true: True labels, shape (n_samples,) integer class indices
            **kwargs: Additional arguments

        Returns:
            Diagonal Hessians, shape (n_samples, n_classes)
        """
        merged_kwargs = {**self._default_kwargs, **kwargs}

        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 1:
            y_pred_2d = y_pred_arr.reshape(-1, self.n_classes)
        else:
            y_pred_2d = y_pred_arr
        y_pred_jax = jnp.asarray(y_pred_2d, dtype=jnp.float32)
        y_true_jax = jnp.asarray(y_true, dtype=jnp.int32)

        def hess_single(logits: jax.Array, label: jax.Array) -> jax.Array:
            return self._hess_fn(logits, label, merged_kwargs)

        hess = jax.vmap(hess_single)(y_pred_jax, y_true_jax)

        return np.asarray(hess, dtype=np.float64)

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
            y_pred: Predictions, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)
            sample_weight: Optional sample weights, shape (n_samples,)
            **kwargs: Additional arguments

        Returns:
            Tuple of (gradients, hessians), each shape (n_samples, n_classes)
        """
        grad = self.gradient(y_pred, y_true, **kwargs)
        hess = self.hessian(y_pred, y_true, **kwargs)

        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.asarray(sample_weight, dtype=np.float64)
            grad = grad * weight[:, np.newaxis]
            hess = hess * weight[:, np.newaxis]

        return grad, hess

    def get_xgb_objective(
        self, **kwargs: Any
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get an XGBoost-compatible objective function for multi-class.

        Use with XGBoost params: {'num_class': n_classes}

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

    def with_params(self, **kwargs: Any) -> MultiClassObjective:
        """Create a new instance with default parameters set."""
        new_instance = MultiClassObjective(
            loss_fn=self.loss_fn, n_classes=self.n_classes
        )
        new_instance._default_kwargs = {**self._default_kwargs, **kwargs}
        return new_instance

    def __repr__(self) -> str:
        params = f", params={self._default_kwargs}" if self._default_kwargs else ""
        return f"MultiClassObjective({self._name}, n_classes={self.n_classes}{params})"


def multiclass_objective(
    n_classes: int = 3,
) -> Callable[[Callable[..., jax.Array]], MultiClassObjective]:
    """
    Decorator factory for multi-class objective functions.

    Args:
        n_classes: Number of classes

    Example:
        >>> @multiclass_objective(n_classes=5)
        ... def my_loss(logits, label):
        ...     probs = jax.nn.softmax(logits)
        ...     return -jnp.log(probs[label] + 1e-10)
    """

    def decorator(func: Callable[..., jax.Array]) -> MultiClassObjective:
        return MultiClassObjective(loss_fn=func, n_classes=n_classes)

    return decorator


# =============================================================================
# Built-in Multi-class Objectives
# =============================================================================


def softmax_cross_entropy(n_classes: int = 3) -> MultiClassObjective:
    """
    Softmax Cross-Entropy Loss for multi-class classification.

    Standard cross-entropy loss with softmax activation.

    Args:
        n_classes: Number of classes

    Returns:
        MultiClassObjective instance

    Example:
        >>> softmax_loss = softmax_cross_entropy(n_classes=5)
        >>> params = {'num_class': 5}
        >>> model = xgb.train(params, dtrain, obj=softmax_loss.xgb_objective)
    """

    @multiclass_objective(n_classes=n_classes)
    def softmax_ce(logits: jax.Array, label: jax.Array) -> jax.Array:
        """Softmax cross-entropy for a single sample."""
        log_probs = jax.nn.log_softmax(logits)
        return -log_probs[label]

    return softmax_ce


def focal_multiclass(
    n_classes: int = 3,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> MultiClassObjective:
    """
    Focal Loss for multi-class classification.

    Extends focal loss to multi-class setting. Down-weights well-classified
    examples to focus on hard examples.

    Args:
        n_classes: Number of classes
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Optional class weight. If None, all classes weighted equally.

    Returns:
        MultiClassObjective instance

    Example:
        >>> focal_mc = focal_multiclass(n_classes=10, gamma=2.0)
        >>> model = xgb.train(params, dtrain, obj=focal_mc.xgb_objective)

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    @multiclass_objective(n_classes=n_classes)
    def focal_mc(logits: jax.Array, label: jax.Array) -> jax.Array:
        """Focal loss for a single sample."""
        probs = jax.nn.softmax(logits)
        probs = jnp.clip(probs, 1e-10, 1.0 - 1e-10)

        p_t = probs[label]
        focal_weight = (1 - p_t) ** gamma
        ce = -jnp.log(p_t)

        if alpha is not None:
            return alpha * focal_weight * ce
        return focal_weight * ce

    return focal_mc


def label_smoothing(
    n_classes: int = 3,
    smoothing: float = 0.1,
) -> MultiClassObjective:
    """
    Label Smoothing Cross-Entropy Loss.

    Softmax cross-entropy with label smoothing for regularization.

    Args:
        n_classes: Number of classes
        smoothing: Smoothing factor in [0, 1]. 0 = no smoothing.

    Returns:
        MultiClassObjective instance

    Example:
        >>> smooth_loss = label_smoothing(n_classes=10, smoothing=0.1)
        >>> model = xgb.train(params, dtrain, obj=smooth_loss.xgb_objective)
    """

    @multiclass_objective(n_classes=n_classes)
    def label_smooth(logits: jax.Array, label: jax.Array) -> jax.Array:
        """Label smoothing cross-entropy for a single sample."""
        log_probs = jax.nn.log_softmax(logits)

        smooth_weight = smoothing / (n_classes - 1)
        true_weight = 1.0 - smoothing

        loss = -true_weight * log_probs[label] - smooth_weight * (
            jnp.sum(log_probs) - log_probs[label]
        )

        return loss

    return label_smooth


def class_balanced(
    n_classes: int = 3,
    samples_per_class: NDArray[np.floating[Any]] | None = None,
    beta: float = 0.999,
) -> MultiClassObjective:
    """
    Class-Balanced Loss for long-tailed distributions.

    Re-weights classes based on effective number of samples.

    Args:
        n_classes: Number of classes
        samples_per_class: Array of sample counts per class. If None, uniform weights.
        beta: Hyperparameter for effective number. Higher = more aggressive reweighting.

    Returns:
        MultiClassObjective instance

    Example:
        >>> cb_loss = class_balanced(
        ...     n_classes=5,
        ...     samples_per_class=np.array([1000, 500, 100, 50, 10]),
        ...     beta=0.999
        ... )
        >>> model = xgb.train(params, dtrain, obj=cb_loss.xgb_objective)

    Reference:
        Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (2019)
    """
    if samples_per_class is not None:
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * n_classes
        weights_jax = jnp.asarray(weights, dtype=jnp.float32)
    else:
        weights_jax = jnp.ones(n_classes, dtype=jnp.float32)

    @multiclass_objective(n_classes=n_classes)
    def cb_ce(logits: jax.Array, label: jax.Array) -> jax.Array:
        """Class-balanced cross-entropy for a single sample."""
        log_probs = jax.nn.log_softmax(logits)
        weight = weights_jax[label]
        return -weight * log_probs[label]

    return cb_ce

