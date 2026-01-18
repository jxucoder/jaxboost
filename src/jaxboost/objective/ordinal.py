"""
Ordinal regression objective functions using Cumulative Link Models.

Ordinal regression handles ordered categorical outcomes (e.g., ratings 1-5,
quality grades) where the intervals between categories are unknown.

XGBoost has no native ordinal objective. Common workarounds lose information:
- Regression: Assumes equal intervals between categories
- Multi-class: Ignores ordering entirely

This module implements proper ordinal regression using the Cumulative Link Model
(also known as Proportional Odds Model), where:

    P(Y = k | x) = Φ(θ_{k+1} - g(x)) - Φ(θ_k - g(x))

Where:
- g(x) is the latent function learned by XGBoost (single scalar output)
- θ = [θ_1, ..., θ_{K-1}] are thresholds (with θ_0 = -∞, θ_K = +∞)
- Φ is the CDF of the link function (probit=normal, logit=sigmoid)

Reference:
    OGBoost: Ordinal Gradient Boosting (https://arxiv.org/pdf/2502.13456)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


class OrdinalObjective:
    """
    Cumulative Link Model objective for ordinal regression.

    Learns a single latent function g(x) that, combined with thresholds,
    produces ordinal class probabilities.

    Args:
        n_classes: Number of ordinal classes (K)
        link: Link function - 'probit' (normal CDF) or 'logit' (sigmoid)
        thresholds: Fixed thresholds array of shape (K-1,), or None to
                    initialize from data on first call

    Example:
        >>> # Wine quality: classes 3-8 (mapped to 0-5)
        >>> ordinal = OrdinalObjective(n_classes=6, link='probit')
        >>>
        >>> # Initialize thresholds from training data
        >>> ordinal.init_thresholds_from_data(y_train)
        >>>
        >>> # Train with XGBoost
        >>> model = xgb.train(params, dtrain, obj=ordinal.xgb_objective)
        >>>
        >>> # Predict class probabilities
        >>> latent = model.predict(dtest)
        >>> probs = ordinal.predict_proba(latent)
        >>> classes = ordinal.predict(latent)
    """

    def __init__(
        self,
        n_classes: int = 6,
        link: Literal["probit", "logit"] = "probit",
        thresholds: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.link = link
        self._thresholds: jax.Array | None = None

        if thresholds is not None:
            self.set_thresholds(thresholds)

        # Select CDF based on link function
        if link == "probit":
            self._cdf = jax.scipy.stats.norm.cdf
            self._pdf = jax.scipy.stats.norm.pdf
        elif link == "logit":
            self._cdf = jax.nn.sigmoid
            self._pdf = lambda x: jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))
        else:
            raise ValueError(f"Unknown link function: {link}. Use 'probit' or 'logit'.")

    def set_thresholds(self, thresholds: NDArray[np.floating[Any]]) -> None:
        """Set fixed thresholds for the ordinal model."""
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if len(thresholds) != self.n_classes - 1:
            raise ValueError(f"Expected {self.n_classes - 1} thresholds, got {len(thresholds)}")
        # Ensure thresholds are sorted
        if not np.all(np.diff(thresholds) > 0):
            raise ValueError("Thresholds must be strictly increasing")
        self._thresholds = jnp.asarray(thresholds, dtype=jnp.float32)

    def init_thresholds_from_data(self, y: NDArray[np.floating[Any]], eps: float = 0.01) -> None:
        """
        Initialize thresholds from empirical class distribution.

        Uses the inverse CDF of cumulative class proportions:
            θ_k = Φ^{-1}(Σ_{j<k} p_j)

        Args:
            y: Training labels, shape (n_samples,), values in [0, n_classes-1]
            eps: Small offset to avoid infinite thresholds at boundaries
        """
        from scipy import stats as scipy_stats

        y = np.asarray(y, dtype=np.int32)
        n_samples = len(y)

        # Count samples per class
        counts = np.bincount(y, minlength=self.n_classes)
        proportions = counts / n_samples

        # Cumulative proportions (excluding last class)
        cum_props = np.cumsum(proportions)[:-1]

        # Clip to avoid infinite values at 0 and 1
        cum_props = np.clip(cum_props, eps, 1 - eps)

        # Inverse CDF to get thresholds (use scipy for numerical stability)
        if self.link == "probit":
            thresholds = scipy_stats.norm.ppf(cum_props)
        else:  # logit
            # Inverse sigmoid: logit(p) = log(p / (1-p))
            thresholds = np.log(cum_props / (1 - cum_props))

        self._thresholds = jnp.asarray(thresholds, dtype=jnp.float32)

    @property
    def thresholds(self) -> jax.Array:
        """Get current thresholds (raises if not initialized)."""
        if self._thresholds is None:
            raise ValueError(
                "Thresholds not initialized. Call init_thresholds_from_data() or "
                "set_thresholds() first."
            )
        return self._thresholds

    def _get_extended_thresholds(self) -> jax.Array:
        """Get thresholds with large finite boundaries (avoid inf for autodiff)."""
        theta = self.thresholds
        # Use large but finite values instead of inf to avoid NaN in gradients
        return jnp.concatenate(
            [
                jnp.array([-30.0]),  # Effectively -inf for CDF purposes
                theta,
                jnp.array([30.0]),  # Effectively +inf for CDF purposes
            ]
        )

    def _class_probability(self, g: jax.Array, k: jax.Array) -> jax.Array:
        """
        Compute P(Y = k | g) for a single sample.

        P(Y = k) = Φ(θ_{k+1} - g) - Φ(θ_k - g)
        """
        theta_ext = self._get_extended_thresholds()
        cdf_upper = self._cdf(theta_ext[k + 1] - g)
        cdf_lower = self._cdf(theta_ext[k] - g)
        return cdf_upper - cdf_lower

    def _loss_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """
        Compute negative log-likelihood for a single sample.

        Loss = -log P(Y = y | g)
        """
        prob = self._class_probability(g, y)
        # Clip for numerical stability
        prob = jnp.clip(prob, 1e-10, 1.0)
        return -jnp.log(prob)

    def loss(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute loss for a batch of samples.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)
            y_true: True ordinal labels, shape (n_samples,), values in [0, K-1]

        Returns:
            Loss values, shape (n_samples,)
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        losses = jax.vmap(self._loss_single)(g, y)
        return np.asarray(losses, dtype=np.float64)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradient of loss w.r.t. latent predictions.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)
            y_true: True ordinal labels, shape (n_samples,)

        Returns:
            Gradients, shape (n_samples,)
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        grad_fn = jax.grad(self._loss_single, argnums=0)
        grads = jax.vmap(grad_fn)(g, y)

        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute Hessian (second derivative) of loss w.r.t. latent predictions.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)
            y_true: True ordinal labels, shape (n_samples,)

        Returns:
            Hessians, shape (n_samples,)
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        grad_fn = jax.grad(self._loss_single, argnums=0)
        hess_fn = jax.grad(grad_fn, argnums=0)
        hess = jax.vmap(hess_fn)(g, y)

        # Ensure positive Hessian for XGBoost stability
        hess = jnp.maximum(hess, 1e-6)

        return np.asarray(hess, dtype=np.float64)

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Compute both gradient and Hessian efficiently.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)
            y_true: True ordinal labels, shape (n_samples,)
            sample_weight: Optional sample weights, shape (n_samples,)

        Returns:
            Tuple of (gradients, hessians), each shape (n_samples,)
        """
        grad = self.gradient(y_pred, y_true)
        hess = self.hessian(y_pred, y_true)

        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.asarray(sample_weight, dtype=np.float64)
            grad = grad * weight
            hess = hess * weight

        return grad, hess

    def get_xgb_objective(
        self,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """
        Get an XGBoost-compatible objective function.

        Returns:
            XGBoost objective function: (y_pred, dtrain) -> (grad, hess)

        Example:
            >>> ordinal = OrdinalObjective(n_classes=6)
            >>> ordinal.init_thresholds_from_data(y_train)
            >>> model = xgb.train(params, dtrain, obj=ordinal.get_xgb_objective())
        """

        def objective(
            y_pred: NDArray[np.floating[Any]], dtrain: Any
        ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()
            return self.grad_hess(y_pred, y_true, sample_weight=sample_weight)

        objective.__name__ = f"ordinal_{self.link}_xgb_objective"
        return objective

    @property
    def xgb_objective(
        self,
    ) -> Callable[
        [NDArray[np.floating[Any]], Any],
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],
    ]:
        """XGBoost-compatible objective function."""
        return self.get_xgb_objective()

    def predict_proba(self, y_pred: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Convert latent predictions to class probabilities.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)

        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        theta_ext = self._get_extended_thresholds()

        def probs_single(g_i: jax.Array) -> jax.Array:
            """Compute all class probabilities for one sample."""
            cdfs = self._cdf(theta_ext - g_i)
            # P(Y = k) = CDF(θ_{k+1} - g) - CDF(θ_k - g)
            return cdfs[1:] - cdfs[:-1]

        probs = jax.vmap(probs_single)(g)
        return np.asarray(probs, dtype=np.float64)

    def predict(self, y_pred: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        """
        Convert latent predictions to class labels.

        Args:
            y_pred: Latent predictions g(x), shape (n_samples,)

        Returns:
            Predicted class labels, shape (n_samples,)
        """
        probs = self.predict_proba(y_pred)
        return np.argmax(probs, axis=1)

    # =========================================================================
    # Metric Properties
    # =========================================================================

    @property
    def qwk_metric(self) -> Any:
        """
        Get Quadratic Weighted Kappa metric for this ordinal objective.

        The metric uses this objective's predict() method to transform
        raw predictions to class labels.

        Returns:
            Metric object with .xgb_metric and .lgb_metric methods

        Example:
            >>> model = xgb.train(
            ...     {'disable_default_eval_metric': 1},
            ...     dtrain, obj=ordinal.xgb_objective,
            ...     custom_metric=ordinal.qwk_metric.xgb_metric,
            ...     evals=[(dtest, 'test')]
            ... )
        """
        from jaxboost.metric.ordinal import qwk_metric

        return qwk_metric(n_classes=self.n_classes, transform=self.predict)

    @property
    def mae_metric(self) -> Any:
        """
        Get Mean Absolute Error metric for this ordinal objective.

        Returns:
            Metric object with .xgb_metric and .lgb_metric methods
        """
        from jaxboost.metric.ordinal import ordinal_mae_metric

        return ordinal_mae_metric(n_classes=self.n_classes, transform=self.predict)

    @property
    def accuracy_metric(self) -> Any:
        """
        Get exact accuracy metric for this ordinal objective.

        Returns:
            Metric object with .xgb_metric and .lgb_metric methods
        """
        from jaxboost.metric.ordinal import ordinal_accuracy_metric

        return ordinal_accuracy_metric(n_classes=self.n_classes, transform=self.predict)

    @property
    def adjacent_accuracy_metric(self) -> Any:
        """
        Get adjacent accuracy (within ±1) metric for this ordinal objective.

        Returns:
            Metric object with .xgb_metric and .lgb_metric methods
        """
        from jaxboost.metric.ordinal import adjacent_accuracy_metric

        return adjacent_accuracy_metric(n_classes=self.n_classes, transform=self.predict)

    def __repr__(self) -> str:
        thresh_str = "initialized" if self._thresholds is not None else "not set"
        return (
            f"OrdinalObjective(n_classes={self.n_classes}, "
            f"link='{self.link}', thresholds={thresh_str})"
        )


def ordinal_regression(
    n_classes: int,
    link: Literal["probit", "logit"] = "probit",
    thresholds: NDArray[np.floating[Any]] | None = None,
) -> OrdinalObjective:
    """
    Create an ordinal regression objective.

    Convenience function for creating OrdinalObjective instances.

    Args:
        n_classes: Number of ordinal classes
        link: Link function - 'probit' or 'logit'
        thresholds: Optional fixed thresholds

    Returns:
        OrdinalObjective instance

    Example:
        >>> # Wine quality (6 classes: 3-8 mapped to 0-5)
        >>> obj = ordinal_regression(n_classes=6, link='probit')
        >>> obj.init_thresholds_from_data(y_train)
        >>> model = xgb.train(params, dtrain, obj=obj.xgb_objective)
    """
    return OrdinalObjective(n_classes=n_classes, link=link, thresholds=thresholds)


def ordinal_probit(n_classes: int) -> OrdinalObjective:
    """Create an ordinal regression objective with probit link."""
    return OrdinalObjective(n_classes=n_classes, link="probit")


def ordinal_logit(n_classes: int) -> OrdinalObjective:
    """Create an ordinal regression objective with logit link."""
    return OrdinalObjective(n_classes=n_classes, link="logit")


class SquaredCDFObjective(OrdinalObjective):
    """
    Squared CDF Loss (Continuous Ranked Probability Score - CRPS).

    Minimizes the squared Earth Mover's Distance (Wasserstein-2) between
    predicted CDF and true cumulative label distribution:

        L = Σ (P(Y ≤ k|x) - I(y ≤ k))²

    This is better for QWK than EQE/MSE because:
    1. It penalizes distance properly (like MSE/QWK numerator)
    2. It matches the FULL distribution (unlike EQE which only matches mean)
    3. It preserves prediction variance, helping the QWK denominator
    4. It is strictly proper and convex

    Args:
        n_classes: Number of ordinal classes (K)
        link: Link function - 'probit' or 'logit'
        use_gauss_newton: Use Gauss-Newton Hessian (default True)
        thresholds: Optional fixed thresholds
    """

    def __init__(
        self,
        n_classes: int = 6,
        link: Literal["probit", "logit"] = "logit",
        use_gauss_newton: bool = True,
        thresholds: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, link=link, thresholds=thresholds)
        self.use_gauss_newton = use_gauss_newton

    def _loss_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """
        Squared CDF Loss for a single sample.
        L = Σ_k (F(k) - I(y <= k))²
        """
        theta_ext = self._get_extended_thresholds()
        # F(k) = P(Y <= k) = CDF(θ_{k+1} - g)
        # Note: theta_ext indices: 0:-inf, 1:θ_1, ..., K-1:θ_{K-1}, K:inf
        # We need P(Y<=k) for k=0..K-2 (last class K-1 always has P(Y<=K-1)=1, const)

        # Calculate P(Y <= k) for k = 0 to K-2
        # Corresponds to thresholds θ_1 to θ_{K-1}
        # indices 1 to K-1 in theta_ext
        thresholds_relevant = theta_ext[1:-1]
        pred_cdfs = self._cdf(thresholds_relevant - g)

        # True CDFs: I(y <= k) for k = 0 to K-2
        class_indices = jnp.arange(self.n_classes - 1, dtype=jnp.float32)
        true_cdfs = (class_indices >= y).astype(jnp.float32)

        return jnp.sum((pred_cdfs - true_cdfs) ** 2)

    def _grad_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """Analytical gradient of Squared CDF Loss."""
        # Use JAX autodiff for simplicity and correctness
        return jax.grad(self._loss_single)(g, y)

    def _hess_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """Hessian of Squared CDF Loss."""
        if self.use_gauss_newton:
            # Gauss-Newton: H ≈ 2 * Σ (∂F/∂g)²
            theta_ext = self._get_extended_thresholds()
            thresholds_relevant = theta_ext[1:-1]

            # ∂F/∂g = -pdf(θ - g)
            pdfs = self._pdf(thresholds_relevant - g)
            grad_F = -pdfs  # noqa: N806

            return 2.0 * jnp.sum(grad_F**2)
        else:
            # Full Hessian via autodiff
            return jax.grad(jax.grad(self._loss_single))(g, y)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        grads = jax.vmap(self._grad_single)(g, y)
        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        hess = jax.vmap(self._hess_single)(g, y)
        hess = jnp.maximum(hess, 1e-6)  # XGBoost stability
        return np.asarray(hess, dtype=np.float64)

    def __repr__(self) -> str:
        thresh_str = "initialized" if self._thresholds is not None else "not set"
        return (
            f"SquaredCDFObjective(n_classes={self.n_classes}, link='{self.link}', "
            f"gauss_newton={self.use_gauss_newton}, thresholds={thresh_str})"
        )


def squared_cdf_ordinal(
    n_classes: int,
    link: Literal["probit", "logit"] = "logit",
) -> SquaredCDFObjective:
    """
    Create a Squared CDF (CRPS) ordinal objective.

    This minimizes the squared Earth Mover's Distance between distributions,
    often outperforming EQE/NLL for QWK optimization.
    """
    return SquaredCDFObjective(n_classes=n_classes, link=link)


class QWKOrdinalObjective(OrdinalObjective):
    """
    QWK-aligned ordinal objective using Expected Quadratic Error (EQE).

    QWK is non-differentiable, but minimizing Expected Squared Error of
    the predicted ordinal score is a provably aligned surrogate:

        L_EQE = (y - ŷ)²  where  ŷ = Σ k·p_k(x)

    This works because:
    - QWK penalizes squared ordinal distance
    - EQE minimizes the same quadratic penalty in expectation
    - Fully differentiable via JAX autodiff

    Implementation notes (following XGBoost custom objective best practices):
    - EQE is non-convex, so we use Gauss-Newton Hessian approximation
    - H_GN = 2·(∂ŷ/∂g)² is always positive (drops problematic ∂²ŷ/∂g² term)
    - This matches standard practice for squared error losses

    Can optionally combine with NLL for stability:

        L = α·NLL + β·EQE

    Args:
        n_classes: Number of ordinal classes (K)
        link: Link function - 'probit' or 'logit'
        alpha: Weight for NLL loss (default 0.0 = pure EQE)
        beta: Weight for EQE loss (default 1.0)
        use_gauss_newton: Use Gauss-Newton Hessian for EQE (default True, recommended)
        thresholds: Optional fixed thresholds

    Example:
        >>> # Pure EQE (QWK-aligned)
        >>> obj = QWKOrdinalObjective(n_classes=7, alpha=0.0, beta=1.0)
        >>>
        >>> # Hybrid: 70% NLL + 30% EQE (stable + QWK-aligned)
        >>> obj = QWKOrdinalObjective(n_classes=7, alpha=0.7, beta=0.3)
    """

    def __init__(
        self,
        n_classes: int = 6,
        link: Literal["probit", "logit"] = "logit",
        alpha: float = 0.0,
        beta: float = 1.0,
        use_gauss_newton: bool = True,
        thresholds: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, link=link, thresholds=thresholds)
        self.alpha = alpha  # NLL weight
        self.beta = beta  # EQE weight
        self.use_gauss_newton = use_gauss_newton

        # Precompute class indices for expected score calculation
        self._class_indices = jnp.arange(n_classes, dtype=jnp.float32)

    def _all_class_probs(self, g: jax.Array) -> jax.Array:
        """Compute P(Y=k|g) for all classes k."""
        theta_ext = self._get_extended_thresholds()
        cdfs = self._cdf(theta_ext - g)
        return cdfs[1:] - cdfs[:-1]

    def _expected_score(self, g: jax.Array) -> jax.Array:
        """Compute expected ordinal score: ŷ = Σ k·p_k."""
        probs = self._all_class_probs(g)
        # Ensure probs are valid (clip for numerical stability)
        probs = jnp.clip(probs, 1e-10, 1.0)
        probs = probs / probs.sum()  # Renormalize
        return jnp.dot(self._class_indices, probs)

    def _eqe_loss_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """Expected Quadratic Error: (y - ŷ)²."""
        y_hat = self._expected_score(g)
        y_float = y.astype(jnp.float32)
        return (y_float - y_hat) ** 2

    def _eqe_grad_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """
        Gradient of EQE loss.

        ∂L/∂g = 2(ŷ - y) · ∂ŷ/∂g
        """
        y_hat = self._expected_score(g)
        y_float = y.astype(jnp.float32)
        residual = y_hat - y_float

        # Compute ∂ŷ/∂g via autodiff
        dy_hat_dg = jax.grad(self._expected_score)(g)

        return 2.0 * residual * dy_hat_dg

    def _eqe_gauss_newton_hess_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """
        Gauss-Newton approximation of EQE Hessian.

        For L = (y - ŷ)²:
        True Hessian: H = 2(∂ŷ/∂g)² + 2(ŷ - y)·∂²ŷ/∂g²

        Gauss-Newton drops the second term (which can be negative):
        H_GN = 2(∂ŷ/∂g)²

        This is always positive and is standard practice for squared error losses.
        At optimum (residual=0), H_GN = H (true Hessian).
        """
        dy_hat_dg = jax.grad(self._expected_score)(g)
        return 2.0 * dy_hat_dg**2

    def _loss_single(self, g: jax.Array, y: jax.Array) -> jax.Array:
        """
        Combined loss: α·NLL + β·EQE.

        - α=0, β=1: Pure EQE (QWK-aligned)
        - α=1, β=0: Pure NLL (standard CLM)
        - α=0.7, β=0.3: Hybrid (stable + QWK-aligned)
        """
        loss = jnp.array(0.0)

        if self.alpha > 0:
            # NLL loss from parent class
            prob = self._class_probability(g, y)
            prob = jnp.clip(prob, 1e-10, 1.0)
            nll = -jnp.log(prob)
            loss = loss + self.alpha * nll

        if self.beta > 0:
            eqe = self._eqe_loss_single(g, y)
            loss = loss + self.beta * eqe

        return loss

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute gradient of loss w.r.t. latent predictions.

        For hybrid loss: grad = α·grad_NLL + β·grad_EQE
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        grad = jnp.zeros_like(g)

        if self.alpha > 0:
            # NLL gradient from parent class
            nll_grad_fn = jax.grad(
                lambda g_i, y_i: super(QWKOrdinalObjective, self)._loss_single(g_i, y_i), argnums=0
            )
            nll_grads = jax.vmap(nll_grad_fn)(g, y)
            grad = grad + self.alpha * nll_grads

        if self.beta > 0:
            eqe_grads = jax.vmap(self._eqe_grad_single)(g, y)
            grad = grad + self.beta * eqe_grads

        return np.asarray(grad, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """
        Compute Hessian of loss w.r.t. latent predictions.

        Uses Gauss-Newton approximation for EQE (always positive).
        For hybrid loss: hess = α·hess_NLL + β·hess_EQE_GN
        """
        g = jnp.asarray(y_pred, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        hess = jnp.zeros_like(g)

        if self.alpha > 0:
            # NLL Hessian from parent class (true Hessian, always positive for NLL)
            nll_grad_fn = jax.grad(
                lambda g_i, y_i: super(QWKOrdinalObjective, self)._loss_single(g_i, y_i), argnums=0
            )
            nll_hess_fn = jax.grad(nll_grad_fn, argnums=0)
            nll_hess = jax.vmap(nll_hess_fn)(g, y)
            hess = hess + self.alpha * nll_hess

        if self.beta > 0:
            if self.use_gauss_newton:
                # Gauss-Newton approximation (always positive)
                eqe_hess = jax.vmap(self._eqe_gauss_newton_hess_single)(g, y)
            else:
                # True Hessian (can be negative)
                eqe_grad_fn = jax.grad(self._eqe_loss_single, argnums=0)
                eqe_hess_fn = jax.grad(eqe_grad_fn, argnums=0)
                eqe_hess = jax.vmap(eqe_hess_fn)(g, y)
            hess = hess + self.beta * eqe_hess

        # Ensure positive Hessian for XGBoost stability
        hess = jnp.maximum(hess, 1e-6)

        return np.asarray(hess, dtype=np.float64)

    def __repr__(self) -> str:
        thresh_str = "initialized" if self._thresholds is not None else "not set"
        return (
            f"QWKOrdinalObjective(n_classes={self.n_classes}, link='{self.link}', "
            f"alpha={self.alpha}, beta={self.beta}, gauss_newton={self.use_gauss_newton}, "
            f"thresholds={thresh_str})"
        )


def qwk_ordinal(
    n_classes: int,
    link: Literal["probit", "logit"] = "logit",
    alpha: float = 0.0,
    beta: float = 1.0,
) -> QWKOrdinalObjective:
    """
    Create a QWK-aligned ordinal objective.

    Uses Expected Quadratic Error (EQE) as a differentiable surrogate for QWK.

    Args:
        n_classes: Number of ordinal classes
        link: Link function - 'probit' or 'logit'
        alpha: Weight for NLL loss (0 = no NLL)
        beta: Weight for EQE loss (1 = full EQE)

    Returns:
        QWKOrdinalObjective instance

    Example:
        >>> # Pure EQE (best QWK alignment)
        >>> obj = qwk_ordinal(n_classes=7)
        >>>
        >>> # Hybrid for stability
        >>> obj = qwk_ordinal(n_classes=7, alpha=0.5, beta=0.5)
    """
    return QWKOrdinalObjective(n_classes=n_classes, link=link, alpha=alpha, beta=beta)


def hybrid_ordinal(
    n_classes: int,
    link: Literal["probit", "logit"] = "logit",
    nll_weight: float = 0.7,
    eqe_weight: float = 0.3,
) -> QWKOrdinalObjective:
    """
    Create a hybrid ordinal objective (NLL + EQE).

    Combines:
    - NLL: Proper probabilistic loss (stable gradients early in training)
    - EQE: QWK-aligned loss (better metric alignment)

    Default weights (0.7/0.3) work well in practice.

    Args:
        n_classes: Number of ordinal classes
        link: Link function
        nll_weight: Weight for NLL loss
        eqe_weight: Weight for EQE loss

    Returns:
        QWKOrdinalObjective instance
    """
    return QWKOrdinalObjective(n_classes=n_classes, link=link, alpha=nll_weight, beta=eqe_weight)


# =============================================================================
# SLACE Paper Objectives (AAAI 2025)
# =============================================================================


class SORDObjective:
    """
    SORD: Soft Ordinal Regression (Diaz & Marathe, CVPR 2019).

    Creates soft target labels based on distance from true class:
        target(k) = softmax(-alpha * |k - y|)
        L = -sum(target(k) * log(p(k)))

    This is essentially label smoothing that respects ordinal structure.

    Args:
        n_classes: Number of ordinal classes
        alpha: Temperature for soft labels (higher = sharper, default=1.0)
    """

    def __init__(self, n_classes: int, alpha: float = 1.0) -> None:
        self.n_classes = n_classes
        self.alpha = alpha
        self._class_indices = jnp.arange(n_classes, dtype=jnp.float32)

    def _ensure_2d(
        self, y_pred: NDArray[np.floating[Any]], n_samples: int
    ) -> NDArray[np.floating[Any]]:
        """Ensure y_pred is 2D with shape (n_samples, n_classes)."""
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.size == n_samples * self.n_classes:
            return y_pred_arr.reshape(n_samples, self.n_classes)
        # First iteration: expand to multi-class
        return np.zeros((n_samples, self.n_classes), dtype=np.float64)

    def _soft_targets(self, y: jax.Array) -> jax.Array:
        """Generate soft target distribution centered on true class y."""
        distances = jnp.abs(self._class_indices - y.astype(jnp.float32))
        return jax.nn.softmax(-self.alpha * distances)

    def _loss_single(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """SORD loss for a single sample."""
        probs = jax.nn.softmax(logits)
        targets = self._soft_targets(y)
        # Cross entropy with soft targets
        return -jnp.sum(targets * jnp.log(jnp.clip(probs, 1e-10, 1.0)))

    def loss(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        losses = jax.vmap(self._loss_single)(logits, y)
        return np.asarray(losses, dtype=np.float64)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        grad_fn = jax.grad(self._loss_single, argnums=0)
        grads = jax.vmap(grad_fn)(logits, y)
        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def hess_diag_single(logits_i: jax.Array, y_i: jax.Array) -> jax.Array:
            grad_fn = jax.grad(self._loss_single, argnums=0)
            hess_fn = jax.jacfwd(grad_fn, argnums=0)
            H = hess_fn(logits_i, y_i)  # noqa: N806
            return jnp.diag(H)

        hess = jax.vmap(hess_diag_single)(logits, y)
        hess = jnp.maximum(hess, 1e-6)
        return np.asarray(hess, dtype=np.float64)

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        grad = self.gradient(y_pred, y_true)
        hess = self.hessian(y_pred, y_true)
        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.repeat(sample_weight, self.n_classes).reshape(grad.shape)
            grad = grad * weight
            hess = hess * weight
        return grad, hess

    def get_xgb_objective(self) -> Callable:
        def objective(y_pred, dtrain):
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()
            return self.grad_hess(y_pred, y_true, sample_weight)

        objective.__name__ = "sord_xgb_objective"
        return objective

    @property
    def xgb_objective(self) -> Callable:
        return self.get_xgb_objective()

    def _probs_grad_hess(
        self,
        y_true: NDArray[np.floating[Any]],
        probs: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute gradient/hessian w.r.t. softmax probabilities (for sklearn API)."""
        probs_jax = jnp.asarray(probs, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def loss_single_probs(p: jax.Array, y_i: jax.Array) -> jax.Array:
            """Loss for a single sample given probabilities."""
            targets = self._soft_targets(y_i)
            return -jnp.sum(targets * jnp.log(jnp.clip(p, 1e-10, 1.0)))

        grad_fn = jax.grad(loss_single_probs, argnums=0)
        grads = jax.vmap(grad_fn)(probs_jax, y)
        # SLACE paper uses constant hessian
        hess = np.ones_like(grads)
        return np.asarray(grads, dtype=np.float64), hess

    @property
    def sklearn_objective(self) -> Callable:
        """Objective for XGBClassifier (receives softmax probs, not logits)."""

        def objective(y_true, probs):
            grad, hess = self._probs_grad_hess(y_true, probs)
            # XGBoost 2.1+ requires (n_samples, n_classes) shape
            return grad, hess

        return objective

    def predict(self, y_pred: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        n_samples = len(y_pred) // self.n_classes
        logits = np.asarray(y_pred).reshape(n_samples, self.n_classes)
        return np.argmax(logits, axis=1)

    def __repr__(self) -> str:
        return f"SORDObjective(n_classes={self.n_classes}, alpha={self.alpha})"


class OLLObjective:
    """
    OLL: Ordinal Log Loss (Castagnos et al., COLING 2022).

    Distance-weighted log loss that penalizes errors by ordinal distance:
        L = -sum(log(1 - p(k)) * |k - y|^alpha)

    Args:
        n_classes: Number of ordinal classes
        alpha: Distance exponent (default=1.0)
    """

    def __init__(self, n_classes: int, alpha: float = 1.0) -> None:
        self.n_classes = n_classes
        self.alpha = alpha
        self._class_indices = jnp.arange(n_classes, dtype=jnp.float32)

    def _ensure_2d(
        self, y_pred: NDArray[np.floating[Any]], n_samples: int
    ) -> NDArray[np.floating[Any]]:
        """Ensure y_pred is 2D with shape (n_samples, n_classes)."""
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.size == n_samples * self.n_classes:
            return y_pred_arr.reshape(n_samples, self.n_classes)
        return np.zeros((n_samples, self.n_classes), dtype=np.float64)

    def _loss_single(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """OLL loss for a single sample."""
        probs = jax.nn.softmax(logits)
        distances = jnp.abs(self._class_indices - y.astype(jnp.float32))
        weights = distances**self.alpha
        # Penalize probabilities on wrong classes, weighted by distance
        return -jnp.sum(weights * jnp.log(jnp.clip(1 - probs, 1e-10, 1.0)))

    def loss(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        losses = jax.vmap(self._loss_single)(logits, y)
        return np.asarray(losses, dtype=np.float64)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        grad_fn = jax.grad(self._loss_single, argnums=0)
        grads = jax.vmap(grad_fn)(logits, y)
        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def hess_diag_single(logits_i: jax.Array, y_i: jax.Array) -> jax.Array:
            grad_fn = jax.grad(self._loss_single, argnums=0)
            hess_fn = jax.jacfwd(grad_fn, argnums=0)
            H = hess_fn(logits_i, y_i)  # noqa: N806
            return jnp.diag(H)

        hess = jax.vmap(hess_diag_single)(logits, y)
        hess = jnp.maximum(hess, 1e-6)
        return np.asarray(hess, dtype=np.float64)

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        grad = self.gradient(y_pred, y_true)
        hess = self.hessian(y_pred, y_true)
        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.repeat(sample_weight, self.n_classes).reshape(grad.shape)
            grad = grad * weight
            hess = hess * weight
        return grad, hess

    def get_xgb_objective(self) -> Callable:
        def objective(y_pred, dtrain):
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()
            return self.grad_hess(y_pred, y_true, sample_weight)

        objective.__name__ = "oll_xgb_objective"
        return objective

    @property
    def xgb_objective(self) -> Callable:
        return self.get_xgb_objective()

    def _probs_grad_hess(
        self,
        y_true: NDArray[np.floating[Any]],
        probs: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute gradient/hessian w.r.t. softmax probabilities (for sklearn API)."""
        probs_jax = jnp.asarray(probs, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def loss_single_probs(p: jax.Array, y_i: jax.Array) -> jax.Array:
            """OLL loss for a single sample given probabilities."""
            distances = jnp.abs(self._class_indices - y_i.astype(jnp.float32))
            weights = distances**self.alpha
            return -jnp.sum(weights * jnp.log(jnp.clip(1 - p, 1e-10, 1.0)))

        grad_fn = jax.grad(loss_single_probs, argnums=0)
        grads = jax.vmap(grad_fn)(probs_jax, y)
        hess = np.ones_like(grads)
        return np.asarray(grads, dtype=np.float64), hess

    @property
    def sklearn_objective(self) -> Callable:
        """Objective for XGBClassifier (receives softmax probs, not logits)."""

        def objective(y_true, probs):
            grad, hess = self._probs_grad_hess(y_true, probs)
            return grad, hess

        return objective

    def predict(self, y_pred: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        n_samples = len(y_pred) // self.n_classes
        logits = np.asarray(y_pred).reshape(n_samples, self.n_classes)
        return np.argmax(logits, axis=1)

    def __repr__(self) -> str:
        return f"OLLObjective(n_classes={self.n_classes}, alpha={self.alpha})"


class SLACEObjective:
    """
    SLACE: Soft Labels Accumulating Cross Entropy (AAAI 2025).

    Key innovation: Uses ACCUMULATED probabilities with soft targets.
    This enforces monotonicity and balance-sensitivity.

    For each true class y, build dominance matrix D[y] where:
        D[y][i,j] = 1 if class j is closer to y than class i

    Accumulated probs: acc_p = D[y] @ softmax(logits)
    Loss: L = -sum(soft_target(k) * log(acc_p(k)))

    Args:
        n_classes: Number of ordinal classes
        alpha: Temperature for soft labels (default=1.0)
    """

    def __init__(self, n_classes: int, alpha: float = 1.0) -> None:
        self.n_classes = n_classes
        self.alpha = alpha
        self._class_indices = jnp.arange(n_classes, dtype=jnp.float32)
        # Precompute dominance matrices for each class
        self._dom_matrices = self._build_dominance_matrices()

    def _ensure_2d(
        self, y_pred: NDArray[np.floating[Any]], n_samples: int
    ) -> NDArray[np.floating[Any]]:
        """Ensure y_pred is 2D with shape (n_samples, n_classes)."""
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.size == n_samples * self.n_classes:
            return y_pred_arr.reshape(n_samples, self.n_classes)
        return np.zeros((n_samples, self.n_classes), dtype=np.float64)

    def _build_dominance_matrices(self) -> jax.Array:
        """Build dominance matrices D[y] for each true class y."""
        K = self.n_classes  # noqa: N806
        matrices = []
        for y in range(K):
            D = np.zeros((K, K), dtype=np.float32)  # noqa: N806
            for i in range(K):
                for j in range(K):
                    dist_i = abs(i - y)
                    dist_j = abs(j - y)
                    if dist_j <= dist_i:
                        D[i, j] = 1.0
            matrices.append(D)
        return jnp.array(matrices)

    def _soft_targets(self, y: jax.Array) -> jax.Array:
        """Generate soft target distribution centered on true class y."""
        distances = jnp.abs(self._class_indices - y.astype(jnp.float32))
        return jax.nn.softmax(-self.alpha * distances)

    def _loss_single(self, logits: jax.Array, y: jax.Array) -> jax.Array:
        """SLACE loss for a single sample."""
        probs = jax.nn.softmax(logits)
        targets = self._soft_targets(y)

        # Get dominance matrix for this true class
        D = self._dom_matrices[y]  # noqa: N806

        # Accumulated probabilities
        acc_probs = D @ probs

        # Cross entropy with soft targets on accumulated probs
        return -jnp.sum(targets * jnp.log(jnp.clip(acc_probs, 1e-10, 1.0)))

    def loss(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        losses = jax.vmap(self._loss_single)(logits, y)
        return np.asarray(losses, dtype=np.float64)

    def gradient(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)
        grad_fn = jax.grad(self._loss_single, argnums=0)
        grads = jax.vmap(grad_fn)(logits, y)
        return np.asarray(grads, dtype=np.float64)

    def hessian(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        n_samples = len(y_true)
        logits = jnp.asarray(self._ensure_2d(y_pred, n_samples), dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def hess_diag_single(logits_i: jax.Array, y_i: jax.Array) -> jax.Array:
            grad_fn = jax.grad(self._loss_single, argnums=0)
            hess_fn = jax.jacfwd(grad_fn, argnums=0)
            H = hess_fn(logits_i, y_i)  # noqa: N806
            return jnp.diag(H)

        hess = jax.vmap(hess_diag_single)(logits, y)
        hess = jnp.maximum(hess, 1e-6)
        return np.asarray(hess, dtype=np.float64)

    def grad_hess(
        self,
        y_pred: NDArray[np.floating[Any]],
        y_true: NDArray[np.floating[Any]],
        sample_weight: NDArray[np.floating[Any]] | None = None,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        grad = self.gradient(y_pred, y_true)
        hess = self.hessian(y_pred, y_true)
        if sample_weight is not None and len(sample_weight) > 0:
            weight = np.repeat(sample_weight, self.n_classes).reshape(grad.shape)
            grad = grad * weight
            hess = hess * weight
        return grad, hess

    def get_xgb_objective(self) -> Callable:
        def objective(y_pred, dtrain):
            y_true = dtrain.get_label()
            sample_weight = dtrain.get_weight()
            return self.grad_hess(y_pred, y_true, sample_weight)

        objective.__name__ = "slace_xgb_objective"
        return objective

    @property
    def xgb_objective(self) -> Callable:
        return self.get_xgb_objective()

    def _probs_grad_hess(
        self,
        y_true: NDArray[np.floating[Any]],
        probs: NDArray[np.floating[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute gradient/hessian w.r.t. softmax probabilities (for sklearn API)."""
        probs_jax = jnp.asarray(probs, dtype=jnp.float32)
        y = jnp.asarray(y_true, dtype=jnp.int32)

        def loss_single_probs(p: jax.Array, y_i: jax.Array) -> jax.Array:
            """SLACE loss for a single sample given probabilities."""
            targets = self._soft_targets(y_i)
            D = self._dom_matrices[y_i]  # noqa: N806
            acc_probs = D @ p
            return -jnp.sum(targets * jnp.log(jnp.clip(acc_probs, 1e-10, 1.0)))

        grad_fn = jax.grad(loss_single_probs, argnums=0)
        grads = jax.vmap(grad_fn)(probs_jax, y)
        hess = np.ones_like(grads)
        return np.asarray(grads, dtype=np.float64), hess

    @property
    def sklearn_objective(self) -> Callable:
        """Objective for XGBClassifier (receives softmax probs, not logits)."""

        def objective(y_true, probs):
            grad, hess = self._probs_grad_hess(y_true, probs)
            return grad, hess

        return objective

    def predict(self, y_pred: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
        n_samples = len(y_pred) // self.n_classes
        logits = np.asarray(y_pred).reshape(n_samples, self.n_classes)
        return np.argmax(logits, axis=1)

    def __repr__(self) -> str:
        return f"SLACEObjective(n_classes={self.n_classes}, alpha={self.alpha})"


# Factory functions
def sord_objective(n_classes: int, alpha: float = 1.0) -> SORDObjective:
    """Create SORD (Soft Ordinal) objective."""
    return SORDObjective(n_classes=n_classes, alpha=alpha)


def oll_objective(n_classes: int, alpha: float = 1.0) -> OLLObjective:
    """Create OLL (Ordinal Log Loss) objective."""
    return OLLObjective(n_classes=n_classes, alpha=alpha)


def slace_objective(n_classes: int, alpha: float = 1.0) -> SLACEObjective:
    """Create SLACE (Soft Labels Accumulating Cross Entropy) objective."""
    return SLACEObjective(n_classes=n_classes, alpha=alpha)
