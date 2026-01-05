"""
Tests for the core AutoObjective class.

Tests gradient and Hessian correctness via numerical differentiation,
API behavior, and edge cases.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxboost.objective import AutoObjective, auto_objective

# =============================================================================
# Numerical Differentiation Utilities
# =============================================================================


def numerical_gradient(loss_fn, y_pred, y_true, eps=1e-5, **kwargs):
    """Compute numerical gradient via central differences."""
    grad = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_plus = y_pred.copy()
        y_minus = y_pred.copy()
        y_plus[i] += eps
        y_minus[i] -= eps

        loss_plus = float(loss_fn(y_plus[i], y_true[i], **kwargs))
        loss_minus = float(loss_fn(y_minus[i], y_true[i], **kwargs))

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def numerical_hessian(loss_fn, y_pred, y_true, eps=1e-4, **kwargs):
    """Compute numerical diagonal Hessian via central differences."""
    hess = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_plus = y_pred.copy()
        y_minus = y_pred.copy()
        y_center = y_pred.copy()
        y_plus[i] += eps
        y_minus[i] -= eps

        loss_plus = float(loss_fn(y_plus[i], y_true[i], **kwargs))
        loss_minus = float(loss_fn(y_minus[i], y_true[i], **kwargs))
        loss_center = float(loss_fn(y_center[i], y_true[i], **kwargs))

        hess[i] = (loss_plus - 2 * loss_center + loss_minus) / (eps**2)
    return hess


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = (np.random.rand(n_samples) > 0.5).astype(np.float64)
    return y_pred, y_true


# =============================================================================
# Core AutoObjective Tests
# =============================================================================


class TestAutoObjectiveBasic:
    """Basic functionality tests for AutoObjective."""

    def test_decorator_creates_auto_objective(self):
        """Test that @auto_objective decorator creates AutoObjective instance."""

        @auto_objective
        def my_loss(y_pred, y_true):
            return (y_pred - y_true) ** 2

        assert isinstance(my_loss, AutoObjective)
        assert my_loss._name == "my_loss"

    def test_class_decorator_creates_auto_objective(self):
        """Test that AutoObjective class can be used as decorator."""

        @AutoObjective
        def my_loss(y_pred, y_true):
            return (y_pred - y_true) ** 2

        assert isinstance(my_loss, AutoObjective)

    def test_gradient_shape(self, sample_data):
        """Test gradient output shape matches input."""
        y_pred, y_true = sample_data

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        grad = mse.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape
        assert grad.dtype == np.float64

    def test_hessian_shape(self, sample_data):
        """Test Hessian output shape matches input."""
        y_pred, y_true = sample_data

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        hess = mse.hessian(y_pred, y_true)
        assert hess.shape == y_pred.shape
        assert hess.dtype == np.float64

    def test_grad_hess_returns_tuple(self, sample_data):
        """Test grad_hess returns tuple of (gradient, hessian)."""
        y_pred, y_true = sample_data

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        grad, hess = mse.grad_hess(y_pred, y_true)
        assert isinstance(grad, np.ndarray)
        assert isinstance(hess, np.ndarray)
        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape


class TestAutoObjectiveGradientAccuracy:
    """Test gradient accuracy via numerical differentiation."""

    def test_mse_gradient_accuracy(self, sample_data):
        """Test MSE gradient matches numerical gradient."""
        y_pred, y_true = sample_data

        def mse_loss(yp, yt):
            return (yp - yt) ** 2

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        grad_auto = mse.gradient(y_pred, y_true)
        grad_num = numerical_gradient(mse_loss, y_pred, y_true)

        np.testing.assert_allclose(grad_auto, grad_num, rtol=1e-4, atol=1e-6)

    def test_huber_gradient_accuracy(self, sample_data):
        """Test Huber gradient matches numerical gradient."""
        y_pred, y_true = sample_data
        delta = 1.0

        def huber_loss(yp, yt, delta=1.0):
            error = yp - yt
            abs_error = np.abs(error)
            if abs_error <= delta:
                return 0.5 * error**2
            else:
                return delta * (abs_error - 0.5 * delta)

        @auto_objective
        def huber(y_pred, y_true, delta=1.0):
            error = y_pred - y_true
            abs_error = jnp.abs(error)
            return jnp.where(
                abs_error <= delta,
                0.5 * error**2,
                delta * (abs_error - 0.5 * delta),
            )

        grad_auto = huber.gradient(y_pred, y_true, delta=delta)
        grad_num = numerical_gradient(huber_loss, y_pred, y_true, delta=delta)

        np.testing.assert_allclose(grad_auto, grad_num, rtol=1e-4, atol=1e-6)

    def test_log_loss_gradient_accuracy(self, binary_data):
        """Test binary log loss gradient matches numerical gradient."""
        y_pred, y_true = binary_data

        def log_loss(yp, yt):
            p = 1 / (1 + np.exp(-yp))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -yt * np.log(p) - (1 - yt) * np.log(1 - p)

        @auto_objective
        def bce(y_pred, y_true):
            import jax

            p = jax.nn.sigmoid(y_pred)
            p = jnp.clip(p, 1e-7, 1 - 1e-7)
            return -y_true * jnp.log(p) - (1 - y_true) * jnp.log(1 - p)

        grad_auto = bce.gradient(y_pred, y_true)
        grad_num = numerical_gradient(log_loss, y_pred, y_true)

        np.testing.assert_allclose(grad_auto, grad_num, rtol=1e-3, atol=1e-5)


class TestAutoObjectiveHessianAccuracy:
    """Test Hessian accuracy via numerical differentiation."""

    def test_mse_hessian_accuracy(self, sample_data):
        """Test MSE Hessian matches numerical Hessian."""
        y_pred, y_true = sample_data

        def mse_loss(yp, yt):
            return (yp - yt) ** 2

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        hess_auto = mse.hessian(y_pred, y_true)
        hess_num = numerical_hessian(mse_loss, y_pred, y_true)

        np.testing.assert_allclose(hess_auto, hess_num, rtol=1e-3, atol=1e-4)

    def test_mse_hessian_constant(self, sample_data):
        """Test MSE Hessian is constant (2.0 for squared loss)."""
        y_pred, y_true = sample_data

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        hess = mse.hessian(y_pred, y_true)
        np.testing.assert_allclose(hess, 2.0, rtol=1e-5)

    def test_log_loss_hessian_positive(self, binary_data):
        """Test binary log loss Hessian is always positive."""
        y_pred, y_true = binary_data

        @auto_objective
        def bce(y_pred, y_true):
            import jax

            p = jax.nn.sigmoid(y_pred)
            p = jnp.clip(p, 1e-7, 1 - 1e-7)
            return -y_true * jnp.log(p) - (1 - y_true) * jnp.log(1 - p)

        hess = bce.hessian(y_pred, y_true)
        assert np.all(hess > 0), "Hessian should be positive for log loss"


class TestAutoObjectiveWithParams:
    """Test with_params functionality."""

    def test_with_params_creates_new_instance(self):
        """Test with_params creates new instance with updated defaults."""

        @auto_objective
        def parametric_loss(y_pred, y_true, alpha=0.5):
            return alpha * (y_pred - y_true) ** 2

        loss_a = parametric_loss.with_params(alpha=0.3)
        loss_b = parametric_loss.with_params(alpha=0.7)

        assert loss_a._default_kwargs["alpha"] == 0.3
        assert loss_b._default_kwargs["alpha"] == 0.7
        assert parametric_loss._default_kwargs == {}

    def test_with_params_affects_gradient(self, sample_data):
        """Test with_params affects gradient computation."""
        y_pred, y_true = sample_data

        @auto_objective
        def scaled_mse(y_pred, y_true, scale=1.0):
            return scale * (y_pred - y_true) ** 2

        loss_1x = scaled_mse.with_params(scale=1.0)
        loss_2x = scaled_mse.with_params(scale=2.0)

        grad_1x = loss_1x.gradient(y_pred, y_true)
        grad_2x = loss_2x.gradient(y_pred, y_true)

        np.testing.assert_allclose(grad_2x, 2 * grad_1x, rtol=1e-5)


class TestAutoObjectiveSampleWeights:
    """Test sample weight handling."""

    def test_sample_weights_scale_gradients(self, sample_data):
        """Test sample weights scale gradients correctly."""
        y_pred, y_true = sample_data
        weights = np.random.rand(len(y_pred))

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        grad_unweighted, hess_unweighted = mse.grad_hess(y_pred, y_true)
        grad_weighted, hess_weighted = mse.grad_hess(y_pred, y_true, sample_weight=weights)

        np.testing.assert_allclose(grad_weighted, grad_unweighted * weights, rtol=1e-5)
        np.testing.assert_allclose(hess_weighted, hess_unweighted * weights, rtol=1e-5)

    def test_empty_sample_weights_ignored(self, sample_data):
        """Test empty sample weight array is ignored."""
        y_pred, y_true = sample_data

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        grad1, hess1 = mse.grad_hess(y_pred, y_true)
        grad2, hess2 = mse.grad_hess(y_pred, y_true, sample_weight=np.array([]))

        np.testing.assert_array_equal(grad1, grad2)
        np.testing.assert_array_equal(hess1, hess2)


class TestAutoObjectiveXGBoostInterface:
    """Test XGBoost objective interface."""

    def test_xgb_objective_callable(self, sample_data):
        """Test xgb_objective returns callable."""

        @auto_objective
        def mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        obj = mse.xgb_objective
        assert callable(obj)

    def test_get_xgb_objective_with_params(self, sample_data):
        """Test get_xgb_objective accepts parameters."""

        @auto_objective
        def parametric(y_pred, y_true, alpha=0.5):
            return alpha * (y_pred - y_true) ** 2

        obj = parametric.get_xgb_objective(alpha=0.3)
        assert callable(obj)
        assert "parametric" in obj.__name__


class TestAutoObjectiveRepr:
    """Test string representation."""

    def test_repr_without_params(self):
        """Test repr without default params."""

        @auto_objective
        def my_loss(y_pred, y_true):
            return (y_pred - y_true) ** 2

        assert repr(my_loss) == "AutoObjective(my_loss)"

    def test_repr_with_params(self):
        """Test repr with default params."""

        @auto_objective
        def my_loss(y_pred, y_true, alpha=0.5):
            return alpha * (y_pred - y_true) ** 2

        loss = my_loss.with_params(alpha=0.3)
        assert "alpha" in repr(loss)
        assert "0.3" in repr(loss)
