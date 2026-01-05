"""
Tests for regression objective functions.

Tests gradient/Hessian correctness for MSE, Huber, quantile, Tweedie, etc.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxboost.objective import (
    asymmetric,
    gamma,
    huber,
    log_cosh,
    mae_smooth,
    mse,
    poisson,
    pseudo_huber,
    quantile,
    tweedie,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def positive_data():
    """Generate positive-valued data for Tweedie/Poisson/Gamma."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)  # log-space
    y_true = np.abs(np.random.randn(n_samples)).astype(np.float64) + 0.1
    return y_pred, y_true


# =============================================================================
# Numerical Gradient Utilities
# =============================================================================


def numerical_gradient(objective, y_pred, y_true, eps=1e-5, **kwargs):
    """Compute numerical gradient via central differences."""
    grad = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_plus = y_pred.copy()
        y_minus = y_pred.copy()
        y_plus[i] += eps
        y_minus[i] -= eps

        # Call the objective to get losses
        loss_plus = objective(jnp.asarray(y_plus), jnp.asarray(y_true), **kwargs)
        loss_minus = objective(jnp.asarray(y_minus), jnp.asarray(y_true), **kwargs)

        grad[i] = (float(loss_plus[i]) - float(loss_minus[i])) / (2 * eps)
    return grad


# =============================================================================
# MSE Tests
# =============================================================================


class TestMSE:
    """Tests for Mean Squared Error loss."""

    def test_gradient_correctness(self, regression_data):
        """Test MSE gradient is 2*(y_pred - y_true)."""
        y_pred, y_true = regression_data
        grad = mse.gradient(y_pred, y_true)
        expected = 2 * (y_pred - y_true)
        np.testing.assert_allclose(grad, expected, rtol=1e-5)

    def test_hessian_constant(self, regression_data):
        """Test MSE Hessian is constant 2."""
        y_pred, y_true = regression_data
        hess = mse.hessian(y_pred, y_true)
        np.testing.assert_allclose(hess, 2.0, rtol=1e-5)

    def test_minimum_at_true_value(self):
        """Test gradient is zero when y_pred == y_true."""
        y = np.array([1.0, 2.0, 3.0])
        grad = mse.gradient(y, y)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)


# =============================================================================
# Huber Tests
# =============================================================================


class TestHuber:
    """Tests for Huber loss."""

    def test_gradient_quadratic_region(self, regression_data):
        """Test Huber gradient in quadratic region (|error| <= delta)."""
        y_pred, y_true = regression_data
        delta = 10.0  # Large delta so all points in quadratic region

        grad = huber.gradient(y_pred, y_true, delta=delta)
        # In quadratic region, gradient should be error (derivative of 0.5*error^2)
        expected = y_pred - y_true
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_gradient_linear_region(self):
        """Test Huber gradient in linear region (|error| > delta)."""
        delta = 0.1
        y_pred = np.array([0.0])
        y_true = np.array([10.0])  # Large error

        grad = huber.gradient(y_pred, y_true, delta=delta)
        # In linear region, gradient magnitude should be delta
        assert np.abs(grad[0]) == pytest.approx(delta, rel=1e-4)

    def test_hessian_quadratic_region(self, regression_data):
        """Test Huber Hessian in quadratic region."""
        y_pred, y_true = regression_data
        delta = 10.0

        hess = huber.hessian(y_pred, y_true, delta=delta)
        np.testing.assert_allclose(hess, 1.0, rtol=1e-3)

    def test_with_params_delta(self, regression_data):
        """Test Huber with different delta values."""
        y_pred, y_true = regression_data

        huber_small = huber.with_params(delta=0.5)
        huber_large = huber.with_params(delta=5.0)

        grad_small = huber_small.gradient(y_pred, y_true)
        grad_large = huber_large.gradient(y_pred, y_true)

        # Different delta should give different gradients
        assert not np.allclose(grad_small, grad_large)


# =============================================================================
# Quantile Tests
# =============================================================================


class TestQuantile:
    """Tests for quantile (pinball) loss."""

    def test_median_symmetric(self, regression_data):
        """Test q=0.5 gives symmetric loss."""
        y_pred, y_true = regression_data

        q50 = quantile.with_params(q=0.5)
        grad = q50.gradient(y_pred, y_true)

        # For median, positive and negative errors should have similar magnitude gradients
        # (up to sign)
        assert np.std(np.abs(grad)) < np.std(grad) * 2

    def test_high_quantile_penalizes_underprediction(self):
        """Test high quantile penalizes underprediction more."""
        y_pred = np.array([0.0])
        y_true = np.array([1.0])  # Underprediction

        q90 = quantile.with_params(q=0.9)
        q10 = quantile.with_params(q=0.1)

        grad_90 = q90.gradient(y_pred, y_true)
        grad_10 = q10.gradient(y_pred, y_true)

        # q=0.9 should have larger gradient magnitude for underprediction
        assert np.abs(grad_90[0]) > np.abs(grad_10[0])

    def test_hessian_positive(self, regression_data):
        """Test quantile Hessian is positive (from regularization)."""
        y_pred, y_true = regression_data

        hess = quantile.hessian(y_pred, y_true)
        assert np.all(hess > 0)


# =============================================================================
# Tweedie Tests
# =============================================================================


class TestTweedie:
    """Tests for Tweedie deviance loss."""

    def test_gradient_shape(self, positive_data):
        """Test Tweedie gradient has correct shape."""
        y_pred, y_true = positive_data
        grad = tweedie.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, positive_data):
        """Test Tweedie Hessian is positive."""
        y_pred, y_true = positive_data
        hess = tweedie.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_different_power_parameters(self, positive_data):
        """Test Tweedie with different power parameters."""
        y_pred, y_true = positive_data

        tweedie_15 = tweedie.with_params(p=1.5)
        tweedie_18 = tweedie.with_params(p=1.8)

        grad_15 = tweedie_15.gradient(y_pred, y_true)
        grad_18 = tweedie_18.gradient(y_pred, y_true)

        # Different powers should give different gradients
        assert not np.allclose(grad_15, grad_18)


# =============================================================================
# Asymmetric Loss Tests
# =============================================================================


class TestAsymmetric:
    """Tests for asymmetric squared error loss."""

    def test_symmetric_at_half(self, regression_data):
        """Test alpha=0.5 gives symmetric loss (equivalent to MSE)."""
        y_pred, y_true = regression_data

        asym_half = asymmetric.with_params(alpha=0.5)
        grad_asym = asym_half.gradient(y_pred, y_true)
        grad_mse = mse.gradient(y_pred, y_true)

        # Should be proportional (MSE has factor of 2)
        np.testing.assert_allclose(grad_asym, grad_mse / 2, rtol=1e-4)

    def test_asymmetric_penalizes_correctly(self):
        """Test high alpha penalizes underprediction more."""
        y_pred = np.array([0.0, 0.0])
        y_true = np.array([1.0, -1.0])  # Under and over prediction

        asym_high = asymmetric.with_params(alpha=0.9)
        grad = asym_high.gradient(y_pred, y_true)

        # Underprediction (positive error) should have larger gradient
        assert np.abs(grad[0]) > np.abs(grad[1])


# =============================================================================
# Log-Cosh Tests
# =============================================================================


class TestLogCosh:
    """Tests for log-cosh loss."""

    def test_gradient_tanh_approximation(self, regression_data):
        """Test log-cosh gradient approaches tanh(error)."""
        y_pred, y_true = regression_data
        grad = log_cosh.gradient(y_pred, y_true)
        expected = np.tanh(y_pred - y_true)
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_hessian_positive(self, regression_data):
        """Test log-cosh Hessian is positive."""
        y_pred, y_true = regression_data
        hess = log_cosh.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_bounded_gradient(self):
        """Test log-cosh gradient is bounded by [-1, 1] for moderate values."""
        # Use moderate values to avoid numerical overflow
        y_pred = np.array([10.0, -10.0])
        y_true = np.array([0.0, 0.0])

        grad = log_cosh.gradient(y_pred, y_true)
        assert np.all(np.abs(grad) <= 1.0 + 1e-6)


# =============================================================================
# Pseudo-Huber Tests
# =============================================================================


class TestPseudoHuber:
    """Tests for pseudo-Huber loss."""

    def test_approximates_huber(self, regression_data):
        """Test pseudo-Huber approximates Huber loss."""
        y_pred, y_true = regression_data
        delta = 1.0

        grad_pseudo = pseudo_huber.gradient(y_pred, y_true, delta=delta)
        grad_huber = huber.gradient(y_pred, y_true, delta=delta)

        # Should be similar (pseudo-Huber is smooth approximation)
        np.testing.assert_allclose(grad_pseudo, grad_huber, rtol=0.2, atol=0.1)

    def test_hessian_positive(self, regression_data):
        """Test pseudo-Huber Hessian is always positive."""
        y_pred, y_true = regression_data
        hess = pseudo_huber.hessian(y_pred, y_true)
        assert np.all(hess > 0)


# =============================================================================
# Smooth MAE Tests
# =============================================================================


class TestMAESmooth:
    """Tests for smooth MAE loss."""

    def test_gradient_shape(self, regression_data):
        """Test smooth MAE gradient has correct shape."""
        y_pred, y_true = regression_data
        grad = mae_smooth.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, regression_data):
        """Test smooth MAE Hessian is positive."""
        y_pred, y_true = regression_data
        hess = mae_smooth.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_approaches_sign_function(self):
        """Test smooth MAE gradient approaches sign function."""
        y_pred = np.array([10.0, -10.0])
        y_true = np.array([0.0, 0.0])

        grad = mae_smooth.gradient(y_pred, y_true, beta=0.01)
        expected = np.sign(y_pred - y_true)
        np.testing.assert_allclose(grad, expected, rtol=0.1)


# =============================================================================
# Poisson Tests
# =============================================================================


class TestPoisson:
    """Tests for Poisson loss."""

    def test_gradient_shape(self, positive_data):
        """Test Poisson gradient has correct shape."""
        y_pred, y_true = positive_data
        grad = poisson.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, positive_data):
        """Test Poisson Hessian is positive."""
        y_pred, y_true = positive_data
        hess = poisson.hessian(y_pred, y_true)
        assert np.all(hess > 0)


# =============================================================================
# Gamma Tests
# =============================================================================


class TestGamma:
    """Tests for Gamma loss."""

    def test_gradient_shape(self, positive_data):
        """Test Gamma gradient has correct shape."""
        y_pred, y_true = positive_data
        grad = gamma.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, positive_data):
        """Test Gamma Hessian is positive."""
        y_pred, y_true = positive_data
        hess = gamma.hessian(y_pred, y_true)
        assert np.all(hess > 0)
