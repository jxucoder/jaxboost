"""
Tests for survival analysis objective functions.

Tests gradient/Hessian correctness for AFT, Weibull AFT objectives.
"""

import numpy as np
import pytest

from jaxboost.objective import (
    aft,
    weibull_aft,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def survival_data_uncensored():
    """Generate uncensored survival data (all events observed)."""
    np.random.seed(42)
    n_samples = 50
    y_pred = np.random.randn(n_samples).astype(np.float64)
    # Positive survival times
    y_true = np.abs(np.random.randn(n_samples)).astype(np.float64) + 0.1
    return y_pred, y_true


@pytest.fixture
def survival_data_right_censored():
    """Generate right-censored survival data."""
    np.random.seed(42)
    n_samples = 50
    y_pred = np.random.randn(n_samples).astype(np.float64)
    # Lower bounds (observed times)
    lower = np.abs(np.random.randn(n_samples)).astype(np.float64) + 0.1
    # Upper bounds (inf for right-censored)
    upper = np.where(np.random.rand(n_samples) > 0.5, np.inf, lower)
    y_true = lower  # Use lower as label
    return y_pred, y_true, lower, upper


@pytest.fixture
def survival_data_interval():
    """Generate interval-censored survival data."""
    np.random.seed(42)
    n_samples = 50
    y_pred = np.random.randn(n_samples).astype(np.float64)
    # Lower bounds
    lower = np.abs(np.random.randn(n_samples)).astype(np.float64) + 0.1
    # Upper bounds (finite, greater than lower)
    upper = lower + np.abs(np.random.randn(n_samples)) + 0.1
    y_true = lower
    return y_pred, y_true, lower, upper


# =============================================================================
# AFT Tests
# =============================================================================


class TestAFT:
    """Tests for Accelerated Failure Time loss."""

    def test_gradient_shape(self, survival_data_uncensored):
        """Test AFT gradient has correct shape."""
        y_pred, y_true = survival_data_uncensored

        grad = aft.gradient(y_pred, y_true)

        assert grad.shape == y_pred.shape
        assert grad.dtype == np.float64

    def test_hessian_shape(self, survival_data_uncensored):
        """Test AFT Hessian has correct shape."""
        y_pred, y_true = survival_data_uncensored

        hess = aft.hessian(y_pred, y_true)

        assert hess.shape == y_pred.shape
        assert hess.dtype == np.float64

    def test_uncensored_no_nan(self, survival_data_uncensored):
        """Test uncensored data produces no NaN."""
        y_pred, y_true = survival_data_uncensored

        grad = aft.gradient(y_pred, y_true)
        hess = aft.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_right_censored_with_bounds(self, survival_data_right_censored):
        """Test right-censored data with label bounds."""
        y_pred, y_true, lower, upper = survival_data_right_censored

        grad = aft.gradient(y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper)
        hess = aft.hessian(y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper)

        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_interval_censored_with_bounds(self, survival_data_interval):
        """Test interval-censored data with label bounds."""
        y_pred, y_true, lower, upper = survival_data_interval

        grad = aft.gradient(y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper)
        hess = aft.hessian(y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper)

        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_sigma_parameter(self, survival_data_uncensored):
        """Test sigma parameter affects gradients."""
        y_pred, y_true = survival_data_uncensored

        grad_s1 = aft.gradient(y_pred, y_true, sigma=1.0)
        grad_s2 = aft.gradient(y_pred, y_true, sigma=2.0)

        # Different sigma should give different gradients
        assert not np.allclose(grad_s1, grad_s2)


# =============================================================================
# Weibull AFT Tests
# =============================================================================


class TestWeibullAFT:
    """Tests for Weibull AFT loss."""

    def test_gradient_shape(self, survival_data_uncensored):
        """Test Weibull AFT gradient has correct shape."""
        y_pred, y_true = survival_data_uncensored

        grad = weibull_aft.gradient(y_pred, y_true)

        assert grad.shape == y_pred.shape
        assert grad.dtype == np.float64

    def test_hessian_shape(self, survival_data_uncensored):
        """Test Weibull AFT Hessian has correct shape."""
        y_pred, y_true = survival_data_uncensored

        hess = weibull_aft.hessian(y_pred, y_true)

        assert hess.shape == y_pred.shape
        assert hess.dtype == np.float64

    def test_uncensored_no_nan(self, survival_data_uncensored):
        """Test uncensored data produces no NaN."""
        y_pred, y_true = survival_data_uncensored

        grad = weibull_aft.gradient(y_pred, y_true)
        hess = weibull_aft.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_right_censored_with_bounds(self, survival_data_right_censored):
        """Test right-censored data with label bounds."""
        y_pred, y_true, lower, upper = survival_data_right_censored

        grad = weibull_aft.gradient(
            y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper
        )
        hess = weibull_aft.hessian(y_pred, y_true, label_lower_bound=lower, label_upper_bound=upper)

        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_k_parameter(self, survival_data_uncensored):
        """Test k (shape) parameter affects gradients."""
        y_pred, y_true = survival_data_uncensored

        grad_k1 = weibull_aft.gradient(y_pred, y_true, k=1.0)
        grad_k2 = weibull_aft.gradient(y_pred, y_true, k=2.0)

        # Different k should give different gradients
        assert not np.allclose(grad_k1, grad_k2)

    def test_exponential_special_case(self, survival_data_uncensored):
        """Test k=1 gives exponential distribution (Weibull special case)."""
        y_pred, y_true = survival_data_uncensored

        # k=1 should work without issues
        grad = weibull_aft.gradient(y_pred, y_true, k=1.0)
        hess = weibull_aft.hessian(y_pred, y_true, k=1.0)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestSurvivalNumericalStability:
    """Tests for numerical stability of survival objectives."""

    def test_aft_very_small_times(self):
        """Test AFT handles very small survival times."""
        y_pred = np.array([0.0])
        y_true = np.array([1e-10])

        grad = aft.gradient(y_pred, y_true)
        hess = aft.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(grad))
        assert not np.any(np.isinf(hess))

    def test_aft_very_large_times(self):
        """Test AFT handles very large survival times."""
        y_pred = np.array([0.0])
        y_true = np.array([1e10])

        grad = aft.gradient(y_pred, y_true)
        hess = aft.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_weibull_very_small_times(self):
        """Test Weibull AFT handles very small survival times."""
        y_pred = np.array([0.0])
        y_true = np.array([1e-10])

        grad = weibull_aft.gradient(y_pred, y_true)
        hess = weibull_aft.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_extreme_predictions(self):
        """Test survival objectives with extreme predictions."""
        y_pred = np.array([50.0, -50.0])
        y_true = np.array([1.0, 1.0])

        grad_aft = aft.gradient(y_pred, y_true)
        grad_weibull = weibull_aft.gradient(y_pred, y_true)

        assert not np.any(np.isnan(grad_aft))
        assert not np.any(np.isnan(grad_weibull))
