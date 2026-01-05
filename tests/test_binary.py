"""
Tests for binary classification objective functions.

Tests gradient/Hessian correctness for focal loss, BCE, hinge, etc.
"""

import numpy as np
import pytest

from jaxboost.objective import (
    binary_crossentropy,
    focal_loss,
    hinge_loss,
    weighted_binary_crossentropy,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = (np.random.rand(n_samples) > 0.5).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def balanced_binary_data():
    """Generate perfectly balanced binary data."""
    np.random.seed(42)
    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.array([0.0, 1.0] * (n_samples // 2))
    return y_pred, y_true


# =============================================================================
# Numerical Gradient Utility
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


# =============================================================================
# Binary Cross-Entropy Tests
# =============================================================================


class TestBinaryCrossEntropy:
    """Tests for binary cross-entropy loss."""

    def test_gradient_shape(self, binary_data):
        """Test BCE gradient has correct shape."""
        y_pred, y_true = binary_data
        grad = binary_crossentropy.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape
        assert grad.dtype == np.float64

    def test_gradient_formula(self, binary_data):
        """Test BCE gradient matches p - y formula."""
        y_pred, y_true = binary_data
        grad = binary_crossentropy.gradient(y_pred, y_true)

        # BCE gradient should be sigmoid(y_pred) - y_true
        p = 1 / (1 + np.exp(-y_pred))
        expected = p - y_true
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_hessian_positive(self, binary_data):
        """Test BCE Hessian is always positive."""
        y_pred, y_true = binary_data
        hess = binary_crossentropy.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_hessian_formula(self, binary_data):
        """Test BCE Hessian matches p*(1-p) formula."""
        y_pred, y_true = binary_data
        hess = binary_crossentropy.hessian(y_pred, y_true)

        # BCE Hessian should be p * (1 - p)
        p = 1 / (1 + np.exp(-y_pred))
        expected = p * (1 - p)
        np.testing.assert_allclose(hess, expected, rtol=1e-3)

    def test_gradient_zero_at_correct_prediction(self):
        """Test gradient approaches zero for correct confident predictions."""
        # Very confident correct prediction
        y_pred = np.array([10.0])
        y_true = np.array([1.0])

        grad = binary_crossentropy.gradient(y_pred, y_true)
        assert np.abs(grad[0]) < 1e-4


# =============================================================================
# Focal Loss Tests
# =============================================================================


class TestFocalLoss:
    """Tests for focal loss."""

    def test_gradient_shape(self, binary_data):
        """Test focal loss gradient has correct shape."""
        y_pred, y_true = binary_data
        grad = focal_loss.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, binary_data):
        """Test focal loss Hessian is positive."""
        y_pred, y_true = binary_data
        hess = focal_loss.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_gamma_zero_equals_bce(self, binary_data):
        """Test focal loss with gamma=0 approaches BCE (up to alpha scaling)."""
        y_pred, y_true = binary_data

        focal_g0 = focal_loss.with_params(gamma=0.0, alpha=0.5)
        grad_focal = focal_g0.gradient(y_pred, y_true)

        # With gamma=0 and alpha=0.5, should be proportional to BCE gradient
        grad_bce = binary_crossentropy.gradient(y_pred, y_true)

        # The relationship involves alpha weighting
        # Just check they have same sign and similar relative magnitudes
        assert np.corrcoef(grad_focal, grad_bce)[0, 1] > 0.99

    def test_higher_gamma_focuses_on_hard_examples(self, binary_data):
        """Test higher gamma reduces gradient for easy examples."""
        y_pred, y_true = binary_data

        focal_low = focal_loss.with_params(gamma=1.0)
        focal_high = focal_loss.with_params(gamma=5.0)

        grad_low = focal_low.gradient(y_pred, y_true)
        grad_high = focal_high.gradient(y_pred, y_true)

        # Higher gamma should have smaller gradients on average
        # (more focus on hard examples means less update for easy ones)
        assert np.mean(np.abs(grad_high)) < np.mean(np.abs(grad_low))

    def test_with_params_gamma(self, binary_data):
        """Test with_params for gamma parameter."""
        y_pred, y_true = binary_data

        focal_2 = focal_loss.with_params(gamma=2.0)
        focal_3 = focal_loss.with_params(gamma=3.0)

        grad_2 = focal_2.gradient(y_pred, y_true)
        grad_3 = focal_3.gradient(y_pred, y_true)

        # Different gammas should give different gradients
        assert not np.allclose(grad_2, grad_3)

    def test_with_params_alpha(self, binary_data):
        """Test with_params for alpha parameter."""
        y_pred, y_true = binary_data

        focal_a25 = focal_loss.with_params(alpha=0.25)
        focal_a75 = focal_loss.with_params(alpha=0.75)

        grad_a25 = focal_a25.gradient(y_pred, y_true)
        grad_a75 = focal_a75.gradient(y_pred, y_true)

        # Different alphas should give different gradients
        assert not np.allclose(grad_a25, grad_a75)


# =============================================================================
# Weighted BCE Tests
# =============================================================================


class TestWeightedBCE:
    """Tests for weighted binary cross-entropy."""

    def test_weight_1_equals_bce(self, binary_data):
        """Test pos_weight=1 equals standard BCE."""
        y_pred, y_true = binary_data

        grad_weighted = weighted_binary_crossentropy.gradient(y_pred, y_true, pos_weight=1.0)
        grad_bce = binary_crossentropy.gradient(y_pred, y_true)

        np.testing.assert_allclose(grad_weighted, grad_bce, rtol=1e-5)

    def test_higher_weight_increases_positive_gradient(self, binary_data):
        """Test higher pos_weight increases gradient for positive examples."""
        y_pred, y_true = binary_data

        grad_w1 = weighted_binary_crossentropy.gradient(y_pred, y_true, pos_weight=1.0)
        grad_w10 = weighted_binary_crossentropy.gradient(y_pred, y_true, pos_weight=10.0)

        # For positive examples (y_true=1), gradient magnitude should increase
        pos_mask = y_true == 1
        assert np.mean(np.abs(grad_w10[pos_mask])) > np.mean(np.abs(grad_w1[pos_mask]))


# =============================================================================
# Hinge Loss Tests
# =============================================================================


class TestHingeLoss:
    """Tests for smooth hinge loss."""

    def test_gradient_shape(self, binary_data):
        """Test hinge loss gradient has correct shape."""
        y_pred, y_true = binary_data
        grad = hinge_loss.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_hessian_positive(self, binary_data):
        """Test hinge loss Hessian is non-negative."""
        y_pred, y_true = binary_data
        hess = hinge_loss.hessian(y_pred, y_true)
        assert np.all(hess >= 0)

    def test_correct_prediction_small_gradient(self):
        """Test correct predictions with large margin have small gradient."""
        # Correct prediction with large margin
        y_pred = np.array([10.0])
        y_true = np.array([1.0])  # Converted to +1 internally

        grad = hinge_loss.gradient(y_pred, y_true)
        assert np.abs(grad[0]) < 0.01

    def test_wrong_prediction_large_gradient(self):
        """Test wrong predictions have larger gradient."""
        # Wrong prediction
        y_pred = np.array([-5.0])
        y_true = np.array([1.0])

        grad = hinge_loss.gradient(y_pred, y_true)
        assert np.abs(grad[0]) > 0.5
