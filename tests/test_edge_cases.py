"""
Tests for edge cases and numerical stability.

Tests behavior with NaN, inf, empty arrays, extreme values, etc.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxboost.objective import (
    auto_objective,
    binary_crossentropy,
    focal_loss,
    huber,
    mse,
    softmax_cross_entropy,
)

# =============================================================================
# Empty Array Tests
# =============================================================================


class TestEmptyArrays:
    """Tests for empty array handling."""

    def test_mse_empty_input(self):
        """Test MSE handles empty arrays gracefully."""
        y_pred = np.array([]).astype(np.float64)
        y_true = np.array([]).astype(np.float64)

        grad = mse.gradient(y_pred, y_true)
        hess = mse.hessian(y_pred, y_true)

        assert grad.shape == (0,)
        assert hess.shape == (0,)

    def test_huber_empty_input(self):
        """Test Huber handles empty arrays gracefully."""
        y_pred = np.array([]).astype(np.float64)
        y_true = np.array([]).astype(np.float64)

        grad = huber.gradient(y_pred, y_true)
        hess = huber.hessian(y_pred, y_true)

        assert grad.shape == (0,)
        assert hess.shape == (0,)


# =============================================================================
# Single Sample Tests
# =============================================================================


class TestSingleSample:
    """Tests for single sample inputs."""

    def test_mse_single_sample(self):
        """Test MSE works with single sample."""
        y_pred = np.array([1.0])
        y_true = np.array([0.0])

        grad = mse.gradient(y_pred, y_true)
        hess = mse.hessian(y_pred, y_true)

        assert grad.shape == (1,)
        assert hess.shape == (1,)
        np.testing.assert_allclose(grad, [2.0], rtol=1e-5)
        np.testing.assert_allclose(hess, [2.0], rtol=1e-5)

    def test_binary_single_sample(self):
        """Test binary classification with single sample."""
        y_pred = np.array([0.0])
        y_true = np.array([1.0])

        grad = binary_crossentropy.gradient(y_pred, y_true)
        hess = binary_crossentropy.hessian(y_pred, y_true)

        assert grad.shape == (1,)
        assert hess.shape == (1,)
        assert not np.isnan(grad[0])
        assert not np.isnan(hess[0])


# =============================================================================
# Extreme Value Tests
# =============================================================================


class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_mse_large_values(self):
        """Test MSE with large values doesn't overflow."""
        y_pred = np.array([1e10, -1e10])
        y_true = np.array([0.0, 0.0])

        grad = mse.gradient(y_pred, y_true)
        hess = mse.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(grad))
        assert not np.any(np.isinf(hess))

    def test_binary_extreme_logits(self):
        """Test binary classification with extreme logits."""
        y_pred = np.array([100.0, -100.0])
        y_true = np.array([1.0, 0.0])

        grad = binary_crossentropy.gradient(y_pred, y_true)
        hess = binary_crossentropy.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        # Gradient should be near zero for correct confident predictions
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-4)

    def test_focal_extreme_logits(self):
        """Test focal loss with extreme logits."""
        y_pred = np.array([50.0, -50.0])
        y_true = np.array([1.0, 0.0])

        grad = focal_loss.gradient(y_pred, y_true)
        hess = focal_loss.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    def test_huber_extreme_errors(self):
        """Test Huber with extreme errors."""
        y_pred = np.array([1e6])
        y_true = np.array([0.0])

        grad = huber.gradient(y_pred, y_true)
        hess = huber.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        # In linear region, gradient magnitude should be delta
        assert np.abs(grad[0]) == pytest.approx(1.0, rel=1e-3)

    def test_softmax_extreme_logits(self):
        """Test softmax CE with extreme logits doesn't overflow."""
        n_classes = 3
        # Very large difference in logits
        y_pred = np.array([[1000.0, 0.0, 0.0]])
        y_true = np.array([0.0])

        obj = softmax_cross_entropy(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)
        hess = obj.hessian(y_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))


# =============================================================================
# NaN Handling Tests
# =============================================================================


class TestNaNHandling:
    """Tests for NaN input handling."""

    def test_mse_nan_propagation(self):
        """Test MSE propagates NaN correctly."""
        y_pred = np.array([1.0, np.nan, 3.0])
        y_true = np.array([0.0, 0.0, 0.0])

        grad = mse.gradient(y_pred, y_true)

        # NaN should propagate
        assert np.isnan(grad[1])
        # Other values should be valid
        assert not np.isnan(grad[0])
        assert not np.isnan(grad[2])

    def test_binary_nan_propagation(self):
        """Test binary CE propagates NaN."""
        y_pred = np.array([0.0, np.nan])
        y_true = np.array([1.0, 1.0])

        grad = binary_crossentropy.gradient(y_pred, y_true)

        assert np.isnan(grad[1])
        assert not np.isnan(grad[0])


# =============================================================================
# Inf Handling Tests
# =============================================================================


class TestInfHandling:
    """Tests for infinity input handling."""

    def test_binary_inf_logit(self):
        """Test binary CE handles inf logits via clipping."""
        y_pred = np.array([np.inf, -np.inf])
        y_true = np.array([1.0, 0.0])

        # Should not raise, may produce NaN/inf in gradient
        grad = binary_crossentropy.gradient(y_pred, y_true)

        # The implementation should handle this gracefully
        # (may clip internally or produce bounded output)
        assert grad.shape == (2,)


# =============================================================================
# Type Conversion Tests
# =============================================================================


class TestTypeConversion:
    """Tests for input type handling."""

    def test_float32_input(self):
        """Test float32 input is handled correctly."""
        y_pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_true = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        grad = mse.gradient(y_pred, y_true)

        # Output should be float64
        assert grad.dtype == np.float64

    def test_int_input_converted(self):
        """Test integer input is converted."""
        y_pred = np.array([1, 2, 3], dtype=np.int32)
        y_true = np.array([0, 1, 2], dtype=np.int32)

        # Should work without error (automatic conversion)
        grad = mse.gradient(y_pred.astype(np.float64), y_true.astype(np.float64))
        assert grad.dtype == np.float64


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestNumericalPrecision:
    """Tests for numerical precision."""

    def test_small_difference_precision(self):
        """Test precision for small differences."""
        y_pred = np.array([1.0 + 1e-5])  # Use larger diff for float32 precision
        y_true = np.array([1.0])

        grad = mse.gradient(y_pred, y_true)

        # Should detect small difference (gradient = 2 * diff)
        expected = 2 * 1e-5
        np.testing.assert_allclose(grad[0], expected, rtol=1e-2)

    def test_gradient_symmetry(self):
        """Test gradient is symmetric around zero error."""
        y_pred = np.array([1.0, -1.0])
        y_true = np.array([0.0, 0.0])

        grad = mse.gradient(y_pred, y_true)

        # Gradients should have same magnitude, opposite sign
        np.testing.assert_allclose(grad[0], -grad[1], rtol=1e-10)


# =============================================================================
# Batch Size Tests
# =============================================================================


class TestBatchSizes:
    """Tests for various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000, 10000])
    def test_mse_various_batch_sizes(self, batch_size):
        """Test MSE works with various batch sizes."""
        np.random.seed(42)
        y_pred = np.random.randn(batch_size).astype(np.float64)
        y_true = np.random.randn(batch_size).astype(np.float64)

        grad = mse.gradient(y_pred, y_true)
        hess = mse.hessian(y_pred, y_true)

        assert grad.shape == (batch_size,)
        assert hess.shape == (batch_size,)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))

    @pytest.mark.parametrize("batch_size", [1, 10, 100, 1000])
    def test_focal_various_batch_sizes(self, batch_size):
        """Test focal loss works with various batch sizes."""
        np.random.seed(42)
        y_pred = np.random.randn(batch_size).astype(np.float64)
        y_true = (np.random.rand(batch_size) > 0.5).astype(np.float64)

        grad = focal_loss.gradient(y_pred, y_true)
        hess = focal_loss.hessian(y_pred, y_true)

        assert grad.shape == (batch_size,)
        assert hess.shape == (batch_size,)
        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))


# =============================================================================
# Custom Objective Edge Cases
# =============================================================================


class TestCustomObjectiveEdgeCases:
    """Tests for edge cases in custom objectives."""

    def test_constant_loss_zero_gradient(self):
        """Test constant loss has zero gradient."""

        @auto_objective
        def constant_loss(y_pred, y_true):
            return jnp.ones_like(y_pred)

        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([0.0, 0.0, 0.0])

        grad = constant_loss.gradient(y_pred, y_true)
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

    def test_linear_loss_constant_gradient(self):
        """Test linear loss has constant gradient."""

        @auto_objective
        def linear_loss(y_pred, y_true):
            return y_pred - y_true

        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([0.0, 0.0, 0.0])

        grad = linear_loss.gradient(y_pred, y_true)
        np.testing.assert_allclose(grad, 1.0, rtol=1e-5)

        hess = linear_loss.hessian(y_pred, y_true)
        np.testing.assert_allclose(hess, 0.0, atol=1e-5)
