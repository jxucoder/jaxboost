"""
Tests for multi-class classification objective functions.

Tests gradient/Hessian correctness for softmax CE, focal multiclass, etc.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxboost.objective import (
    MultiClassObjective,
    class_balanced,
    focal_multiclass,
    label_smoothing,
    multiclass_objective,
    softmax_cross_entropy,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def multiclass_data():
    """Generate sample multi-class classification data."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 5
    # Logits shape: (n_samples, n_classes)
    y_pred = np.random.randn(n_samples, n_classes).astype(np.float64)
    # Labels: integer class indices
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_classes


@pytest.fixture
def multiclass_data_3():
    """Generate sample 3-class classification data."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 3
    y_pred = np.random.randn(n_samples, n_classes).astype(np.float64)
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_classes


# =============================================================================
# MultiClassObjective Tests
# =============================================================================


class TestMultiClassObjective:
    """Tests for MultiClassObjective base class."""

    def test_decorator_creates_multiclass_objective(self):
        """Test @multiclass_objective decorator creates instance."""

        @multiclass_objective(n_classes=3)
        def my_loss(logits, label):
            import jax

            probs = jax.nn.softmax(logits)
            return -jnp.log(probs[label] + 1e-10)

        assert isinstance(my_loss, MultiClassObjective)
        assert my_loss.n_classes == 3

    def test_gradient_shape(self, multiclass_data_3):
        """Test gradient output shape is (n_samples, n_classes)."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (len(y_true), n_classes)
        assert grad.dtype == np.float64

    def test_hessian_shape(self, multiclass_data_3):
        """Test Hessian output shape is (n_samples, n_classes)."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)
        hess = obj.hessian(y_pred, y_true)

        assert hess.shape == (len(y_true), n_classes)
        assert hess.dtype == np.float64

    def test_handles_flattened_input(self, multiclass_data_3):
        """Test handles flattened y_pred input."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)

        # Flatten y_pred
        y_pred_flat = y_pred.flatten()
        grad = obj.gradient(y_pred_flat, y_true)

        assert grad.shape == (len(y_true), n_classes)


# =============================================================================
# Softmax Cross-Entropy Tests
# =============================================================================


class TestSoftmaxCrossEntropy:
    """Tests for softmax cross-entropy loss."""

    def test_gradient_sum_zero(self, multiclass_data_3):
        """Test gradient sums to approximately zero per sample."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        # For softmax CE, gradients should sum to 0 per sample
        # grad_i = p_i - 1{i=y} where sum(p_i) = 1
        row_sums = np.sum(grad, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-5)

    def test_gradient_formula(self, multiclass_data_3):
        """Test gradient matches p - one_hot(y) formula."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        # Expected: p - one_hot(y)
        exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1

        expected = probs - one_hot
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_hessian_positive(self, multiclass_data_3):
        """Test Hessian diagonal is positive."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = softmax_cross_entropy(n_classes=n_classes)
        hess = obj.hessian(y_pred, y_true)

        assert np.all(hess > 0)

    def test_correct_prediction_small_loss(self):
        """Test correct confident prediction has small gradient."""
        n_classes = 3
        # Very confident correct prediction
        y_pred = np.array([[10.0, 0.0, 0.0]])  # Confident class 0
        y_true = np.array([0.0])

        obj = softmax_cross_entropy(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        # Gradient for true class should be near zero
        assert np.abs(grad[0, 0]) < 0.01


# =============================================================================
# Focal Multiclass Tests
# =============================================================================


class TestFocalMulticlass:
    """Tests for focal loss multi-class."""

    def test_gradient_shape(self, multiclass_data_3):
        """Test focal multiclass gradient has correct shape."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = focal_multiclass(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (len(y_true), n_classes)

    def test_hessian_shape(self, multiclass_data_3):
        """Test focal multiclass Hessian has correct shape."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = focal_multiclass(n_classes=n_classes)
        hess = obj.hessian(y_pred, y_true)

        # Focal loss Hessian can be negative in some regions due to
        # the focal weight derivative - check shape only
        assert hess.shape == (len(y_true), n_classes)

    def test_gamma_effect(self, multiclass_data_3):
        """Test higher gamma reduces gradient for easy examples."""
        y_pred, y_true, n_classes = multiclass_data_3

        focal_low = focal_multiclass(n_classes=n_classes, gamma=1.0)
        focal_high = focal_multiclass(n_classes=n_classes, gamma=5.0)

        grad_low = focal_low.gradient(y_pred, y_true)
        grad_high = focal_high.gradient(y_pred, y_true)

        # Higher gamma should reduce average gradient magnitude
        assert np.mean(np.abs(grad_high)) < np.mean(np.abs(grad_low))


# =============================================================================
# Label Smoothing Tests
# =============================================================================


class TestLabelSmoothing:
    """Tests for label smoothing cross-entropy."""

    def test_gradient_shape(self, multiclass_data_3):
        """Test label smoothing gradient has correct shape."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = label_smoothing(n_classes=n_classes, smoothing=0.1)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (len(y_true), n_classes)

    def test_zero_smoothing_equals_softmax_ce(self, multiclass_data_3):
        """Test smoothing=0 equals standard softmax CE."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj_smooth = label_smoothing(n_classes=n_classes, smoothing=0.0)
        obj_ce = softmax_cross_entropy(n_classes=n_classes)

        grad_smooth = obj_smooth.gradient(y_pred, y_true)
        grad_ce = obj_ce.gradient(y_pred, y_true)

        np.testing.assert_allclose(grad_smooth, grad_ce, rtol=1e-4)

    def test_smoothing_reduces_confidence(self, multiclass_data_3):
        """Test label smoothing reduces gradient magnitude."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj_no_smooth = label_smoothing(n_classes=n_classes, smoothing=0.0)
        obj_smooth = label_smoothing(n_classes=n_classes, smoothing=0.2)

        grad_no = obj_no_smooth.gradient(y_pred, y_true)
        grad_yes = obj_smooth.gradient(y_pred, y_true)

        # Smoothing generally reduces gradient magnitude
        # (targets are softer, so less aggressive updates)
        assert np.mean(np.abs(grad_yes)) <= np.mean(np.abs(grad_no)) * 1.1


# =============================================================================
# Class Balanced Tests
# =============================================================================


class TestClassBalanced:
    """Tests for class-balanced loss."""

    def test_gradient_shape(self, multiclass_data_3):
        """Test class-balanced gradient has correct shape."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj = class_balanced(n_classes=n_classes)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (len(y_true), n_classes)

    def test_no_samples_equals_uniform(self, multiclass_data_3):
        """Test without samples_per_class, weights are uniform."""
        y_pred, y_true, n_classes = multiclass_data_3

        obj_balanced = class_balanced(n_classes=n_classes, samples_per_class=None)
        obj_ce = softmax_cross_entropy(n_classes=n_classes)

        grad_balanced = obj_balanced.gradient(y_pred, y_true)
        grad_ce = obj_ce.gradient(y_pred, y_true)

        # Should be identical when no class weights
        np.testing.assert_allclose(grad_balanced, grad_ce, rtol=1e-4)

    def test_imbalanced_weights_affect_gradient(self, multiclass_data_3):
        """Test imbalanced class counts affect gradients."""
        y_pred, y_true, n_classes = multiclass_data_3

        # Highly imbalanced: class 0 has 1000 samples, others have 10
        samples = np.array([1000, 10, 10])
        obj = class_balanced(n_classes=n_classes, samples_per_class=samples)

        grad = obj.gradient(y_pred, y_true)

        # Just check it runs and produces valid output
        assert grad.shape == (len(y_true), n_classes)
        assert not np.any(np.isnan(grad))


# =============================================================================
# Sample Weight Tests
# =============================================================================


class TestMulticlassSampleWeights:
    """Tests for sample weight handling in multiclass objectives."""

    def test_sample_weights_scale_gradients(self, multiclass_data_3):
        """Test sample weights scale gradients correctly."""
        y_pred, y_true, n_classes = multiclass_data_3
        weights = np.random.rand(len(y_true))

        obj = softmax_cross_entropy(n_classes=n_classes)

        grad_unweighted, hess_unweighted = obj.grad_hess(y_pred, y_true)
        grad_weighted, hess_weighted = obj.grad_hess(y_pred, y_true, sample_weight=weights)

        # Each row should be scaled by corresponding weight
        expected_grad = grad_unweighted * weights[:, np.newaxis]
        expected_hess = hess_unweighted * weights[:, np.newaxis]

        np.testing.assert_allclose(grad_weighted, expected_grad, rtol=1e-5)
        np.testing.assert_allclose(hess_weighted, expected_hess, rtol=1e-5)
