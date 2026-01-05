"""
Tests for multi-task and multi-output objective functions.

Tests gradient/Hessian correctness for MTL objectives with missing label support.
"""

import numpy as np
import pytest

from jaxboost.objective import (
    MaskedMultiTaskObjective,
    MultiOutputObjective,
    gaussian_nll,
    laplace_nll,
    masked_multi_task_objective,
    multi_output_objective,
    multi_task_classification,
    multi_task_huber,
    multi_task_quantile,
    multi_task_regression,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def multi_task_data():
    """Generate sample multi-task data."""
    np.random.seed(42)
    n_samples = 50
    n_tasks = 3
    # Predictions and labels shape: (n_samples, n_tasks)
    y_pred = np.random.randn(n_samples * n_tasks).astype(np.float64)
    y_true = np.random.randn(n_samples * n_tasks).astype(np.float64)
    return y_pred, y_true, n_samples, n_tasks


@pytest.fixture
def multi_task_data_with_mask():
    """Generate multi-task data with missing labels."""
    np.random.seed(42)
    n_samples = 50
    n_tasks = 3
    y_pred = np.random.randn(n_samples, n_tasks).astype(np.float64)
    y_true = np.random.randn(n_samples, n_tasks).astype(np.float64)
    # 30% missing labels
    mask = (np.random.rand(n_samples, n_tasks) > 0.3).astype(np.float32)
    return y_pred, y_true, mask, n_samples, n_tasks


@pytest.fixture
def multi_output_data():
    """Generate sample multi-output data (for uncertainty estimation)."""
    np.random.seed(42)
    n_samples = 50
    n_outputs = 2
    y_pred = np.random.randn(n_samples * n_outputs).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)
    return y_pred, y_true, n_samples, n_outputs


# =============================================================================
# MaskedMultiTaskObjective Tests
# =============================================================================


class TestMaskedMultiTaskObjective:
    """Tests for MaskedMultiTaskObjective base class."""

    def test_default_creates_mse_loss(self):
        """Test default creates MSE loss."""
        obj = MaskedMultiTaskObjective(n_tasks=3)
        assert obj.n_tasks == 3
        assert obj._name == "<lambda>"  # Default lambda

    def test_decorator_creates_instance(self):
        """Test @masked_multi_task_objective decorator."""

        @masked_multi_task_objective(n_tasks=3)
        def my_loss(y_pred, y_true):
            return (y_pred - y_true) ** 2

        assert isinstance(my_loss, MaskedMultiTaskObjective)
        assert my_loss.n_tasks == 3

    def test_gradient_shape(self, multi_task_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_regression(n_tasks=n_tasks)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_tasks,)
        assert grad.dtype == np.float64

    def test_hessian_shape(self, multi_task_data):
        """Test Hessian output shape."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_regression(n_tasks=n_tasks)
        hess = obj.hessian(y_pred, y_true)

        assert hess.shape == (n_samples * n_tasks,)
        assert hess.dtype == np.float64


# =============================================================================
# Missing Label Tests
# =============================================================================


class TestMissingLabels:
    """Tests for missing label handling."""

    def test_mask_zeros_gradient(self, multi_task_data_with_mask):
        """Test mask=0 produces zero gradient."""
        y_pred, y_true, mask, n_samples, n_tasks = multi_task_data_with_mask

        obj = multi_task_regression(n_tasks=n_tasks)
        grad = obj.gradient(y_pred.flatten(), y_true.flatten(), mask=mask)

        # Reshape for easier checking
        grad_2d = grad.reshape(n_samples, n_tasks)
        mask_2d = mask.reshape(n_samples, n_tasks)

        # Where mask is 0, gradient should be 0
        masked_grad = grad_2d[mask_2d == 0]
        np.testing.assert_allclose(masked_grad, 0.0, atol=1e-10)

    def test_mask_ones_nonzero_gradient(self, multi_task_data_with_mask):
        """Test mask=1 produces non-zero gradient."""
        y_pred, y_true, mask, n_samples, n_tasks = multi_task_data_with_mask

        obj = multi_task_regression(n_tasks=n_tasks)
        grad = obj.gradient(y_pred.flatten(), y_true.flatten(), mask=mask)

        # Reshape for easier checking
        grad_2d = grad.reshape(n_samples, n_tasks)
        mask_2d = mask.reshape(n_samples, n_tasks)

        # Where mask is 1 and y_pred != y_true, gradient should be non-zero
        valid_grad = grad_2d[mask_2d == 1]
        # At least some should be non-zero
        assert np.any(valid_grad != 0)

    def test_no_mask_all_valid(self, multi_task_data):
        """Test without mask, all labels are valid."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_regression(n_tasks=n_tasks)
        grad = obj.gradient(y_pred, y_true, mask=None)

        # All gradients should be non-zero (unless y_pred == y_true)
        assert grad.shape == (n_samples * n_tasks,)


# =============================================================================
# Multi-Task Regression Tests
# =============================================================================


class TestMultiTaskRegression:
    """Tests for multi-task regression."""

    def test_gradient_correctness(self, multi_task_data):
        """Test gradient matches MSE gradient per task."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_regression(n_tasks=n_tasks)
        grad = obj.gradient(y_pred, y_true)

        # MSE gradient: 2 * (y_pred - y_true)
        expected = 2 * (y_pred - y_true)
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_hessian_positive(self, multi_task_data):
        """Test Hessian is positive."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_regression(n_tasks=n_tasks)
        hess = obj.hessian(y_pred, y_true)

        # Hessian should be positive (at least 1e-6 due to clipping)
        assert np.all(hess >= 1e-6)


# =============================================================================
# Multi-Task Classification Tests
# =============================================================================


class TestMultiTaskClassification:
    """Tests for multi-task binary classification."""

    def test_gradient_shape(self):
        """Test gradient output shape."""
        np.random.seed(42)
        n_samples, n_tasks = 50, 3
        y_pred = np.random.randn(n_samples * n_tasks).astype(np.float64)
        y_true = (np.random.rand(n_samples * n_tasks) > 0.5).astype(np.float64)

        obj = multi_task_classification(n_tasks=n_tasks)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_tasks,)

    def test_hessian_positive(self):
        """Test Hessian is positive."""
        np.random.seed(42)
        n_samples, n_tasks = 50, 3
        y_pred = np.random.randn(n_samples * n_tasks).astype(np.float64)
        y_true = (np.random.rand(n_samples * n_tasks) > 0.5).astype(np.float64)

        obj = multi_task_classification(n_tasks=n_tasks)
        hess = obj.hessian(y_pred, y_true)

        assert np.all(hess > 0)


# =============================================================================
# Multi-Task Huber Tests
# =============================================================================


class TestMultiTaskHuber:
    """Tests for multi-task Huber loss."""

    def test_gradient_shape(self, multi_task_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_huber(n_tasks=n_tasks, delta=1.0)
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_tasks,)

    def test_hessian_positive(self, multi_task_data):
        """Test Hessian is positive (with tolerance for float precision)."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_huber(n_tasks=n_tasks)
        hess = obj.hessian(y_pred, y_true)

        # Allow small tolerance for floating point precision
        assert np.all(hess >= 1e-7)


# =============================================================================
# Multi-Task Quantile Tests
# =============================================================================


class TestMultiTaskQuantile:
    """Tests for multi-task quantile regression."""

    def test_gradient_shape(self, multi_task_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data

        obj = multi_task_quantile(n_tasks=n_tasks, quantiles=[0.1, 0.5, 0.9])
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_tasks,)

    def test_default_quantiles(self):
        """Test default quantiles are evenly spaced."""
        n_tasks = 5
        obj = multi_task_quantile(n_tasks=n_tasks)

        expected = np.linspace(0.1, 0.9, n_tasks)
        np.testing.assert_allclose(obj.quantiles, expected, rtol=1e-5)

    def test_quantiles_length_validation(self):
        """Test quantiles must match n_tasks."""
        with pytest.raises(ValueError):
            multi_task_quantile(n_tasks=3, quantiles=[0.1, 0.5])  # Wrong length


# =============================================================================
# Task Weights Tests
# =============================================================================


class TestTaskWeights:
    """Tests for task weighting."""

    def test_task_weights_scale_gradients(self, multi_task_data):
        """Test task weights scale gradients correctly."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data
        weights = [1.0, 2.0, 0.5]

        obj_unweighted = multi_task_regression(n_tasks=n_tasks)
        obj_weighted = MaskedMultiTaskObjective(n_tasks=n_tasks, task_weights=weights)

        grad_u = obj_unweighted.gradient(y_pred, y_true)
        grad_w = obj_weighted.gradient(y_pred, y_true)

        # Reshape for comparison
        grad_u_2d = grad_u.reshape(n_samples, n_tasks)
        grad_w_2d = grad_w.reshape(n_samples, n_tasks)

        # Each task's gradient should be scaled
        for t in range(n_tasks):
            np.testing.assert_allclose(grad_w_2d[:, t], grad_u_2d[:, t] * weights[t], rtol=1e-4)


# =============================================================================
# MultiOutputObjective Tests
# =============================================================================


class TestMultiOutputObjective:
    """Tests for MultiOutputObjective (uncertainty estimation)."""

    def test_decorator_creates_instance(self):
        """Test @multi_output_objective decorator."""

        @multi_output_objective(n_outputs=2)
        def my_loss(params, y_true):
            return (params[0] - y_true) ** 2

        assert isinstance(my_loss, MultiOutputObjective)
        assert my_loss.n_outputs == 2

    def test_gradient_shape(self, multi_output_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_outputs = multi_output_data

        obj = gaussian_nll()
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_outputs,)


# =============================================================================
# Gaussian NLL Tests
# =============================================================================


class TestGaussianNLL:
    """Tests for Gaussian negative log-likelihood."""

    def test_requires_two_outputs(self):
        """Test gaussian_nll requires n_outputs=2."""
        with pytest.raises(ValueError):
            gaussian_nll(n_outputs=3)

    def test_gradient_shape(self, multi_output_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_outputs = multi_output_data

        obj = gaussian_nll()
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_outputs,)

    def test_hessian_shape(self, multi_output_data):
        """Test Hessian output shape."""
        y_pred, y_true, n_samples, n_outputs = multi_output_data

        obj = gaussian_nll()
        hess = obj.hessian(y_pred, y_true)

        assert hess.shape == (n_samples * n_outputs,)


# =============================================================================
# Laplace NLL Tests
# =============================================================================


class TestLaplaceNLL:
    """Tests for Laplace negative log-likelihood."""

    def test_requires_two_outputs(self):
        """Test laplace_nll requires n_outputs=2."""
        with pytest.raises(ValueError):
            laplace_nll(n_outputs=3)

    def test_gradient_shape(self, multi_output_data):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_outputs = multi_output_data

        obj = laplace_nll()
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (n_samples * n_outputs,)


# =============================================================================
# Sample Weight Tests
# =============================================================================


class TestMultiTaskSampleWeights:
    """Tests for sample weight handling."""

    def test_sample_weights_scale_correctly(self, multi_task_data):
        """Test sample weights are applied per-sample."""
        y_pred, y_true, n_samples, n_tasks = multi_task_data
        weights = np.random.rand(n_samples)

        obj = multi_task_regression(n_tasks=n_tasks)

        grad_u, hess_u = obj.grad_hess(y_pred, y_true)
        grad_w, hess_w = obj.grad_hess(y_pred, y_true, sample_weight=weights)

        # Each sample's tasks should be scaled by same weight
        grad_u_2d = grad_u.reshape(n_samples, n_tasks)
        grad_w_2d = grad_w.reshape(n_samples, n_tasks)

        for s in range(n_samples):
            np.testing.assert_allclose(grad_w_2d[s, :], grad_u_2d[s, :] * weights[s], rtol=1e-4)
