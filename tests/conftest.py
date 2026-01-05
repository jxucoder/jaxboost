"""
Pytest configuration and shared fixtures for jaxboost tests.
"""

# Force JAX to use CPU to avoid Metal/GPU compatibility issues in tests
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def regression_data_small():
    """Small regression dataset for quick tests."""
    np.random.seed(42)
    n_samples = 20
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def regression_data_medium():
    """Medium regression dataset."""
    np.random.seed(42)
    n_samples = 500
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def binary_data_small():
    """Small binary classification dataset."""
    np.random.seed(42)
    n_samples = 20
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = (np.random.rand(n_samples) > 0.5).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def multiclass_data_small():
    """Small multi-class dataset."""
    np.random.seed(42)
    n_samples = 20
    n_classes = 3
    y_pred = np.random.randn(n_samples, n_classes).astype(np.float64)
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_classes


# =============================================================================
# Numerical Testing Utilities
# =============================================================================


def numerical_gradient(func, x, eps=1e-5):
    """Compute numerical gradient via central differences.

    Args:
        func: Function that takes x and returns scalar or array
        x: Input array
        eps: Step size for finite differences

    Returns:
        Numerical gradient array
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        f_plus = func(x_plus)
        f_minus = func(x_minus)

        if hasattr(f_plus, "__len__"):
            grad[i] = (f_plus[i] - f_minus[i]) / (2 * eps)
        else:
            grad[i] = (f_plus - f_minus) / (2 * eps)

    return grad


def numerical_hessian_diag(func, x, eps=1e-4):
    """Compute diagonal of Hessian via central differences.

    Args:
        func: Function that takes x and returns scalar or array
        x: Input array
        eps: Step size for finite differences

    Returns:
        Diagonal Hessian array
    """
    hess = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_center = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        f_plus = func(x_plus)
        f_minus = func(x_minus)
        f_center = func(x_center)

        if hasattr(f_plus, "__len__"):
            hess[i] = (f_plus[i] - 2 * f_center[i] + f_minus[i]) / (eps**2)
        else:
            hess[i] = (f_plus - 2 * f_center + f_minus) / (eps**2)

    return hess


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# =============================================================================
# Tolerances
# =============================================================================


GRADIENT_RTOL = 1e-4
GRADIENT_ATOL = 1e-6
HESSIAN_RTOL = 1e-3
HESSIAN_ATOL = 1e-4
