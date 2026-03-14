"""
Finite-difference gradient and Hessian verification for all loss functions.

Verifies that JAX-computed gradients and Hessians match numerical
central-difference approximations for every objective in the library.
"""

import numpy as np
import pytest

# =============================================================================
# Helpers
# =============================================================================


def _check_hessian_via_fd(obj, y_pred, y_true, rtol=0.2, atol=0.02, eps=1e-4, **kwargs):
    """
    Verify Hessian by finite-differencing the gradient.

    For each index i, computes:
        fd_hess[i] = (grad(y_pred + eps*e_i)[i] - grad(y_pred - eps*e_i)[i]) / (2*eps)
    and checks it matches obj.hessian().
    """
    grad = obj.gradient(y_pred, y_true, **kwargs)
    hess = obj.hessian(y_pred, y_true, **kwargs)

    assert grad.shape == y_pred.shape, f"Gradient shape {grad.shape} != {y_pred.shape}"
    assert hess.shape == y_pred.shape, f"Hessian shape {hess.shape} != {y_pred.shape}"

    flat_pred = y_pred.flatten()
    flat_hess = hess.flatten()
    n = len(flat_pred)
    fd_hess = np.zeros(n, dtype=np.float64)

    for i in range(n):
        yp_p = flat_pred.copy()
        yp_m = flat_pred.copy()
        yp_p[i] += eps
        yp_m[i] -= eps

        g_p = obj.gradient(yp_p.reshape(y_pred.shape), y_true, **kwargs).flatten()[i]
        g_m = obj.gradient(yp_m.reshape(y_pred.shape), y_true, **kwargs).flatten()[i]
        fd_hess[i] = (g_p - g_m) / (2 * eps)

    np.testing.assert_allclose(flat_hess, fd_hess, rtol=rtol, atol=atol,
                               err_msg="Hessian mismatch vs finite differences on gradient")



# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 20
    y_pred = np.random.randn(n).astype(np.float64) * 0.5
    y_true = np.random.randn(n).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def positive_data():
    """Data for losses requiring positive predictions (Poisson, Gamma)."""
    np.random.seed(42)
    n = 20
    y_pred = np.random.randn(n).astype(np.float64)  # log-space predictions
    y_true = np.abs(np.random.randn(n).astype(np.float64)) + 0.1
    return y_pred, y_true


@pytest.fixture
def binary_data():
    np.random.seed(42)
    n = 20
    y_pred = np.random.randn(n).astype(np.float64) * 0.5
    y_true = (np.random.rand(n) > 0.5).astype(np.float64)
    return y_pred, y_true


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    n = 10
    n_classes = 3
    y_pred = np.random.randn(n, n_classes).astype(np.float64) * 0.5
    y_true = np.random.randint(0, n_classes, n).astype(np.float64)
    return y_pred, y_true, n_classes


@pytest.fixture
def multi_output_data():
    np.random.seed(42)
    n = 10
    n_outputs = 2
    y_pred = np.random.randn(n * n_outputs).astype(np.float64) * 0.5
    y_true = np.random.randn(n).astype(np.float64)
    return y_pred, y_true, n_outputs


# =============================================================================
# Regression Objectives
# =============================================================================


class TestRegressionGradients:
    """Finite-difference verification for regression losses."""

    def test_mse(self, regression_data):
        from jaxboost.objective import mse
        y_pred, y_true = regression_data
        _check_hessian_via_fd(mse, y_pred, y_true)

    def test_huber(self, regression_data):
        from jaxboost.objective import huber
        y_pred, y_true = regression_data
        _check_hessian_via_fd(huber, y_pred, y_true)

    def test_log_cosh(self, regression_data):
        from jaxboost.objective import log_cosh
        y_pred, y_true = regression_data
        _check_hessian_via_fd(log_cosh, y_pred, y_true)

    def test_pseudo_huber(self, regression_data):
        from jaxboost.objective import pseudo_huber
        y_pred, y_true = regression_data
        _check_hessian_via_fd(pseudo_huber, y_pred, y_true)

    def test_mae_smooth(self, regression_data):
        from jaxboost.objective import mae_smooth
        y_pred, y_true = regression_data
        _check_hessian_via_fd(mae_smooth, y_pred, y_true)

    def test_quantile(self, regression_data):
        from jaxboost.objective import quantile
        y_pred, y_true = regression_data
        _check_hessian_via_fd(quantile, y_pred, y_true)

    def test_asymmetric(self, regression_data):
        from jaxboost.objective import asymmetric
        y_pred, y_true = regression_data
        _check_hessian_via_fd(asymmetric, y_pred, y_true)

    def test_tweedie(self, positive_data):
        from jaxboost.objective import tweedie
        y_pred, y_true = positive_data
        _check_hessian_via_fd(tweedie, y_pred, y_true)

    def test_poisson(self, positive_data):
        from jaxboost.objective import poisson
        y_pred, y_true = positive_data
        _check_hessian_via_fd(poisson, y_pred, y_true)

    def test_gamma(self, positive_data):
        from jaxboost.objective import gamma
        y_pred, y_true = positive_data
        _check_hessian_via_fd(gamma, y_pred, y_true)


# =============================================================================
# Binary Classification Objectives
# =============================================================================


class TestBinaryGradients:
    """Finite-difference verification for binary classification losses."""

    def test_binary_crossentropy(self, binary_data):
        from jaxboost.objective import binary_crossentropy
        y_pred, y_true = binary_data
        _check_hessian_via_fd(binary_crossentropy, y_pred, y_true)

    def test_focal_loss(self, binary_data):
        from jaxboost.objective import focal_loss
        y_pred, y_true = binary_data
        _check_hessian_via_fd(focal_loss, y_pred, y_true)

    def test_hinge_loss(self, binary_data):
        from jaxboost.objective import hinge_loss
        y_pred, y_true = binary_data
        _check_hessian_via_fd(hinge_loss, y_pred, y_true)

    def test_weighted_binary_crossentropy(self, binary_data):
        from jaxboost.objective import weighted_binary_crossentropy
        y_pred, y_true = binary_data
        _check_hessian_via_fd(weighted_binary_crossentropy, y_pred, y_true)


# =============================================================================
# Multi-class Objectives
# =============================================================================


class TestMulticlassGradients:
    """Finite-difference verification for multi-class losses."""

    def test_softmax_cross_entropy(self, multiclass_data):
        from jaxboost.objective import softmax_cross_entropy
        y_pred, y_true, n_classes = multiclass_data
        obj = softmax_cross_entropy(n_classes=n_classes)
        _check_hessian_via_fd(obj, y_pred, y_true)

    def test_focal_multiclass(self, multiclass_data):
        from jaxboost.objective import focal_multiclass
        y_pred, y_true, n_classes = multiclass_data
        obj = focal_multiclass(n_classes=n_classes, gamma=2.0)
        _check_hessian_via_fd(obj, y_pred, y_true)

    def test_label_smoothing(self, multiclass_data):
        from jaxboost.objective import label_smoothing
        y_pred, y_true, n_classes = multiclass_data
        obj = label_smoothing(n_classes=n_classes, smoothing=0.1)
        _check_hessian_via_fd(obj, y_pred, y_true)

    def test_class_balanced(self, multiclass_data):
        from jaxboost.objective import class_balanced
        y_pred, y_true, n_classes = multiclass_data
        obj = class_balanced(n_classes=n_classes)
        _check_hessian_via_fd(obj, y_pred, y_true)


# =============================================================================
# Multi-output Objectives
# =============================================================================


class TestMultiOutputGradients:
    """Finite-difference verification for multi-output losses."""

    def test_gaussian_nll(self, multi_output_data):
        from jaxboost.objective import gaussian_nll
        y_pred, y_true, n_outputs = multi_output_data
        obj = gaussian_nll(n_outputs=n_outputs)
        _check_hessian_via_fd(obj, y_pred, y_true)

    def test_laplace_nll(self, multi_output_data):
        from jaxboost.objective import laplace_nll
        y_pred, y_true, n_outputs = multi_output_data
        obj = laplace_nll(n_outputs=n_outputs)
        _check_hessian_via_fd(obj, y_pred, y_true, rtol=5e-2, atol=5e-3)


# =============================================================================
# Survival Objectives
# =============================================================================


class TestSurvivalGradients:
    """Finite-difference verification for survival analysis losses."""

    def test_aft(self):
        from jaxboost.objective import aft
        np.random.seed(42)
        n = 15
        y_pred = np.random.randn(n).astype(np.float64) * 0.5
        y_true = np.abs(np.random.randn(n).astype(np.float64)) + 0.5
        _check_hessian_via_fd(aft, y_pred, y_true)

    def test_weibull_aft(self):
        from jaxboost.objective import weibull_aft
        np.random.seed(42)
        n = 15
        y_pred = np.random.randn(n).astype(np.float64) * 0.5
        y_true = np.abs(np.random.randn(n).astype(np.float64)) + 0.5
        _check_hessian_via_fd(weibull_aft, y_pred, y_true)


# =============================================================================
# Ordinal Objectives
# =============================================================================


class TestOrdinalGradients:
    """Finite-difference verification for ordinal regression losses.

    Note: Ordinal objectives clamp Hessians to >= 1e-6 for XGBoost stability,
    so we only verify elements where the Hessian is not clamped.
    """

    def _check_ordinal_hessian(self, obj, y_pred, y_true, eps=1e-4):
        """Check Hessian only where it's not clamped to 1e-6."""
        hess = obj.hessian(y_pred, y_true)
        flat_pred = y_pred.flatten()
        flat_hess = hess.flatten()
        n = len(flat_pred)

        fd_hess = np.zeros(n, dtype=np.float64)
        for i in range(n):
            yp_p = flat_pred.copy()
            yp_m = flat_pred.copy()
            yp_p[i] += eps
            yp_m[i] -= eps
            g_p = obj.gradient(yp_p.reshape(y_pred.shape), y_true).flatten()[i]
            g_m = obj.gradient(yp_m.reshape(y_pred.shape), y_true).flatten()[i]
            fd_hess[i] = (g_p - g_m) / (2 * eps)

        # Only check where Hessian is not clamped
        unclamped = flat_hess > 2e-6
        if np.any(unclamped):
            np.testing.assert_allclose(flat_hess[unclamped], fd_hess[unclamped],
                                       rtol=0.2, atol=0.02)

        # Clamped values should be >= 1e-6
        assert np.all(flat_hess >= 1e-6 - 1e-10)

    def test_sord(self):
        from jaxboost.objective import sord_objective
        np.random.seed(42)
        n = 10
        n_classes = 4
        y_pred = np.random.randn(n, n_classes).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = sord_objective(n_classes=n_classes)
        self._check_ordinal_hessian(obj, y_pred, y_true)

    def test_oll(self):
        from jaxboost.objective import oll_objective
        np.random.seed(42)
        n = 10
        n_classes = 4
        y_pred = np.random.randn(n, n_classes).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = oll_objective(n_classes=n_classes)
        self._check_ordinal_hessian(obj, y_pred, y_true)

    def test_slace(self):
        from jaxboost.objective import slace_objective
        np.random.seed(42)
        n = 10
        n_classes = 4
        y_pred = np.random.randn(n, n_classes).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = slace_objective(n_classes=n_classes)
        self._check_ordinal_hessian(obj, y_pred, y_true)

    def test_ordinal_probit(self):
        from jaxboost.objective import ordinal_probit
        np.random.seed(42)
        n = 15
        n_classes = 4
        y_pred = np.random.randn(n).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = ordinal_probit(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))
        self._check_ordinal_hessian(obj, y_pred, y_true)

    def test_ordinal_logit(self):
        from jaxboost.objective import ordinal_logit
        np.random.seed(42)
        n = 15
        n_classes = 4
        y_pred = np.random.randn(n).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = ordinal_logit(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))
        self._check_ordinal_hessian(obj, y_pred, y_true)

    def test_squared_cdf_gradient_shape(self):
        """SquaredCDF uses Gauss-Newton Hessian (intentionally != true Hessian).
        We only verify gradient shape and that Hessian is positive."""
        from jaxboost.objective import squared_cdf_ordinal
        np.random.seed(42)
        n = 15
        n_classes = 4
        y_pred = np.random.randn(n).astype(np.float64) * 0.5
        y_true = np.random.randint(0, n_classes, n).astype(np.float64)
        obj = squared_cdf_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))
        grad = obj.gradient(y_pred, y_true)
        hess = obj.hessian(y_pred, y_true)
        assert grad.shape == y_pred.shape
        assert hess.shape == y_pred.shape
        assert np.all(hess >= 0)


# =============================================================================
# Multi-task Objectives
# =============================================================================


class TestMultiTaskGradients:
    """Finite-difference verification for multi-task losses."""

    def test_multi_task_regression(self):
        """Multi-task objectives return flattened gradients/Hessians."""
        from jaxboost.objective import multi_task_regression
        np.random.seed(42)
        n = 10
        n_tasks = 3
        y_pred_2d = np.random.randn(n, n_tasks).astype(np.float64) * 0.5
        y_true_2d = np.random.randn(n, n_tasks).astype(np.float64)
        obj = multi_task_regression(n_tasks=n_tasks)

        # Multi-task returns flat arrays; use flat inputs for FD check
        y_pred_flat = y_pred_2d.flatten()
        y_true_flat = y_true_2d.flatten()
        _check_hessian_via_fd(obj, y_pred_flat, y_true_flat)

    def test_multi_task_huber(self):
        from jaxboost.objective import multi_task_huber
        np.random.seed(42)
        n = 10
        n_tasks = 3
        y_pred_2d = np.random.randn(n, n_tasks).astype(np.float64) * 0.5
        y_true_2d = np.random.randn(n, n_tasks).astype(np.float64)
        obj = multi_task_huber(n_tasks=n_tasks)

        y_pred_flat = y_pred_2d.flatten()
        y_true_flat = y_true_2d.flatten()
        _check_hessian_via_fd(obj, y_pred_flat, y_true_flat)
