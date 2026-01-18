"""
Tests for ordinal regression objective functions.

Tests the Cumulative Link Model implementation with probit and logit links.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxboost.objective import (
    OLLObjective,
    OrdinalObjective,
    SLACEObjective,
    SORDObjective,
    SquaredCDFObjective,
    hybrid_ordinal,
    oll_objective,
    ordinal_logit,
    ordinal_probit,
    ordinal_regression,
    qwk_ordinal,
    slace_objective,
    sord_objective,
    squared_cdf_ordinal,
)
from jaxboost.objective.ordinal import QWKOrdinalObjective

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def ordinal_data_6():
    """Generate sample ordinal data with 6 classes (like wine quality)."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 6
    # Latent predictions g(x)
    y_pred = np.random.randn(n_samples).astype(np.float64)
    # Ordinal labels: 0, 1, 2, 3, 4, 5
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_classes


@pytest.fixture
def ordinal_logits_data_6():
    """Generate sample ordinal data with 6 classes (logits)."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 6
    # Logits: (n_samples, n_classes)
    y_pred = np.random.randn(n_samples * n_classes).astype(np.float64)
    # Ordinal labels: 0, 1, 2, 3, 4, 5
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_samples, n_classes


@pytest.fixture
def ordinal_data_3():
    """Generate sample ordinal data with 3 classes."""
    np.random.seed(42)
    n_samples = 50
    n_classes = 3
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randint(0, n_classes, n_samples).astype(np.float64)
    return y_pred, y_true, n_classes


@pytest.fixture
def initialized_ordinal_6(ordinal_data_6):
    """Create an OrdinalObjective with thresholds initialized."""
    y_pred, y_true, n_classes = ordinal_data_6
    obj = OrdinalObjective(n_classes=n_classes, link="probit")
    obj.init_thresholds_from_data(y_true)
    return obj, y_pred, y_true


# =============================================================================
# OrdinalObjective Creation Tests
# =============================================================================


class TestOrdinalObjectiveCreation:
    """Tests for OrdinalObjective initialization."""

    def test_create_with_n_classes(self):
        """Test creating OrdinalObjective with n_classes."""
        obj = OrdinalObjective(n_classes=5)
        assert obj.n_classes == 5
        assert obj.link == "probit"  # default

    def test_create_with_logit_link(self):
        """Test creating with logit link function."""
        obj = OrdinalObjective(n_classes=5, link="logit")
        assert obj.link == "logit"

    def test_create_with_probit_link(self):
        """Test creating with probit link function."""
        obj = OrdinalObjective(n_classes=5, link="probit")
        assert obj.link == "probit"

    def test_invalid_link_raises(self):
        """Test invalid link function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown link function"):
            OrdinalObjective(n_classes=5, link="invalid")

    def test_factory_functions(self):
        """Test ordinal_probit and ordinal_logit factory functions."""
        probit = ordinal_probit(n_classes=4)
        logit = ordinal_logit(n_classes=4)

        assert probit.n_classes == 4
        assert probit.link == "probit"
        assert logit.n_classes == 4
        assert logit.link == "logit"

    def test_ordinal_regression_factory(self):
        """Test ordinal_regression factory function."""
        obj = ordinal_regression(n_classes=6, link="logit")
        assert obj.n_classes == 6
        assert obj.link == "logit"


# =============================================================================
# Threshold Tests
# =============================================================================


class TestThresholds:
    """Tests for threshold initialization and setting."""

    def test_thresholds_not_initialized_raises(self, ordinal_data_6):
        """Test accessing thresholds before init raises."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = OrdinalObjective(n_classes=n_classes)

        with pytest.raises(ValueError, match="Thresholds not initialized"):
            _ = obj.thresholds

    def test_init_thresholds_from_data(self, ordinal_data_6):
        """Test threshold initialization from data."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = OrdinalObjective(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        assert obj.thresholds is not None
        assert len(obj.thresholds) == n_classes - 1
        # Thresholds should be strictly increasing
        assert np.all(np.diff(obj.thresholds) > 0)

    def test_set_thresholds_manually(self):
        """Test manually setting thresholds."""
        obj = OrdinalObjective(n_classes=4)
        thresholds = np.array([-1.0, 0.0, 1.0])
        obj.set_thresholds(thresholds)

        np.testing.assert_array_equal(obj.thresholds, thresholds)

    def test_wrong_threshold_count_raises(self):
        """Test wrong number of thresholds raises error."""
        obj = OrdinalObjective(n_classes=4)  # expects 3 thresholds

        with pytest.raises(ValueError, match="Expected 3 thresholds"):
            obj.set_thresholds(np.array([-1.0, 0.0]))  # only 2

    def test_non_increasing_thresholds_raises(self):
        """Test non-increasing thresholds raises error."""
        obj = OrdinalObjective(n_classes=4)

        with pytest.raises(ValueError, match="strictly increasing"):
            obj.set_thresholds(np.array([1.0, 0.0, -1.0]))  # decreasing

    def test_create_with_thresholds(self):
        """Test creating OrdinalObjective with thresholds."""
        thresholds = np.array([-1.0, 0.0, 1.0])
        obj = OrdinalObjective(n_classes=4, thresholds=thresholds)

        np.testing.assert_array_equal(obj.thresholds, thresholds)


# =============================================================================
# Gradient and Hessian Tests
# =============================================================================


class TestGradientHessian:
    """Tests for gradient and Hessian computation."""

    def test_gradient_shape(self, initialized_ordinal_6):
        """Test gradient has correct shape."""
        obj, y_pred, y_true = initialized_ordinal_6
        grad = obj.gradient(y_pred, y_true)

        assert grad.shape == (len(y_pred),)
        assert grad.dtype == np.float64

    def test_hessian_shape(self, initialized_ordinal_6):
        """Test Hessian has correct shape."""
        obj, y_pred, y_true = initialized_ordinal_6
        hess = obj.hessian(y_pred, y_true)

        assert hess.shape == (len(y_pred),)
        assert hess.dtype == np.float64

    def test_hessian_positive(self, initialized_ordinal_6):
        """Test Hessian is positive (for XGBoost stability)."""
        obj, y_pred, y_true = initialized_ordinal_6
        hess = obj.hessian(y_pred, y_true)

        assert np.all(hess > 0)

    def test_grad_hess_returns_tuple(self, initialized_ordinal_6):
        """Test grad_hess returns tuple of (grad, hess)."""
        obj, y_pred, y_true = initialized_ordinal_6
        grad, hess = obj.grad_hess(y_pred, y_true)

        assert grad.shape == hess.shape == (len(y_pred),)

    def test_gradient_not_nan(self, initialized_ordinal_6):
        """Test gradient doesn't contain NaN."""
        obj, y_pred, y_true = initialized_ordinal_6
        grad = obj.gradient(y_pred, y_true)

        assert not np.any(np.isnan(grad))

    def test_hessian_not_nan(self, initialized_ordinal_6):
        """Test Hessian doesn't contain NaN."""
        obj, y_pred, y_true = initialized_ordinal_6
        hess = obj.hessian(y_pred, y_true)

        assert not np.any(np.isnan(hess))


# =============================================================================
# Link Function Tests
# =============================================================================


class TestLinkFunctions:
    """Tests comparing probit and logit link functions."""

    def test_probit_and_logit_both_work(self, ordinal_data_6):
        """Test both link functions produce valid gradients."""
        y_pred, y_true, n_classes = ordinal_data_6

        probit = ordinal_probit(n_classes=n_classes)
        logit = ordinal_logit(n_classes=n_classes)

        probit.init_thresholds_from_data(y_true)
        logit.init_thresholds_from_data(y_true)

        grad_probit = probit.gradient(y_pred, y_true)
        grad_logit = logit.gradient(y_pred, y_true)

        assert not np.any(np.isnan(grad_probit))
        assert not np.any(np.isnan(grad_logit))

    def test_probit_logit_similar_behavior(self, ordinal_data_6):
        """Test probit and logit have similar (not identical) behavior."""
        y_pred, y_true, n_classes = ordinal_data_6

        probit = ordinal_probit(n_classes=n_classes)
        logit = ordinal_logit(n_classes=n_classes)

        probit.init_thresholds_from_data(y_true)
        logit.init_thresholds_from_data(y_true)

        grad_probit = probit.gradient(y_pred, y_true)
        grad_logit = logit.gradient(y_pred, y_true)

        # Correlation should be high (same direction)
        correlation = np.corrcoef(grad_probit, grad_logit)[0, 1]
        assert correlation > 0.9


# =============================================================================
# Prediction Tests
# =============================================================================


class TestPrediction:
    """Tests for predict and predict_proba methods."""

    def test_predict_proba_shape(self, initialized_ordinal_6):
        """Test predict_proba returns correct shape."""
        obj, y_pred, y_true = initialized_ordinal_6
        probs = obj.predict_proba(y_pred)

        assert probs.shape == (len(y_pred), obj.n_classes)

    def test_predict_proba_sums_to_one(self, initialized_ordinal_6):
        """Test probabilities sum to 1 for each sample."""
        obj, y_pred, y_true = initialized_ordinal_6
        probs = obj.predict_proba(y_pred)

        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_non_negative(self, initialized_ordinal_6):
        """Test all probabilities are non-negative."""
        obj, y_pred, y_true = initialized_ordinal_6
        probs = obj.predict_proba(y_pred)

        assert np.all(probs >= 0)

    def test_predict_shape(self, initialized_ordinal_6):
        """Test predict returns correct shape."""
        obj, y_pred, y_true = initialized_ordinal_6
        predictions = obj.predict(y_pred)

        assert predictions.shape == (len(y_pred),)

    def test_predict_valid_classes(self, initialized_ordinal_6):
        """Test predicted classes are in valid range."""
        obj, y_pred, y_true = initialized_ordinal_6
        predictions = obj.predict(y_pred)

        assert np.all(predictions >= 0)
        assert np.all(predictions < obj.n_classes)

    def test_predict_matches_argmax_proba(self, initialized_ordinal_6):
        """Test predict matches argmax of predict_proba."""
        obj, y_pred, y_true = initialized_ordinal_6

        predictions = obj.predict(y_pred)
        probs = obj.predict_proba(y_pred)
        argmax_probs = np.argmax(probs, axis=1)

        np.testing.assert_array_equal(predictions, argmax_probs)


# =============================================================================
# Ordinal Property Tests
# =============================================================================


class TestOrdinalProperties:
    """Tests for ordinal-specific properties."""

    def test_higher_latent_predicts_higher_class(self):
        """Test higher latent values predict higher ordinal classes."""
        n_classes = 5
        obj = OrdinalObjective(n_classes=n_classes, link="probit")
        # Set evenly spaced thresholds
        obj.set_thresholds(np.array([-1.5, -0.5, 0.5, 1.5]))

        # Low latent value should predict low class
        low_pred = obj.predict(np.array([-3.0]))
        # High latent value should predict high class
        high_pred = obj.predict(np.array([3.0]))

        assert low_pred[0] < high_pred[0]

    def test_monotonic_cumulative_probs(self):
        """Test cumulative probabilities are monotonic."""
        n_classes = 5
        obj = OrdinalObjective(n_classes=n_classes, link="probit")
        obj.set_thresholds(np.array([-1.5, -0.5, 0.5, 1.5]))

        y_pred = np.array([0.0])
        probs = obj.predict_proba(y_pred)[0]

        # Cumulative probs should be monotonically increasing
        cum_probs = np.cumsum(probs)
        assert np.all(np.diff(cum_probs) >= 0)


# =============================================================================
# Sample Weight Tests
# =============================================================================


class TestSampleWeights:
    """Tests for sample weight handling."""

    def test_sample_weights_scale_gradients(self, initialized_ordinal_6):
        """Test sample weights scale gradients correctly."""
        obj, y_pred, y_true = initialized_ordinal_6
        weights = np.random.rand(len(y_true))

        grad_unweighted, hess_unweighted = obj.grad_hess(y_pred, y_true)
        grad_weighted, hess_weighted = obj.grad_hess(y_pred, y_true, sample_weight=weights)

        expected_grad = grad_unweighted * weights
        expected_hess = hess_unweighted * weights

        np.testing.assert_allclose(grad_weighted, expected_grad, rtol=1e-5)
        np.testing.assert_allclose(hess_weighted, expected_hess, rtol=1e-5)

    def test_zero_weight_zeros_gradient(self, initialized_ordinal_6):
        """Test zero weight produces zero gradient."""
        obj, y_pred, y_true = initialized_ordinal_6
        weights = np.zeros(len(y_true))

        grad, hess = obj.grad_hess(y_pred, y_true, sample_weight=weights)

        np.testing.assert_array_equal(grad, 0.0)
        np.testing.assert_array_equal(hess, 0.0)


# =============================================================================
# XGBoost Objective Tests
# =============================================================================


class TestXGBoostObjective:
    """Tests for XGBoost objective interface."""

    def test_xgb_objective_property(self, initialized_ordinal_6):
        """Test xgb_objective property returns callable."""
        obj, y_pred, y_true = initialized_ordinal_6

        xgb_obj = obj.xgb_objective
        assert callable(xgb_obj)

    def test_get_xgb_objective(self, initialized_ordinal_6):
        """Test get_xgb_objective returns callable."""
        obj, y_pred, y_true = initialized_ordinal_6

        xgb_obj = obj.get_xgb_objective()
        assert callable(xgb_obj)

    def test_repr(self, initialized_ordinal_6):
        """Test __repr__ returns informative string."""
        obj, y_pred, y_true = initialized_ordinal_6

        repr_str = repr(obj)
        assert "OrdinalObjective" in repr_str
        assert "n_classes=6" in repr_str
        assert "probit" in repr_str


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_extreme_positive_prediction(self, ordinal_data_6):
        """Test stability with very large positive predictions."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = OrdinalObjective(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        # Very large positive values
        extreme_pred = np.ones(len(y_pred)) * 100.0

        grad = obj.gradient(extreme_pred, y_true)
        hess = obj.hessian(extreme_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(grad))
        assert not np.any(np.isinf(hess))

    def test_extreme_negative_prediction(self, ordinal_data_6):
        """Test stability with very large negative predictions."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = OrdinalObjective(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        # Very large negative values
        extreme_pred = np.ones(len(y_pred)) * -100.0

        grad = obj.gradient(extreme_pred, y_true)
        hess = obj.hessian(extreme_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))
        assert not np.any(np.isinf(grad))
        assert not np.any(np.isinf(hess))

    def test_zero_prediction(self, ordinal_data_6):
        """Test stability with zero predictions."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = OrdinalObjective(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        zero_pred = np.zeros(len(y_pred))

        grad = obj.gradient(zero_pred, y_true)
        hess = obj.hessian(zero_pred, y_true)

        assert not np.any(np.isnan(grad))
        assert not np.any(np.isnan(hess))


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestLossFunction:
    """Tests for the loss function computation."""

    def test_loss_shape(self, initialized_ordinal_6):
        """Test loss has correct shape."""
        obj, y_pred, y_true = initialized_ordinal_6
        loss = obj.loss(y_pred, y_true)

        assert loss.shape == (len(y_pred),)

    def test_loss_non_negative(self, initialized_ordinal_6):
        """Test loss values are non-negative."""
        obj, y_pred, y_true = initialized_ordinal_6
        loss = obj.loss(y_pred, y_true)

        assert np.all(loss >= 0)

    def test_correct_prediction_low_loss(self):
        """Test correct prediction has lower loss than incorrect."""
        n_classes = 3
        obj = OrdinalObjective(n_classes=n_classes)
        obj.set_thresholds(np.array([-1.0, 1.0]))

        # Middle class (1) is most likely when g(x) = 0
        y_true = np.array([1.0])

        # Correct prediction (middle range)
        correct_pred = np.array([0.0])
        # Wrong prediction (far from middle)
        wrong_pred = np.array([5.0])  # Predicts class 2

        loss_correct = obj.loss(correct_pred, y_true)[0]
        loss_wrong = obj.loss(wrong_pred, y_true)[0]

        assert loss_correct < loss_wrong


# =============================================================================
# QWKOrdinalObjective Tests
# =============================================================================


class TestQWKOrdinalObjective:
    """Tests for QWK-aligned ordinal objective (Expected Quadratic Error)."""

    def test_qwk_ordinal_creation(self):
        """Test creating QWKOrdinalObjective."""
        obj = qwk_ordinal(n_classes=6)
        assert isinstance(obj, QWKOrdinalObjective)
        assert obj.n_classes == 6
        assert obj.alpha == 0.0  # Pure EQE
        assert obj.beta == 1.0

    def test_hybrid_ordinal_creation(self):
        """Test creating hybrid ordinal objective."""
        obj = hybrid_ordinal(n_classes=6, nll_weight=0.7, eqe_weight=0.3)
        assert obj.alpha == 0.7
        assert obj.beta == 0.3

    def test_qwk_ordinal_gradient_shape(self, ordinal_data_6):
        """Test QWK ordinal gradient has correct shape."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = qwk_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))

        grad = obj.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_qwk_ordinal_gradient_not_nan(self, ordinal_data_6):
        """Test QWK ordinal gradient is not NaN."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = qwk_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))

        grad = obj.gradient(y_pred, y_true)
        assert not np.any(np.isnan(grad))

    def test_qwk_ordinal_hessian_positive(self, ordinal_data_6):
        """Test QWK ordinal Hessian is positive (for XGBoost stability)."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = qwk_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true.astype(np.int32))

        hess = obj.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_hybrid_loss_combines_nll_and_eqe(self, ordinal_data_6):
        """Test hybrid loss is between pure NLL and pure EQE."""
        y_pred, y_true, n_classes = ordinal_data_6

        nll_obj = ordinal_logit(n_classes=n_classes)
        eqe_obj = qwk_ordinal(n_classes=n_classes)
        hybrid_obj = hybrid_ordinal(n_classes=n_classes, nll_weight=0.5, eqe_weight=0.5)

        for obj in [nll_obj, eqe_obj, hybrid_obj]:
            obj.init_thresholds_from_data(y_true.astype(np.int32))

        nll_loss = nll_obj.loss(y_pred, y_true).mean()
        eqe_loss = eqe_obj.loss(y_pred, y_true).mean()
        hybrid_loss = hybrid_obj.loss(y_pred, y_true).mean()

        # Hybrid should be roughly 0.5 * NLL + 0.5 * EQE
        expected = 0.5 * nll_loss + 0.5 * eqe_loss
        assert np.isclose(hybrid_loss, expected, rtol=0.1)

    def test_qwk_ordinal_repr(self):
        """Test QWKOrdinalObjective string representation."""
        obj = qwk_ordinal(n_classes=6)
        repr_str = repr(obj)
        assert "QWKOrdinalObjective" in repr_str
        assert "n_classes=6" in repr_str
        assert "alpha=0.0" in repr_str
        assert "beta=1.0" in repr_str
        assert "gauss_newton=True" in repr_str


# =============================================================================
# Squared CDF (CRPS) Tests
# =============================================================================


class TestSquaredCDF:
    """Tests for SquaredCDFObjective."""

    def test_creation(self):
        """Test creating SquaredCDFObjective."""
        obj = squared_cdf_ordinal(n_classes=6)
        assert isinstance(obj, SquaredCDFObjective)
        assert obj.n_classes == 6
        assert obj.link == "logit"  # default

    def test_gradient_shape(self, ordinal_data_6):
        """Test gradient output shape."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = squared_cdf_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        grad = obj.gradient(y_pred, y_true)
        assert grad.shape == y_pred.shape
        assert grad.dtype == np.float64
        assert not np.any(np.isnan(grad))

    def test_hessian_positive(self, ordinal_data_6):
        """Test Hessian is positive (Gauss-Newton approximation)."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = squared_cdf_ordinal(n_classes=n_classes)
        obj.init_thresholds_from_data(y_true)

        hess = obj.hessian(y_pred, y_true)
        assert np.all(hess > 0)

    def test_probit_link(self, ordinal_data_6):
        """Test with probit link."""
        y_pred, y_true, n_classes = ordinal_data_6
        obj = squared_cdf_ordinal(n_classes=n_classes, link="probit")
        obj.init_thresholds_from_data(y_true)

        grad = obj.gradient(y_pred, y_true)
        assert not np.any(np.isnan(grad))


# =============================================================================
# SORD Tests
# =============================================================================


class TestSORD:
    """Tests for SORDObjective."""

    def test_creation(self):
        """Test creating SORDObjective."""
        obj = sord_objective(n_classes=6, alpha=2.0)
        assert isinstance(obj, SORDObjective)
        assert obj.n_classes == 6
        assert obj.alpha == 2.0

    def test_soft_targets(self):
        """Test soft target generation."""
        n_classes = 5
        obj = sord_objective(n_classes=n_classes, alpha=1.0)

        # Test for class 2
        y_true = jnp.array(2)
        targets = obj._soft_targets(y_true)

        assert targets.shape == (n_classes,)
        # Should peak at 2
        assert np.argmax(targets) == 2
        # Should be symmetric around 2 (1 and 3 equal, 0 and 4 equal)
        np.testing.assert_allclose(targets[1], targets[3])
        np.testing.assert_allclose(targets[0], targets[4])
        # Should sum to 1
        np.testing.assert_allclose(np.sum(targets), 1.0, atol=1e-5)

    def test_gradient_shape(self, ordinal_logits_data_6):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_classes = ordinal_logits_data_6
        obj = sord_objective(n_classes=n_classes)

        grad = obj.gradient(y_pred, y_true)
        # Should be flattened for XGBoost
        assert grad.shape == (n_samples, n_classes)
        assert grad.dtype == np.float64
        assert not np.any(np.isnan(grad))

    def test_sklearn_objective(self, ordinal_logits_data_6):
        """Test sklearn-compatible objective."""
        y_pred, y_true, n_samples, n_classes = ordinal_logits_data_6
        y_probs = np.abs(y_pred.reshape(n_samples, n_classes))
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        obj = sord_objective(n_classes=n_classes)
        sk_obj = obj.sklearn_objective

        grad, hess = sk_obj(y_true, y_probs)
        assert grad.shape == (n_samples, n_classes)
        assert hess.shape == (n_samples, n_classes)


# =============================================================================
# OLL Tests
# =============================================================================


class TestOLL:
    """Tests for OLLObjective."""

    def test_creation(self):
        """Test creating OLLObjective."""
        obj = oll_objective(n_classes=6, alpha=1.5)
        assert isinstance(obj, OLLObjective)
        assert obj.n_classes == 6
        assert obj.alpha == 1.5

    def test_gradient_shape(self, ordinal_logits_data_6):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_classes = ordinal_logits_data_6
        obj = oll_objective(n_classes=n_classes)

        grad = obj.gradient(y_pred, y_true)
        assert grad.shape == (n_samples, n_classes)
        assert not np.any(np.isnan(grad))

    def test_distance_weighting(self):
        """Test that errors further away are penalized more."""
        n_classes = 5
        obj = oll_objective(n_classes=n_classes, alpha=2.0)

        # Single sample, true class 2
        y_true = np.array([2])

        # Pred A: high prob on 1 (dist 1)
        logits_a = np.array([-10, 10, -10, -10, -10]).reshape(1, 5)
        # Pred B: high prob on 0 (dist 2)
        logits_b = np.array([10, -10, -10, -10, -10]).reshape(1, 5)

        loss_a = obj.loss(logits_a, y_true)
        loss_b = obj.loss(logits_b, y_true)

        # Dist 2 error should have higher loss than Dist 1 error
        assert loss_b > loss_a


# =============================================================================
# SLACE Tests
# =============================================================================


class TestSLACE:
    """Tests for SLACEObjective."""

    def test_creation(self):
        """Test creating SLACEObjective."""
        obj = slace_objective(n_classes=6)
        assert isinstance(obj, SLACEObjective)
        assert obj.n_classes == 6

    def test_dominance_matrices(self):
        """Test dominance matrix construction."""
        obj = slace_objective(n_classes=3)
        # Check dominance matrix for y=1 (middle)
        # Classes: 0, 1, 2
        # Distances to 1: |0-1|=1, |1-1|=0, |2-1|=1
        # D[1][i,j] = 1 if dist(j,1) <= dist(i,1)

        d1 = obj._dom_matrices[1]

        # Row 0 (class 0, dist 1): dominated by 1 (dist 0), 0 (dist 1), 2 (dist 1)
        assert d1[0, 1] == 1  # 1 dominates 0
        assert d1[0, 0] == 1  # 0 dominates 0
        assert d1[0, 2] == 1  # 2 dominates 0

        # Row 1 (class 1, dist 0): dominated only by 1 (dist 0)
        assert d1[1, 1] == 1
        assert d1[1, 0] == 0
        assert d1[1, 2] == 0

    def test_gradient_shape(self, ordinal_logits_data_6):
        """Test gradient output shape."""
        y_pred, y_true, n_samples, n_classes = ordinal_logits_data_6
        obj = slace_objective(n_classes=n_classes)

        grad = obj.gradient(y_pred, y_true)
        assert grad.shape == (n_samples, n_classes)
        assert not np.any(np.isnan(grad))

    def test_sklearn_objective(self, ordinal_logits_data_6):
        """Test sklearn-compatible objective."""
        y_pred, y_true, n_samples, n_classes = ordinal_logits_data_6
        y_probs = np.abs(y_pred.reshape(n_samples, n_classes))
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        obj = slace_objective(n_classes=n_classes)
        sk_obj = obj.sklearn_objective

        grad, hess = sk_obj(y_true, y_probs)
        assert grad.shape == (n_samples, n_classes)
        assert hess.shape == (n_samples, n_classes)
