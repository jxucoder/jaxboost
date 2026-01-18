"""
Tests for jaxboost.metric module.
"""

import numpy as np
import pytest

from jaxboost.metric import (
    Metric,
    accuracy_metric,
    adjacent_accuracy_metric,
    # Classification
    auc_metric,
    # Bounded
    bounded_mse_metric,
    f1_metric,
    log_loss_metric,
    mae_metric,
    make_metric,
    # Regression
    mse_metric,
    ordinal_accuracy_metric,
    ordinal_mae_metric,
    out_of_bounds_metric,
    qwk_metric,
    r2_metric,
    rmse_metric,
)


class TestMetricBase:
    """Test base Metric class."""

    def test_metric_creation(self):
        """Test creating a custom metric."""
        metric = Metric(
            name="test",
            fn=lambda y, p: np.mean((y - p) ** 2),
            higher_is_better=False,
        )
        assert metric.name == "test"
        assert metric.higher_is_better is False

    def test_metric_call(self):
        """Test calling a metric directly."""
        metric = Metric(
            name="mse",
            fn=lambda y, p: np.mean((y - p) ** 2),
            higher_is_better=False,
        )
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        assert metric(y_true, y_pred) == 0.0

    def test_metric_with_transform(self):
        """Test metric with prediction transform."""
        metric = Metric(
            name="acc",
            fn=lambda y, p: np.mean(y == p),
            transform=lambda p: (p > 0.5).astype(int),
            higher_is_better=True,
        )
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.4, 0.6, 0.8])
        assert metric(y_true, y_pred) == 1.0

    def test_make_metric_decorator(self):
        """Test make_metric decorator."""

        @make_metric("my_mse", higher_is_better=False)
        def my_mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        assert my_mse(y_true, y_pred) == pytest.approx(1 / 3)


class TestOrdinalMetrics:
    """Test ordinal regression metrics."""

    def test_qwk_perfect(self):
        """Test QWK with perfect predictions."""
        metric = qwk_metric(n_classes=5)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        assert metric(y_true, y_pred) == pytest.approx(1.0)

    def test_qwk_off_by_one(self):
        """Test QWK with off-by-one errors (less penalized)."""
        metric = qwk_metric(n_classes=5)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4, 4])  # Off by 1
        score = metric(y_true, y_pred)
        assert 0 < score < 1  # Should be positive but not perfect

    def test_qwk_with_transform(self):
        """Test QWK with custom transform."""

        def transform(p):
            return np.clip(np.round(p * 4), 0, 4).astype(int)

        metric = qwk_metric(n_classes=5, transform=transform)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert metric(y_true, y_pred) == pytest.approx(1.0)

    def test_ordinal_mae(self):
        """Test ordinal MAE."""
        metric = ordinal_mae_metric(n_classes=5)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4, 4])  # Off by 1 on average
        assert metric(y_true, y_pred) == pytest.approx(0.8)

    def test_ordinal_accuracy(self):
        """Test ordinal accuracy."""
        metric = ordinal_accuracy_metric(n_classes=5)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 3])  # 4/5 correct
        assert metric(y_true, y_pred) == pytest.approx(0.8)

    def test_adjacent_accuracy(self):
        """Test adjacent accuracy (within ±1)."""
        metric = adjacent_accuracy_metric(n_classes=5)
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4, 2])  # All within 1 except last (off by 2)
        assert metric(y_true, y_pred) == pytest.approx(0.8)


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_auc_perfect(self):
        """Test AUC with perfect separation."""
        metric = auc_metric(transform=lambda p: p)  # No sigmoid
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert metric(y_true, y_score) == pytest.approx(1.0)

    def test_auc_random(self):
        """Test AUC with random predictions."""
        metric = auc_metric(transform=lambda p: p)
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 1000)
        y_score = np.random.rand(1000)
        score = metric(y_true, y_score)
        assert 0.4 < score < 0.6  # Should be around 0.5

    def test_accuracy(self):
        """Test accuracy metric."""
        metric = accuracy_metric(threshold=0.5, transform=lambda p: p)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.8, 0.6, 0.9])  # 3/4 correct
        assert metric(y_true, y_pred) == pytest.approx(0.75)

    def test_f1_score(self):
        """Test F1 score."""
        metric = f1_metric(threshold=0.5, transform=lambda p: p)
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])  # TP=2, FP=0, FN=1
        # Precision = 2/2 = 1.0, Recall = 2/3 = 0.67, F1 = 0.8
        assert metric(y_true, y_pred) == pytest.approx(0.8)

    def test_log_loss(self):
        """Test log loss."""
        metric = log_loss_metric(transform=lambda p: p)
        y_true = np.array([1, 0])
        y_pred = np.array([0.9, 0.1])  # Good predictions
        score = metric(y_true, y_pred)
        assert score < 0.2  # Should be low for good predictions


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_mse(self):
        """Test MSE metric."""
        metric = mse_metric()
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 5])  # Error of 1 on last sample
        assert metric(y_true, y_pred) == pytest.approx(0.25)

    def test_rmse(self):
        """Test RMSE metric."""
        metric = rmse_metric()
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 5])
        assert metric(y_true, y_pred) == pytest.approx(0.5)

    def test_mae(self):
        """Test MAE metric."""
        metric = mae_metric()
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 3, 4, 5])  # All off by 1
        assert metric(y_true, y_pred) == pytest.approx(1.0)

    def test_r2(self):
        """Test R² metric."""
        metric = r2_metric()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect
        assert metric(y_true, y_pred) == pytest.approx(1.0)


class TestBoundedMetrics:
    """Test bounded regression metrics."""

    def test_bounded_mse(self):
        """Test bounded MSE with sigmoid transform."""
        metric = bounded_mse_metric()
        y_true = np.array([0.2, 0.5, 0.8])
        # Logits that produce these probabilities via sigmoid
        y_pred = np.array([-1.386, 0, 1.386])  # sigmoid gives ~[0.2, 0.5, 0.8]
        score = metric(y_true, y_pred)
        assert score < 0.01  # Should be very small

    def test_out_of_bounds(self):
        """Test out-of-bounds metric."""
        metric = out_of_bounds_metric(lower=0, upper=1)
        y_true = np.array([0.5, 0.5, 0.5, 0.5])
        y_pred = np.array([-0.1, 0.5, 0.9, 1.2])  # 2/4 out of bounds
        assert metric(y_true, y_pred) == pytest.approx(0.5)


class TestXGBoostInterface:
    """Test XGBoost/LightGBM metric interface."""

    def test_xgb_metric_returns_tuple(self):
        """Test that xgb_metric returns (name, value) tuple."""
        metric = mse_metric()

        class MockDMatrix:
            def get_label(self):
                return np.array([1, 2, 3])

        predt = np.array([1, 2, 4])
        name, value = metric.xgb_metric(predt, MockDMatrix())
        assert name == "mse"
        assert isinstance(value, float)

    def test_lgb_metric_returns_triple(self):
        """Test that lgb_metric returns (name, value, is_higher_better) tuple."""
        metric = mse_metric()

        class MockDataset:
            def get_label(self):
                return np.array([1, 2, 3])

        preds = np.array([1, 2, 4])
        name, value, higher_better = metric.lgb_metric(preds, MockDataset())
        assert name == "mse"
        assert isinstance(value, float)
        assert higher_better is False
