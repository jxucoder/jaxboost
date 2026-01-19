"""T10: Edge case tests."""

import numpy as np
import pytest
import xgboost as xgb


class TestEdgeCases:
    """Test handling of edge cases."""

    def test_small_dataset(self, xgb_params):
        """Training on very small dataset."""
        from jaxboost import huber

        X = np.random.randn(20, 5)
        y = X[:, 0] + np.random.randn(20) * 0.1

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=10, obj=huber.xgb_objective)
        preds = model.predict(dtrain)

        assert len(preds) == 20

    def test_single_feature(self, xgb_params):
        """Training with single feature."""
        from jaxboost import mse

        X = np.random.randn(100, 1)
        y = X[:, 0] * 2 + np.random.randn(100) * 0.1

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=20, obj=mse.xgb_objective)
        preds = model.predict(dtrain)

        assert np.all(np.isfinite(preds))

    def test_constant_target(self, xgb_params):
        """Training with constant target (edge case)."""
        from jaxboost import huber

        X = np.random.randn(100, 5)
        y = np.ones(100) * 5.0  # Constant target

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=20, obj=huber.xgb_objective)
        preds = model.predict(dtrain)

        # Predictions should be close to constant
        assert np.all(np.isfinite(preds))

    def test_large_values(self, xgb_params):
        """Training with large target values."""
        from jaxboost import huber

        X = np.random.randn(100, 5)
        y = X[:, 0] * 1e6 + np.random.randn(100) * 1e4

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=huber.xgb_objective)
        preds = model.predict(dtrain)

        assert np.all(np.isfinite(preds))

    def test_negative_targets(self, xgb_params):
        """Training with negative target values."""
        from jaxboost import mse

        X = np.random.randn(100, 5)
        y = -np.abs(X[:, 0]) - 10  # All negative

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=mse.xgb_objective)
        preds = model.predict(dtrain)

        assert np.all(np.isfinite(preds))

    def test_zero_boost_rounds(self, xgb_params):
        """Zero boost rounds should still return model."""
        from jaxboost import huber

        X = np.random.randn(100, 5)
        y = X[:, 0]

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=0, obj=huber.xgb_objective)

        # Model exists even with 0 rounds
        assert model is not None

    def test_sample_weights(self, xgb_params):
        """Training with sample weights."""
        from jaxboost import huber

        X = np.random.randn(100, 5)
        y = X[:, 0] + np.random.randn(100) * 0.1
        weights = np.random.rand(100) + 0.5  # Positive weights

        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=huber.xgb_objective)
        preds = model.predict(dtrain)

        assert np.all(np.isfinite(preds))


class TestOrdinalEdgeCases:
    """Edge cases specific to ordinal regression."""

    def test_binary_ordinal(self, xgb_params):
        """Ordinal regression with only 2 classes."""
        from jaxboost.objective import ordinal_logit

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)  # Binary: 0 or 1

        ordinal = ordinal_logit(n_classes=2)
        ordinal.init_thresholds_from_data(y)

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=ordinal.xgb_objective)
        preds = model.predict(dtrain)

        classes = ordinal.predict(preds)
        assert np.all((classes == 0) | (classes == 1))

    def test_imbalanced_ordinal(self, xgb_params):
        """Ordinal regression with imbalanced classes."""
        from jaxboost.objective import ordinal_logit

        np.random.seed(42)
        X = np.random.randn(200, 5)
        # Heavily imbalanced: mostly class 2
        y = np.array([2] * 150 + [0] * 20 + [1] * 15 + [3] * 10 + [4] * 5)
        np.random.shuffle(y)

        ordinal = ordinal_logit(n_classes=5)
        ordinal.init_thresholds_from_data(y)

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=ordinal.xgb_objective)
        preds = model.predict(dtrain)

        probs = ordinal.predict_proba(preds)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
