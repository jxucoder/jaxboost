"""
Tests for LightGBM objective wrappers.

LightGBM flattens 2D grad/hess with ``ravel(order="F")`` when ``num_class > 1``,
so multi-output objectives must return (n_samples, n_outputs) arrays.
"""

import numpy as np
import pytest

from jaxboost.objective import (
    OLLObjective,
    SLACEObjective,
    SORDObjective,
    gaussian_nll,
    softmax_cross_entropy,
)


class MockDataset:
    def __init__(self, label, weight=None):
        self._label = label
        self._weight = weight

    def get_label(self):
        return self._label

    def get_weight(self):
        return self._weight


class TestMultiOutputLgbObjective:
    def test_returns_sample_major_2d(self):
        rng = np.random.default_rng(0)
        n = 20
        y_pred = rng.normal(size=(n, 2))
        y_true = rng.normal(size=n)
        obj = gaussian_nll()

        grad, hess = obj.lgb_objective(y_pred, MockDataset(y_true))
        expected_grad, expected_hess = obj.grad_hess(y_pred, y_true)

        assert grad.shape == (n, 2)
        assert hess.shape == (n, 2)
        np.testing.assert_allclose(grad, expected_grad.reshape(n, 2))
        np.testing.assert_allclose(hess, expected_hess.reshape(n, 2))

    def test_sample_weight(self):
        rng = np.random.default_rng(1)
        n = 10
        y_pred = rng.normal(size=(n, 2))
        y_true = rng.normal(size=n)
        weight = rng.uniform(0.5, 2.0, size=n)
        obj = gaussian_nll()

        grad, _ = obj.lgb_objective(y_pred, MockDataset(y_true, weight))
        unweighted, _ = obj.lgb_objective(y_pred, MockDataset(y_true))

        np.testing.assert_allclose(grad, unweighted * weight[:, None], rtol=1e-6)


class TestMultiClassOrdinalPredict:
    @pytest.mark.parametrize("cls", [SORDObjective, OLLObjective, SLACEObjective])
    def test_predict_accepts_2d_and_flat(self, cls):
        rng = np.random.default_rng(2)
        logits = rng.normal(size=(10, 3))
        obj = cls(n_classes=3)

        expected = np.argmax(logits, axis=1)
        np.testing.assert_array_equal(obj.predict(logits), expected)
        np.testing.assert_array_equal(obj.predict(logits.ravel()), expected)


class TestLightGBMTraining:
    def test_gaussian_nll_learns_mean_and_variance(self):
        lgb = pytest.importorskip("lightgbm")
        rng = np.random.default_rng(0)
        n = 500
        x = rng.normal(size=(n, 3))
        y = 2 * x[:, 0] + rng.normal(size=n) * np.exp(x[:, 1])

        model = lgb.train(
            {"objective": gaussian_nll().lgb_objective, "num_class": 2, "verbose": -1},
            lgb.Dataset(x, y),
            num_boost_round=50,
        )
        pred = model.predict(x, raw_score=True)

        assert np.corrcoef(pred[:, 0], y)[0, 1] > 0.5
        assert np.corrcoef(pred[:, 1], x[:, 1])[0, 1] > 0.5

    def test_softmax_cross_entropy_trains(self):
        lgb = pytest.importorskip("lightgbm")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 3))
        y = (x[:, 0] > 0).astype(int) + (x[:, 1] > 0).astype(int)

        model = lgb.train(
            {
                "objective": softmax_cross_entropy(n_classes=3).lgb_objective,
                "num_class": 3,
                "verbose": -1,
            },
            lgb.Dataset(x, y),
            num_boost_round=30,
        )
        acc = (model.predict(x, raw_score=True).argmax(axis=1) == y).mean()
        assert acc > 0.9
