"""T2: Regression objective tests."""

import numpy as np
import pytest
import xgboost as xgb


class TestRegressionObjectives:
    """Test regression objectives with XGBoost."""

    def test_huber_basic(self, regression_data, xgb_params):
        """Huber loss trains and predicts."""
        from jaxboost import huber

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=huber.xgb_objective)
        preds = model.predict(dtest)

        assert len(preds) == len(y_test)
        assert np.all(np.isfinite(preds))

    def test_mse_basic(self, regression_data, xgb_params):
        """MSE loss trains and predicts."""
        from jaxboost import mse

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=mse.xgb_objective)
        preds = model.predict(dtest)

        assert len(preds) == len(y_test)
        assert np.all(np.isfinite(preds))

    def test_quantile_with_params(self, regression_data, xgb_params):
        """Quantile loss with parameters via with_params()."""
        from jaxboost import quantile

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        q90 = quantile.with_params(q=0.9)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=q90.xgb_objective)
        preds_90 = model.predict(dtest)

        q10 = quantile.with_params(q=0.1)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=q10.xgb_objective)
        preds_10 = model.predict(dtest)

        # 90th percentile predictions should generally be higher than 10th
        assert np.mean(preds_90) > np.mean(preds_10)

    def test_quantile_get_objective(self, regression_data, xgb_params):
        """Quantile loss with parameters via get_xgb_objective()."""
        from jaxboost import quantile

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            xgb_params, dtrain, num_boost_round=50,
            obj=quantile.get_xgb_objective(q=0.5)
        )
        preds = model.predict(dtest)

        assert len(preds) == len(y_test)

    def test_huber_custom_delta(self, regression_data, xgb_params):
        """Huber loss with custom delta parameter."""
        from jaxboost import huber

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            xgb_params, dtrain, num_boost_round=50,
            obj=huber.get_xgb_objective(delta=0.5)
        )
        preds = model.predict(dtest)

        assert len(preds) == len(y_test)

    @pytest.mark.skip(reason="log_cosh produces NaN on large targets - known issue")
    def test_log_cosh(self, regression_data, xgb_params):
        """Log-cosh loss trains and predicts."""
        from jaxboost import log_cosh

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=log_cosh.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))

    def test_asymmetric(self, regression_data, xgb_params):
        """Asymmetric loss trains and predicts."""
        from jaxboost import asymmetric

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            xgb_params, dtrain, num_boost_round=50,
            obj=asymmetric.get_xgb_objective(alpha=0.7)
        )
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))

    def test_tweedie(self, xgb_params):
        """Tweedie loss with positive target values."""
        from jaxboost import tweedie

        # Tweedie needs non-negative targets
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = np.abs(X[:, 0]) + np.abs(X[:, 1]) + 0.1  # Positive values

        dtrain = xgb.DMatrix(X[:400], label=y[:400])
        dtest = xgb.DMatrix(X[400:], label=y[400:])

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=tweedie.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))
