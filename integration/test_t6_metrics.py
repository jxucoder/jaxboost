"""T6: Evaluation metrics tests."""

import numpy as np
import pytest
import xgboost as xgb


class TestRegressionMetrics:
    """Test regression evaluation metrics."""

    def test_mse_metric(self, regression_data, xgb_params):
        """MSE metric integrates with XGBoost training."""
        from jaxboost import huber
        from jaxboost.metric import mse_metric

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = mse_metric()

        evals_result = {}
        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=huber.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            evals_result=evals_result,
            verbose_eval=False
        )

        # Verify metric was recorded
        assert "test" in evals_result
        assert "mse" in evals_result["test"]
        assert len(evals_result["test"]["mse"]) == 50

        # MSE should decrease during training
        first_mse = evals_result["test"]["mse"][0]
        last_mse = evals_result["test"]["mse"][-1]
        assert last_mse < first_mse

    def test_mae_metric(self, regression_data, xgb_params):
        """MAE metric integrates with XGBoost training."""
        from jaxboost import huber
        from jaxboost.metric import mae_metric

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = mae_metric()

        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=huber.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        preds = model.predict(dtest)
        assert np.all(np.isfinite(preds))

    def test_rmse_metric(self, regression_data, xgb_params):
        """RMSE metric."""
        from jaxboost import huber
        from jaxboost.metric import rmse_metric

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = rmse_metric()

        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=huber.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        preds = model.predict(dtest)
        assert np.all(np.isfinite(preds))

    def test_r2_metric(self, regression_data, xgb_params):
        """R2 metric."""
        from jaxboost import huber
        from jaxboost.metric import r2_metric

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = r2_metric()

        evals_result = {}
        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=huber.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            evals_result=evals_result,
            verbose_eval=False
        )

        # R2 should be positive for a reasonable model
        last_r2 = evals_result["test"]["r2"][-1]
        assert last_r2 > 0


class TestOrdinalMetrics:
    """Test ordinal-specific evaluation metrics."""

    def test_qwk_metric(self, ordinal_data, xgb_params):
        """QWK metric for ordinal regression."""
        from jaxboost.objective import ordinal_logit
        from jaxboost.metric import qwk_metric

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        ordinal = ordinal_logit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = qwk_metric(n_classes=6, transform=ordinal.predict)

        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=ordinal.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        preds = model.predict(dtest)
        assert len(preds) == len(y_test)

    def test_ordinal_accuracy_metric(self, ordinal_data, xgb_params):
        """Ordinal accuracy metric."""
        from jaxboost.objective import ordinal_logit
        from jaxboost.metric import ordinal_accuracy_metric

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        ordinal = ordinal_logit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        params = {**xgb_params, "disable_default_eval_metric": 1}
        metric = ordinal_accuracy_metric(n_classes=6, transform=ordinal.predict)

        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=ordinal.xgb_objective,
            custom_metric=metric.xgb_metric,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        preds = model.predict(dtest)
        assert len(preds) == len(y_test)
