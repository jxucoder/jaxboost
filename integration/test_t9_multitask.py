"""T9: Multi-task learning objective tests."""

import numpy as np
import pytest
import xgboost as xgb


class TestMultiTaskObjectives:
    """Test multi-task learning objectives."""

    @pytest.fixture
    def multitask_data(self):
        """Multi-task regression dataset: 3 targets."""
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = np.column_stack([
            X[:, 0] + np.random.randn(500) * 0.1,
            X[:, 1] * 2 + np.random.randn(500) * 0.1,
            X[:, 0] + X[:, 1] + np.random.randn(500) * 0.1,
        ])

        return X[:400], X[400:], y[:400], y[400:]

    def test_multi_task_regression(self, multitask_data):
        """Multi-task regression objective."""
        from jaxboost import multi_task_regression

        X_train, X_test, y_train, y_test = multitask_data
        n_tasks = y_train.shape[1]

        # Flatten targets for XGBoost
        y_train_flat = y_train.flatten()
        dtrain = xgb.DMatrix(np.tile(X_train, (n_tasks, 1)), label=y_train_flat)

        params = {"max_depth": 4, "eta": 0.1, "verbosity": 0}
        mt_obj = multi_task_regression(n_tasks=n_tasks)

        model = xgb.train(params, dtrain, num_boost_round=50, obj=mt_obj.xgb_objective)
        
        # Verify model trained without error
        assert model is not None

    def test_multi_task_huber(self, multitask_data):
        """Multi-task Huber loss."""
        from jaxboost import multi_task_huber

        X_train, X_test, y_train, y_test = multitask_data
        n_tasks = y_train.shape[1]

        y_train_flat = y_train.flatten()
        dtrain = xgb.DMatrix(np.tile(X_train, (n_tasks, 1)), label=y_train_flat)

        params = {"max_depth": 4, "eta": 0.1, "verbosity": 0}
        mt_obj = multi_task_huber(n_tasks=n_tasks)

        model = xgb.train(params, dtrain, num_boost_round=50, obj=mt_obj.xgb_objective)
        
        assert model is not None


class TestUncertaintyObjectives:
    """Test uncertainty estimation objectives."""

    def test_gaussian_nll(self, regression_data, xgb_params):
        """Gaussian NLL for mean + variance prediction."""
        from jaxboost import gaussian_nll

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Gaussian NLL outputs 2 values: mean and log_var
        params = {**xgb_params, "num_class": 2}
        gnll = gaussian_nll()

        model = xgb.train(params, dtrain, num_boost_round=50, obj=gnll.xgb_objective)
        preds = model.predict(dtest)

        # Should have predictions
        assert preds is not None

    def test_laplace_nll(self, regression_data, xgb_params):
        """Laplace NLL for median + scale prediction."""
        from jaxboost import laplace_nll

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "num_class": 2}
        lnll = laplace_nll()

        model = xgb.train(params, dtrain, num_boost_round=50, obj=lnll.xgb_objective)
        preds = model.predict(dtest)

        assert preds is not None
