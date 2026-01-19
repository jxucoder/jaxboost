"""T4: Custom objective tests using @auto_objective decorator."""

import numpy as np
import pytest
import xgboost as xgb


class TestAutoObjective:
    """Test custom objectives via @auto_objective decorator."""

    def test_custom_mse(self, regression_data, xgb_params):
        """Custom MSE implementation matches expected behavior."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def custom_mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=custom_mse.xgb_objective)
        preds = model.predict(dtest)

        assert len(preds) == len(y_test)
        assert np.all(np.isfinite(preds))

    def test_custom_with_parameter(self, regression_data, xgb_params):
        """Custom objective with tunable parameter."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def asymmetric_mse(y_pred, y_true, alpha=0.5):
            error = y_true - y_pred
            return jnp.where(error > 0, alpha * error**2, (1 - alpha) * error**2)

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Use default alpha
        model1 = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=asymmetric_mse.xgb_objective)
        preds1 = model1.predict(dtest)

        # Use custom alpha via get_xgb_objective
        model2 = xgb.train(
            xgb_params, dtrain, num_boost_round=50,
            obj=asymmetric_mse.get_xgb_objective(alpha=0.9)
        )
        preds2 = model2.predict(dtest)

        assert np.all(np.isfinite(preds1))
        assert np.all(np.isfinite(preds2))

    def test_custom_with_params_method(self, regression_data, xgb_params):
        """Custom objective using with_params() method."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def weighted_mse(y_pred, y_true, weight=1.0):
            return weight * (y_pred - y_true) ** 2

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        heavy_weighted = weighted_mse.with_params(weight=2.0)
        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=heavy_weighted.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))

    def test_custom_huber(self, regression_data, xgb_params):
        """Custom Huber implementation."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def my_huber(y_pred, y_true, delta=1.0):
            error = y_pred - y_true
            abs_error = jnp.abs(error)
            return jnp.where(
                abs_error <= delta,
                0.5 * error**2,
                delta * (abs_error - 0.5 * delta)
            )

        X_train, X_test, y_train, y_test = regression_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=my_huber.xgb_objective)
        preds = model.predict(dtest)

        mse = np.mean((preds - y_test) ** 2)
        assert mse < 50000  # Reasonable MSE threshold

    def test_gradients_computed(self, regression_data):
        """Verify gradients and hessians are computed."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def simple_loss(y_pred, y_true):
            return (y_pred - y_true) ** 2

        X_train, X_test, y_train, y_test = regression_data

        # Manually call gradient and hessian methods
        y_pred = np.zeros_like(y_train)
        grad = simple_loss.gradient(y_pred, y_train)
        hess = simple_loss.hessian(y_pred, y_train)

        assert grad.shape == y_train.shape
        assert hess.shape == y_train.shape
        assert np.all(np.isfinite(grad))
        assert np.all(np.isfinite(hess))
        assert np.all(hess > 0)  # Hessian should be positive for MSE
