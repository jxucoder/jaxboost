"""T7: LightGBM integration tests."""

import numpy as np
import pytest
import lightgbm as lgb


class TestLightGBMIntegration:
    """Test JAXBoost objectives with LightGBM."""

    def test_huber_lightgbm(self, regression_data, lgb_params):
        """Huber loss works with LightGBM 4.x API."""
        from jaxboost import huber

        X_train, X_test, y_train, y_test = regression_data
        train_data = lgb.Dataset(X_train, label=y_train)

        # LightGBM 4.x: objective in params dict
        params = {**lgb_params, "objective": huber.lgb_objective}

        model = lgb.train(params, train_data, num_boost_round=50)
        preds = model.predict(X_test)

        assert len(preds) == len(y_test)
        assert np.all(np.isfinite(preds))

    def test_mse_lightgbm(self, regression_data, lgb_params):
        """MSE loss works with LightGBM."""
        from jaxboost import mse

        X_train, X_test, y_train, y_test = regression_data
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {**lgb_params, "objective": mse.lgb_objective}

        model = lgb.train(params, train_data, num_boost_round=50)
        preds = model.predict(X_test)

        assert np.all(np.isfinite(preds))

    def test_quantile_lightgbm(self, regression_data, lgb_params):
        """Quantile loss with parameters for LightGBM."""
        from jaxboost import quantile

        X_train, X_test, y_train, y_test = regression_data
        train_data = lgb.Dataset(X_train, label=y_train)

        q90 = quantile.with_params(q=0.9)
        params = {**lgb_params, "objective": q90.lgb_objective}

        model = lgb.train(params, train_data, num_boost_round=50)
        preds = model.predict(X_test)

        assert np.all(np.isfinite(preds))

    def test_focal_loss_lightgbm(self, binary_data, lgb_params):
        """Focal loss for binary classification with LightGBM."""
        from jaxboost import focal_loss
        import jax.nn

        X_train, X_test, y_train, y_test = binary_data
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {**lgb_params, "objective": focal_loss.lgb_objective}

        model = lgb.train(params, train_data, num_boost_round=50)
        preds = model.predict(X_test)

        # Convert to probabilities and check accuracy
        probs = np.array(jax.nn.sigmoid(preds))
        y_pred = (probs > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)

        assert accuracy > 0.6  # Should be better than random

    def test_custom_objective_lightgbm(self, regression_data, lgb_params):
        """Custom @auto_objective works with LightGBM."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def my_loss(y_pred, y_true, alpha=0.5):
            error = y_true - y_pred
            return jnp.where(error > 0, alpha * error**2, (1 - alpha) * error**2)

        X_train, X_test, y_train, y_test = regression_data
        train_data = lgb.Dataset(X_train, label=y_train)

        params = {**lgb_params, "objective": my_loss.lgb_objective}

        model = lgb.train(params, train_data, num_boost_round=50)
        preds = model.predict(X_test)

        assert np.all(np.isfinite(preds))

    @pytest.mark.skip(reason="OrdinalObjective doesn't have LightGBM support yet")
    def test_ordinal_lightgbm(self, ordinal_data):
        """Ordinal regression works with LightGBM."""
        pass
