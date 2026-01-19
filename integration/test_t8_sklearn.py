"""T8: Sklearn API integration tests."""

import numpy as np
import pytest
from xgboost import XGBRegressor, XGBClassifier


class TestSklearnAPI:
    """Test JAXBoost with sklearn-style XGBoost API."""

    def test_xgbregressor_huber(self, regression_data):
        """XGBRegressor with Huber objective."""
        from jaxboost import huber

        X_train, X_test, y_train, y_test = regression_data

        reg = XGBRegressor(
            objective=huber.sklearn_objective,
            n_estimators=50,
            max_depth=4,
            verbosity=0
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        assert len(preds) == len(y_test)
        assert np.all(np.isfinite(preds))

    def test_xgbregressor_quantile(self, regression_data):
        """XGBRegressor with quantile objective."""
        from jaxboost import quantile

        X_train, X_test, y_train, y_test = regression_data

        q90 = quantile.with_params(q=0.9)
        reg = XGBRegressor(
            objective=q90.sklearn_objective,
            n_estimators=50,
            max_depth=4,
            verbosity=0
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        assert len(preds) == len(y_test)

    def test_xgbregressor_custom(self, regression_data):
        """XGBRegressor with custom @auto_objective."""
        import jax.numpy as jnp
        from jaxboost import auto_objective

        @auto_objective
        def my_mse(y_pred, y_true):
            return (y_pred - y_true) ** 2

        X_train, X_test, y_train, y_test = regression_data

        reg = XGBRegressor(
            objective=my_mse.sklearn_objective,
            n_estimators=50,
            max_depth=4,
            verbosity=0
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)

        assert np.all(np.isfinite(preds))

    def test_xgbclassifier_focal(self, binary_data):
        """XGBClassifier with focal loss."""
        from jaxboost import focal_loss

        X_train, X_test, y_train, y_test = binary_data

        clf = XGBClassifier(
            objective=focal_loss.sklearn_objective,
            n_estimators=50,
            max_depth=4,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        clf.fit(X_train, y_train)

        # predict_proba should work
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)

        # predict should work
        preds = clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        assert accuracy > 0.6

    def test_sklearn_fit_predict_cycle(self, regression_data):
        """Complete fit-predict cycle works."""
        from jaxboost import huber

        X_train, X_test, y_train, y_test = regression_data

        reg = XGBRegressor(
            objective=huber.sklearn_objective,
            n_estimators=50,
            max_depth=4,
            verbosity=0
        )

        # fit
        reg.fit(X_train, y_train)

        # predict
        preds = reg.predict(X_test)

        # score (R2)
        score = reg.score(X_test, y_test)
        assert score > 0  # Should have positive R2
