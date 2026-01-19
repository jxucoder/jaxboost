"""T5: Ordinal regression objective tests."""

import numpy as np
import pytest
import xgboost as xgb


class TestOrdinalObjectives:
    """Test ordinal regression objectives."""

    def test_ordinal_logit_basic(self, ordinal_data, xgb_params):
        """Ordinal logit trains and produces valid probabilities."""
        from jaxboost.objective import ordinal_logit

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        ordinal = ordinal_logit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        model = xgb.train(xgb_params, dtrain, num_boost_round=100, obj=ordinal.xgb_objective)
        raw_preds = model.predict(dtest)

        # Get probabilities
        probs = ordinal.predict_proba(raw_preds)

        # Verify probability properties
        assert probs.shape == (len(y_test), 6)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_ordinal_probit(self, ordinal_data, xgb_params):
        """Ordinal probit trains and predicts."""
        from jaxboost.objective import ordinal_probit

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        ordinal = ordinal_probit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        model = xgb.train(xgb_params, dtrain, num_boost_round=100, obj=ordinal.xgb_objective)
        raw_preds = model.predict(dtest)

        classes = ordinal.predict(raw_preds)

        assert classes.shape == (len(y_test),)
        assert np.all(classes >= 0)
        assert np.all(classes <= 5)

    def test_qwk_ordinal(self, ordinal_data, xgb_params):
        """QWK-aligned ordinal objective."""
        from jaxboost.objective import qwk_ordinal

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        qwk = qwk_ordinal(n_classes=6)
        qwk.init_thresholds_from_data(y_train)

        model = xgb.train(xgb_params, dtrain, num_boost_round=100, obj=qwk.xgb_objective)
        raw_preds = model.predict(dtest)

        classes = qwk.predict(raw_preds)
        accuracy = np.mean(classes == y_test)

        assert accuracy > 0.15  # Better than random (1/6)

    def test_squared_cdf_ordinal(self, ordinal_data, xgb_params):
        """Squared CDF (CRPS) ordinal objective."""
        from jaxboost.objective import squared_cdf_ordinal

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        sqcdf = squared_cdf_ordinal(n_classes=6)
        sqcdf.init_thresholds_from_data(y_train)

        model = xgb.train(xgb_params, dtrain, num_boost_round=100, obj=sqcdf.xgb_objective)
        raw_preds = model.predict(dtest)

        probs = sqcdf.predict_proba(raw_preds)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_thresholds_increasing(self, ordinal_data):
        """Initialized thresholds are strictly increasing."""
        from jaxboost.objective import ordinal_logit

        X_train, X_test, y_train, y_test = ordinal_data

        ordinal = ordinal_logit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        thresholds = np.array(ordinal.thresholds)
        assert len(thresholds) == 5  # K-1 thresholds for K classes
        assert np.all(np.diff(thresholds) > 0)  # Strictly increasing

    def test_ordinal_with_qwk_metric(self, ordinal_data, xgb_params):
        """Ordinal objective with built-in QWK metric."""
        from jaxboost.objective import ordinal_logit

        X_train, X_test, y_train, y_test = ordinal_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        ordinal = ordinal_logit(n_classes=6)
        ordinal.init_thresholds_from_data(y_train)

        params = {**xgb_params, "disable_default_eval_metric": 1}

        # Train with QWK metric monitoring
        model = xgb.train(
            params, dtrain, num_boost_round=50,
            obj=ordinal.xgb_objective,
            custom_metric=ordinal.qwk_metric.xgb_metric,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        # Model should train without error
        preds = model.predict(dtest)
        assert len(preds) == len(y_test)
