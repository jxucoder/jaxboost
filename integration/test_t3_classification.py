"""T3: Classification objective tests (binary and multi-class)."""

import numpy as np
import pytest
import xgboost as xgb


class TestBinaryClassification:
    """Test binary classification objectives."""

    def test_focal_loss_basic(self, binary_data, xgb_params):
        """Focal loss trains and achieves reasonable accuracy."""
        from jaxboost import focal_loss
        import jax.nn

        X_train, X_test, y_train, y_test = binary_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=focal_loss.xgb_objective)
        preds = model.predict(dtest)

        # Convert logits to probabilities
        probs = np.array(jax.nn.sigmoid(preds))
        y_pred = (probs > 0.5).astype(int)
        accuracy = np.mean(y_pred == y_test)

        assert accuracy > 0.7  # Should beat random

    def test_binary_crossentropy(self, binary_data, xgb_params):
        """Binary cross-entropy trains and predicts."""
        from jaxboost import binary_crossentropy
        import jax.nn

        X_train, X_test, y_train, y_test = binary_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=binary_crossentropy.xgb_objective)
        preds = model.predict(dtest)

        probs = np.array(jax.nn.sigmoid(preds))
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_focal_loss_custom_gamma(self, binary_data, xgb_params):
        """Focal loss with custom gamma parameter."""
        from jaxboost import focal_loss

        X_train, X_test, y_train, y_test = binary_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            xgb_params, dtrain, num_boost_round=50,
            obj=focal_loss.get_xgb_objective(gamma=3.0)
        )
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))

    def test_hinge_loss(self, binary_data, xgb_params):
        """Hinge loss trains and predicts."""
        from jaxboost import hinge_loss

        X_train, X_test, y_train, y_test = binary_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(xgb_params, dtrain, num_boost_round=50, obj=hinge_loss.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))


class TestMulticlassClassification:
    """Test multi-class classification objectives."""

    def test_softmax_cross_entropy(self, multiclass_data, xgb_params):
        """Softmax cross-entropy for 3-class problem."""
        from jaxboost import softmax_cross_entropy

        X_train, X_test, y_train, y_test = multiclass_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "num_class": 3}
        softmax_loss = softmax_cross_entropy(n_classes=3)

        model = xgb.train(params, dtrain, num_boost_round=50, obj=softmax_loss.xgb_objective)
        preds = model.predict(dtest)

        # XGBoost returns class labels for multi-class
        y_pred = preds.astype(int)
        accuracy = np.mean(y_pred == y_test)

        assert accuracy > 0.4  # Should beat random (1/3)

    def test_focal_multiclass(self, multiclass_data, xgb_params):
        """Focal loss for multi-class."""
        from jaxboost import focal_multiclass

        X_train, X_test, y_train, y_test = multiclass_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "num_class": 3}
        focal = focal_multiclass(n_classes=3)

        model = xgb.train(params, dtrain, num_boost_round=50, obj=focal.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))

    def test_label_smoothing(self, multiclass_data, xgb_params):
        """Label smoothing for multi-class."""
        from jaxboost import label_smoothing

        X_train, X_test, y_train, y_test = multiclass_data
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {**xgb_params, "num_class": 3}
        smoothed = label_smoothing(n_classes=3, smoothing=0.1)

        model = xgb.train(params, dtrain, num_boost_round=50, obj=smoothed.xgb_objective)
        preds = model.predict(dtest)

        assert np.all(np.isfinite(preds))
