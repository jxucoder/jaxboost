"""T1: Core import tests - verify package imports without errors."""

import pytest


class TestCoreImports:
    """Test that core package imports work."""

    def test_import_jaxboost(self):
        """Import main package."""
        import jaxboost
        assert hasattr(jaxboost, "__version__")

    def test_version_format(self):
        """Version string is valid semver."""
        import jaxboost
        parts = jaxboost.__version__.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts[:2])

    def test_import_regression_objectives(self):
        """Import regression objectives."""
        from jaxboost import huber, mse, quantile, tweedie, asymmetric, log_cosh
        from jaxboost import pseudo_huber, mae_smooth

    def test_import_binary_objectives(self):
        """Import binary classification objectives."""
        from jaxboost import focal_loss, binary_crossentropy, hinge_loss

    def test_import_multiclass_objectives(self):
        """Import multi-class objectives."""
        from jaxboost import softmax_cross_entropy, focal_multiclass, label_smoothing

    def test_import_auto_objective(self):
        """Import auto_objective decorator."""
        from jaxboost import auto_objective, AutoObjective

    def test_import_ordinal_from_submodule(self):
        """Import ordinal objectives from submodule."""
        from jaxboost.objective import ordinal_logit, ordinal_probit, qwk_ordinal
        from jaxboost.objective import squared_cdf_ordinal

    def test_import_metrics(self):
        """Import metrics."""
        from jaxboost.metric import mse_metric, mae_metric, qwk_metric

    def test_import_survival(self):
        """Import survival objectives."""
        from jaxboost import aft, weibull_aft

    def test_import_multitask(self):
        """Import multi-task objectives."""
        from jaxboost import multi_task_regression, multi_task_classification
        from jaxboost import multi_task_huber, multi_task_quantile

    def test_import_multioutput(self):
        """Import multi-output (uncertainty) objectives."""
        from jaxboost import gaussian_nll, laplace_nll


class TestDependencyImports:
    """Test that dependencies are available."""

    def test_import_jax(self):
        """JAX is installed and importable."""
        import jax
        import jax.numpy as jnp

    def test_import_xgboost(self):
        """XGBoost is installed."""
        import xgboost as xgb

    def test_import_lightgbm(self):
        """LightGBM is installed."""
        import lightgbm as lgb

    def test_import_sklearn(self):
        """Scikit-learn is installed."""
        from sklearn.datasets import make_regression
