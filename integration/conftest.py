"""Shared fixtures for JAXBoost integration tests."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def regression_data():
    """Regression dataset: 500 samples, 10 features."""
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def binary_data():
    """Binary classification dataset."""
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_data():
    """3-class classification dataset."""
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=3,
        n_clusters_per_class=1, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def ordinal_data():
    """Ordinal regression dataset: 6 ordered classes."""
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y_cont = X[:, 0] * 2 + X[:, 1] + np.random.randn(500) * 0.5
    y = np.clip(np.digitize(y_cont, bins=[-2, -1, 0, 1, 2]), 0, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def xgb_params():
    """Default XGBoost parameters."""
    return {"max_depth": 4, "eta": 0.1, "verbosity": 0}


@pytest.fixture
def lgb_params():
    """Default LightGBM parameters."""
    return {"max_depth": 4, "learning_rate": 0.1, "verbosity": -1, "metric": "None"}
