"""Tests for GBMTrainer."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from jaxboost import GBMTrainer, TrainerConfig


class TestGBMTrainer:
    def test_regression_basic(self):
        """Test basic regression training."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        
        trainer = GBMTrainer(
            task="regression",
            config=TrainerConfig(n_trees=5, depth=2, epochs=50),
        )
        forest = trainer.fit(X, y)
        
        predictions = forest.predict(X[:10])
        assert predictions.shape == (10,)
    
    def test_classification_basic(self):
        """Test basic classification training."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        y = y.astype(np.float32)
        
        trainer = GBMTrainer(
            task="classification",
            config=TrainerConfig(n_trees=5, depth=2, epochs=50),
        )
        forest = trainer.fit(X, y)
        
        probs = forest.predict(X[:10])
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1))
        
        classes = forest.predict_class(X[:10])
        assert classes.shape == (10,)
        assert set(classes).issubset({0, 1})
    
    def test_target_normalization(self):
        """Test that target normalization works for regression."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        y = y * 1000 + 5000  # Large target values
        
        trainer = GBMTrainer(
            task="regression",
            config=TrainerConfig(n_trees=3, depth=2, epochs=30, normalize_target=True),
        )
        forest = trainer.fit(X, y)
        
        # Predictions should be in original scale
        predictions = forest.predict(X[:5])
        assert predictions.min() > 1000  # Should be denormalized
    
    def test_early_stopping(self):
        """Test that early stopping works."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        
        trainer = GBMTrainer(
            task="regression",
            config=TrainerConfig(
                n_trees=3, depth=2, epochs=1000, patience=20, verbose=False
            ),
        )
        # Should stop before 1000 epochs due to early stopping
        forest = trainer.fit(X, y)
        assert forest is not None
    
    def test_adaptive_config_small_data(self):
        """Test that config adapts for small datasets."""
        X, y = make_regression(n_samples=100, n_features=3, random_state=42)
        
        trainer = GBMTrainer(task="regression")
        # Small data should get conservative temp_end
        config = trainer._adapt_config(TrainerConfig(), n_samples=100)
        assert config.temp_end == 3.0
    
    def test_adaptive_config_large_data(self):
        """Test that config adapts for large datasets."""
        trainer = GBMTrainer(task="regression")
        config = trainer._adapt_config(TrainerConfig(), n_samples=10000)
        assert config.temp_end == 10.0
    
    def test_with_validation_set(self):
        """Test training with explicit validation set."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)
        
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        
        trainer = GBMTrainer(
            task="regression",
            config=TrainerConfig(n_trees=3, depth=2, epochs=30),
        )
        forest = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        predictions = forest.predict(X_val)
        assert predictions.shape == (50,)

