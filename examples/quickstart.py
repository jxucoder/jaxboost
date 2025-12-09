"""
jaxboost Quickstart
===================

Simple example using the high-level GBMTrainer API.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from jaxboost import GBMTrainer, TrainerConfig


def regression_example():
    """Regression example: California Housing."""
    print("=" * 60)
    print(" Regression: California Housing")
    print("=" * 60)
    
    # Load data
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Normalize features (recommended)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    
    # Train with jaxboost
    trainer = GBMTrainer(
        task="regression",
        config=TrainerConfig(
            n_trees=20,
            depth=4,
            epochs=500,
            verbose=True,
        )
    )
    
    print("\nTraining...")
    model = trainer.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, predictions)
    print(f"\nTest R²: {r2:.4f}")
    
    return r2


def classification_example():
    """Classification example: Breast Cancer."""
    print("\n" + "=" * 60)
    print(" Classification: Breast Cancer")
    print("=" * 60)
    
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(np.float32)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    
    # Train
    trainer = GBMTrainer(
        task="classification",
        config=TrainerConfig(
            n_trees=20,
            depth=4,
            epochs=500,
            verbose=True,
        )
    )
    
    print("\nTraining...")
    model = trainer.fit(X_train, y_train)
    
    # Predict classes
    predictions = model.predict_class(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {acc:.4f}")
    
    return acc


def main():
    print("=" * 60)
    print(" jaxboost Quickstart")
    print("=" * 60)
    
    r2 = regression_example()
    acc = classification_example()
    
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  Regression R²: {r2:.4f}")
    print(f"  Classification Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

