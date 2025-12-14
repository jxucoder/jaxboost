"""Mixture of Experts (MOE) Demo.

This example demonstrates using MOE with GBDT experts for regression
and classification tasks.

Key features demonstrated:
1. Different gating types (Linear, MLP, Tree)
2. Sparse top-k routing
3. Expert weight analysis
"""

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from jaxboost.ensemble import (
    MOEEnsemble,
    LinearGating,
    MLPGating,
    TreeGating,
)


def demo_regression():
    """Demo MOE for regression task."""
    print("=" * 60)
    print("MOE Regression Demo")
    print("=" * 60)
    
    # Generate synthetic data with multiple patterns
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create data with 4 distinct patterns (for 4 experts to specialize)
    X = np.random.randn(n_samples, n_features)
    
    # Pattern based on first 2 features
    pattern_idx = (X[:, 0] > 0).astype(int) * 2 + (X[:, 1] > 0).astype(int)
    
    # Different functions for different patterns
    y = np.zeros(n_samples)
    y[pattern_idx == 0] = X[pattern_idx == 0, 2] + X[pattern_idx == 0, 3]
    y[pattern_idx == 1] = X[pattern_idx == 1, 4] * X[pattern_idx == 1, 5]
    y[pattern_idx == 2] = np.sin(X[pattern_idx == 2, 6])
    y[pattern_idx == 3] = X[pattern_idx == 3, 7] ** 2
    
    y += np.random.randn(n_samples) * 0.1  # Add noise
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different gating types
    gating_types = ["linear", "mlp", "tree"]
    
    for gating in gating_types:
        print(f"\n--- Gating: {gating} ---")
        
        moe = MOEEnsemble(
            num_experts=4,
            trees_per_expert=5,
            tree_depth=3,
            gating=gating,
            top_k=2,  # Only use top-2 experts
            task="regression",
        )
        
        # Train
        params = moe.fit(
            X_train, y_train,
            epochs=200,
            learning_rate=0.02,
            verbose=False,
        )
        
        # Predict
        y_pred = moe.predict(params, X_test)
        mse = mean_squared_error(y_test, np.array(y_pred))
        print(f"Test MSE: {mse:.4f}")
        
        # Analyze expert usage
        expert_weights = moe.get_expert_weights(params, X_test)
        avg_weights = np.array(expert_weights).mean(axis=0)
        print(f"Avg expert weights: {avg_weights.round(3)}")


def demo_classification():
    """Demo MOE for classification task."""
    print("\n" + "=" * 60)
    print("MOE Classification Demo")
    print("=" * 60)
    
    # Generate classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Tree gating (most interpretable)
    print("\n--- Tree Gating (4 experts) ---")
    
    moe = MOEEnsemble(
        num_experts=4,
        trees_per_expert=8,
        tree_depth=4,
        gating="tree",
        top_k=None,  # Use all experts
        task="classification",
    )
    
    params = moe.fit(
        X_train, y_train.astype(float),
        epochs=300,
        learning_rate=0.01,
        verbose=False,
    )
    
    # Predict probabilities
    y_prob = moe.predict(params, X_test)
    y_pred = (np.array(y_prob) > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Expert specialization analysis
    expert_weights = moe.get_expert_weights(params, X_test)
    expert_preds = moe.get_expert_predictions(params, X_test)
    
    print("\nExpert Analysis:")
    print(f"  Expert weight std (higher = more specialized): {np.array(expert_weights).std(axis=1).mean():.4f}")
    
    # Show dominant expert per sample
    dominant_expert = np.array(expert_weights).argmax(axis=1)
    for k in range(4):
        count = (dominant_expert == k).sum()
        print(f"  Expert {k} dominant for {count} samples ({100*count/len(y_test):.1f}%)")


def demo_sparse_routing():
    """Demo sparse top-k routing efficiency."""
    print("\n" + "=" * 60)
    print("Sparse Routing Demo")
    print("=" * 60)
    
    # Simple regression data
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = X[:, 0] + X[:, 1] ** 2 + np.random.randn(500) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Compare sparse vs dense routing
    for top_k in [None, 2, 1]:
        label = "All" if top_k is None else f"Top-{top_k}"
        
        moe = MOEEnsemble(
            num_experts=4,
            trees_per_expert=5,
            tree_depth=3,
            gating="linear",
            top_k=top_k,
            task="regression",
        )
        
        params = moe.fit(
            X_train, y_train,
            epochs=150,
            verbose=False,
        )
        
        y_pred = moe.predict(params, X_test)
        mse = mean_squared_error(y_test, np.array(y_pred))
        
        # Check sparsity
        weights = np.array(moe.get_expert_weights(params, X_test))
        sparsity = (weights < 0.01).mean()
        
        print(f"{label:6s}: MSE={mse:.4f}, Sparsity={sparsity:.2%}")


def demo_gating_comparison():
    """Compare different gating mechanisms."""
    print("\n" + "=" * 60)
    print("Gating Comparison")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X, y = make_regression(
        n_samples=800,
        n_features=15,
        n_informative=10,
        noise=10,
        random_state=42,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    results = {}
    
    for gating_name, gating in [
        ("Linear", LinearGating()),
        ("MLP-16", MLPGating(hidden_dim=16)),
        ("MLP-32", MLPGating(hidden_dim=32)),
        ("Tree-d2", TreeGating(depth=2)),  # 4 experts
    ]:
        print(f"\nTraining with {gating_name} gating...")
        
        moe = MOEEnsemble(
            num_experts=4,
            trees_per_expert=8,
            tree_depth=4,
            gating=gating,
            task="regression",
        )
        
        params = moe.fit(
            X_train, y_train_norm,
            epochs=200,
            verbose=False,
        )
        
        y_pred = moe.predict(params, X_test)
        y_pred_denorm = np.array(y_pred) * y_std + y_mean
        mse = mean_squared_error(y_test, y_pred_denorm)
        results[gating_name] = mse
    
    print("\n--- Results ---")
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:10s}: MSE = {mse:.2f}")


if __name__ == "__main__":
    print("JaxBoost MOE Demo")
    print("=" * 60)
    
    demo_regression()
    demo_classification()
    demo_sparse_routing()
    demo_gating_comparison()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)

