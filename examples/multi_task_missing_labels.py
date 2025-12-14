"""
Multi-task Learning with Missing Labels Example.

This example demonstrates how to use jaxboost's MaskedMultiTaskObjective
to train XGBoost models when some labels are missing.

Key features:
1. Automatic gradient/Hessian computation via JAX
2. Support for arbitrary label missingness patterns
3. Zero gradients for missing labels (no parameter update)

Scenario: Predicting multiple properties for molecules
- Some molecules have all properties measured
- Some molecules have only partial measurements
- Traditional XGBoost multi-output requires all labels present
"""

import numpy as np

# Check if XGBoost is available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

from jaxboost.objective import (
    MaskedMultiTaskObjective,
    masked_multi_task_objective,
    multi_task_regression,
    multi_task_classification,
    multi_task_huber,
)


def generate_multi_task_data_with_missing(
    n_samples: int = 1000,
    n_features: int = 10,
    n_tasks: int = 3,
    missing_rate: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-task data with missing labels.

    Returns:
        X: Features, shape (n_samples, n_features)
        y_true: Labels with NaN for missing, shape (n_samples, n_tasks)
        mask: Boolean mask, True where label is valid, shape (n_samples, n_tasks)
    """
    np.random.seed(seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate labels for each task (correlated with features)
    y_true = np.zeros((n_samples, n_tasks))
    for k in range(n_tasks):
        # Each task uses different feature combinations
        coeffs = np.random.randn(n_features) * (k + 1) / n_tasks
        y_true[:, k] = X @ coeffs + np.random.randn(n_samples) * 0.1

    # Create missing pattern
    mask = np.random.rand(n_samples, n_tasks) > missing_rate

    # Ensure at least one label per sample
    for i in range(n_samples):
        if not mask[i].any():
            mask[i, np.random.randint(n_tasks)] = True

    # Set missing labels to NaN
    y_true_with_nan = y_true.copy()
    y_true_with_nan[~mask] = np.nan

    return X, y_true_with_nan, mask


def example_basic_usage():
    """Basic usage of MaskedMultiTaskObjective."""
    print("=" * 60)
    print("Example 1: Basic Multi-Task Regression with Missing Labels")
    print("=" * 60)

    if not HAS_XGB:
        print("Skipping (XGBoost not installed)")
        return

    # Generate data
    n_tasks = 3
    X, y_true, mask = generate_multi_task_data_with_missing(
        n_samples=500, n_tasks=n_tasks, missing_rate=0.3
    )

    print(f"Data shape: X={X.shape}, y={y_true.shape}")
    print(f"Missing rate: {(~mask).mean():.1%}")
    print(f"Missing per task: {(~mask).mean(axis=0)}")

    # Prepare for XGBoost
    # - Fill NaN with 0 (the value doesn't matter since gradient will be 0)
    # - Flatten labels for multi-output
    y_filled = np.nan_to_num(y_true, nan=0.0)

    # Split train/test
    n_train = 400
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_filled[:n_train], y_true[n_train:]  # Test uses original
    mask_train, mask_test = mask[:n_train], mask[n_train:]

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train.flatten())
    dtest = xgb.DMatrix(X_test)

    # Create objective with mask passed via get_xgb_objective
    obj = multi_task_regression(n_tasks=n_tasks)

    # XGBoost parameters for multi-output
    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "num_target": n_tasks,
        "max_depth": 4,
        "eta": 0.1,
    }

    # Train - pass mask to get_xgb_objective
    print("\nTraining XGBoost with masked multi-task objective...")
    model = xgb.train(
        params, dtrain, num_boost_round=100,
        obj=obj.get_xgb_objective(mask=mask_train)
    )

    # Predict
    y_pred = model.predict(dtest).reshape(-1, n_tasks)

    # Evaluate (only on valid labels)
    for k in range(n_tasks):
        valid_mask = mask_test[:, k]
        if valid_mask.any():
            mse = np.mean((y_pred[valid_mask, k] - y_test[valid_mask, k]) ** 2)
            print(f"Task {k} MSE: {mse:.4f} (n={valid_mask.sum()})")

    print()


def example_custom_loss():
    """Using custom loss function with decorator."""
    print("=" * 60)
    print("Example 2: Custom Multi-Task Loss with Missing Labels")
    print("=" * 60)

    import jax.numpy as jnp

    # Custom loss: asymmetric loss (penalize under-prediction more)
    @masked_multi_task_objective(n_tasks=3, task_weights=[1.0, 2.0, 0.5])
    def asymmetric_mtl_loss(y_pred, y_true):
        """Asymmetric loss: penalize under-prediction 2x more."""
        error = y_true - y_pred
        return jnp.where(error > 0, 2.0 * error**2, error**2)

    print(f"Created: {asymmetric_mtl_loss}")
    print(f"Task weights: [1.0, 2.0, 0.5] - Task 1 weighted 2x")

    if not HAS_XGB:
        print("Skipping training (XGBoost not installed)")
        return

    # Generate data
    X, y_true, mask = generate_multi_task_data_with_missing(
        n_samples=300, n_tasks=3, missing_rate=0.2
    )

    y_filled = np.nan_to_num(y_true, nan=0.0)
    dtrain = xgb.DMatrix(X, label=y_filled.flatten())

    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "num_target": 3,
        "max_depth": 3,
        "eta": 0.1,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=50,
        obj=asymmetric_mtl_loss.get_xgb_objective(mask=mask)
    )
    print("Training completed!")
    print()


def example_classification():
    """Multi-task binary classification with missing labels."""
    print("=" * 60)
    print("Example 3: Multi-Task Classification with Missing Labels")
    print("=" * 60)

    if not HAS_XGB:
        print("Skipping (XGBoost not installed)")
        return

    np.random.seed(42)

    n_samples, n_features, n_tasks = 500, 10, 4
    X = np.random.randn(n_samples, n_features)

    # Binary labels for each task
    y_true = np.zeros((n_samples, n_tasks))
    for k in range(n_tasks):
        coeffs = np.random.randn(n_features)
        logits = X @ coeffs
        y_true[:, k] = (logits > 0).astype(float)

    # Create missing pattern
    mask = np.random.rand(n_samples, n_tasks) > 0.25
    for i in range(n_samples):
        if not mask[i].any():
            mask[i, np.random.randint(n_tasks)] = True

    y_filled = y_true.copy()
    y_filled[~mask] = 0.0

    print(f"Data: {n_samples} samples, {n_tasks} binary classification tasks")
    print(f"Missing rate: {(~mask).mean():.1%}")

    # Create objective
    obj = multi_task_classification(n_tasks=n_tasks)

    # Prepare data
    dtrain = xgb.DMatrix(X, label=y_filled.flatten())

    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "num_target": n_tasks,
        "max_depth": 4,
        "eta": 0.1,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=100,
        obj=obj.get_xgb_objective(mask=mask)
    )

    # Evaluate
    y_pred_logits = model.predict(xgb.DMatrix(X)).reshape(-1, n_tasks)
    y_pred_class = (y_pred_logits > 0).astype(float)

    for k in range(n_tasks):
        valid = mask[:, k]
        acc = (y_pred_class[valid, k] == y_true[valid, k]).mean()
        print(f"Task {k} Accuracy: {acc:.1%} (n={valid.sum()})")

    print()


def example_quantile_regression():
    """Multi-task quantile regression for prediction intervals."""
    print("=" * 60)
    print("Example 4: Quantile Multi-Task for Prediction Intervals")
    print("=" * 60)

    from jaxboost.objective import multi_task_quantile

    # Predict 10th, 50th, 90th percentiles
    obj = multi_task_quantile(n_tasks=3, quantiles=[0.1, 0.5, 0.9])
    print(f"Created: {obj}")
    print("Predicting quantiles: [0.1, 0.5, 0.9] for prediction intervals")

    if not HAS_XGB:
        print("Skipping training (XGBoost not installed)")
        return

    np.random.seed(42)

    # Generate heteroscedastic data (variance depends on x)
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    noise_scale = 0.5 + np.abs(X[:, 0])  # Heteroscedastic
    y_true = X[:, 0] + X[:, 1] ** 2 + np.random.randn(n_samples) * noise_scale

    # Replicate y for each quantile (all same target)
    y_multi = np.column_stack([y_true, y_true, y_true])

    # Create some missing (for demo)
    mask = np.ones_like(y_multi, dtype=bool)
    mask[np.random.rand(n_samples) < 0.1, 0] = False  # 10% missing for q=0.1

    y_filled = y_multi.copy()
    y_filled[~mask] = 0.0

    dtrain = xgb.DMatrix(X, label=y_filled.flatten())

    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "num_target": 3,
        "max_depth": 4,
        "eta": 0.1,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=100,
        obj=obj.get_xgb_objective(mask=mask)
    )

    # Check calibration
    y_pred = model.predict(xgb.DMatrix(X)).reshape(-1, 3)
    q10, q50, q90 = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    coverage_90 = ((y_true >= q10) & (y_true <= q90)).mean()
    print(f"\n90% Prediction Interval Coverage: {coverage_90:.1%} (target: 80%)")
    print(f"Median prediction MSE: {np.mean((q50 - y_true)**2):.4f}")
    print()


def example_compare_with_without_mask():
    """Compare results with and without proper masking."""
    print("=" * 60)
    print("Example 5: Impact of Proper Masking vs Naive Approach")
    print("=" * 60)

    if not HAS_XGB:
        print("Skipping (XGBoost not installed)")
        return

    np.random.seed(42)

    # Generate data with high missing rate
    X, y_true, mask = generate_multi_task_data_with_missing(
        n_samples=500, n_tasks=3, missing_rate=0.5  # 50% missing!
    )

    # Split
    n_train = 400
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_true[:n_train], y_true[n_train:]
    mask_train, mask_test = mask[:n_train], mask[n_train:]

    params = {
        "tree_method": "hist",
        "multi_strategy": "multi_output_tree",
        "num_target": 3,
        "max_depth": 4,
        "eta": 0.1,
    }

    print(f"Missing rate: {(~mask).mean():.0%}")

    # Approach 1: Naive - fill missing with 0 (wrong!)
    print("\n1. Naive approach (fill missing with 0, no mask):")
    y_naive = np.nan_to_num(y_train, nan=0.0)
    dtrain_naive = xgb.DMatrix(X_train, label=y_naive.flatten())

    # Use default MSE objective
    params_naive = {**params, "objective": "reg:squarederror"}
    model_naive = xgb.train(params_naive, dtrain_naive, num_boost_round=100)

    y_pred_naive = model_naive.predict(xgb.DMatrix(X_test)).reshape(-1, 3)

    # Approach 2: Proper masking
    print("2. Proper masking (jaxboost MaskedMultiTaskObjective):")
    y_filled = np.nan_to_num(y_train, nan=0.0)
    dtrain_masked = xgb.DMatrix(X_train, label=y_filled.flatten())

    obj = multi_task_regression(n_tasks=3)
    model_masked = xgb.train(
        params, dtrain_masked, num_boost_round=100,
        obj=obj.get_xgb_objective(mask=mask_train)
    )

    y_pred_masked = model_masked.predict(xgb.DMatrix(X_test)).reshape(-1, 3)

    # Compare on valid test labels only
    print("\nTest MSE per task (lower is better):")
    print(f"{'Task':<6} {'Naive':<12} {'Masked':<12} {'Improvement':<12}")
    print("-" * 42)

    for k in range(3):
        valid = mask_test[:, k]
        if valid.any():
            mse_naive = np.mean((y_pred_naive[valid, k] - y_test[valid, k]) ** 2)
            mse_masked = np.mean((y_pred_masked[valid, k] - y_test[valid, k]) ** 2)
            improvement = (mse_naive - mse_masked) / mse_naive * 100
            print(f"{k:<6} {mse_naive:<12.4f} {mse_masked:<12.4f} {improvement:>+.1f}%")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multi-Task Learning with Missing Labels - jaxboost Demo")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_custom_loss()
    example_classification()
    example_quantile_regression()
    example_compare_with_without_mask()

    print("All examples completed!")

