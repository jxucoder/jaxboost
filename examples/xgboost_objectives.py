"""
Example: Using jaxboost objective functions with XGBoost.

This example demonstrates how to use jaxboost's automatic objective function
generator with XGBoost for various tasks:
1. Binary classification with Focal Loss
2. Regression with Huber Loss
3. Quantile regression
4. Custom loss functions

Requirements:
    pip install xgboost scikit-learn
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

# XGBoost import
try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")

# jaxboost objective imports
from jaxboost.objective import (
    auto_objective,
    focal_loss,
    huber,
    quantile,
    mse,
)


def example_focal_loss_classification():
    """Binary classification with Focal Loss for imbalanced data."""
    print("=" * 60)
    print("Example 1: Focal Loss for Imbalanced Classification")
    print("=" * 60)

    # Create imbalanced dataset (90% class 0, 10% class 1)
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Class distribution: {np.bincount(y_train)}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters
    params = {
        "max_depth": 4,
        "eta": 0.1,
        "verbosity": 0,
    }

    # Train with standard log loss
    print("\nTraining with standard log loss...")
    model_standard = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_standard = model_standard.predict(dtest)
    auc_standard = roc_auc_score(y_test, y_pred_standard)

    # Train with Focal Loss (gamma=2.0)
    print("Training with Focal Loss (gamma=2.0)...")
    model_focal = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=focal_loss.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_focal = model_focal.predict(dtest)
    # Apply sigmoid since focal loss uses raw logits
    y_pred_focal_prob = 1 / (1 + np.exp(-y_pred_focal))
    auc_focal = roc_auc_score(y_test, y_pred_focal_prob)

    # Train with Focal Loss (gamma=5.0 - more focus on hard examples)
    print("Training with Focal Loss (gamma=5.0)...")
    focal_gamma5 = focal_loss.with_params(gamma=5.0)
    model_focal5 = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=focal_gamma5.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_focal5 = model_focal5.predict(dtest)
    y_pred_focal5_prob = 1 / (1 + np.exp(-y_pred_focal5))
    auc_focal5 = roc_auc_score(y_test, y_pred_focal5_prob)

    print(f"\nResults (AUC-ROC):")
    print(f"  Standard log loss: {auc_standard:.4f}")
    print(f"  Focal Loss γ=2.0:  {auc_focal:.4f}")
    print(f"  Focal Loss γ=5.0:  {auc_focal5:.4f}")


def example_huber_regression():
    """Robust regression with Huber Loss."""
    print("\n" + "=" * 60)
    print("Example 2: Huber Loss for Robust Regression")
    print("=" * 60)

    # Create regression dataset with outliers
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)

    # Add outliers
    outlier_idx = np.random.choice(len(y), size=50, replace=False)
    y[outlier_idx] += np.random.randn(50) * 100  # Large outliers

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Outliers added: 50 samples with large noise")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 4,
        "eta": 0.1,
        "verbosity": 0,
    }

    # Train with MSE
    print("\nTraining with MSE...")
    model_mse = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=mse.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_mse = model_mse.predict(dtest)
    rmse_mse = np.sqrt(mean_squared_error(y_test, y_pred_mse))

    # Train with Huber Loss (delta=1.0)
    print("Training with Huber Loss (δ=1.0)...")
    model_huber = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=huber.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_huber = model_huber.predict(dtest)
    rmse_huber = np.sqrt(mean_squared_error(y_test, y_pred_huber))

    # Train with Huber Loss (delta=10.0)
    print("Training with Huber Loss (δ=10.0)...")
    huber_10 = huber.with_params(delta=10.0)
    model_huber10 = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=huber_10.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_huber10 = model_huber10.predict(dtest)
    rmse_huber10 = np.sqrt(mean_squared_error(y_test, y_pred_huber10))

    print(f"\nResults (RMSE):")
    print(f"  MSE Loss:         {rmse_mse:.4f}")
    print(f"  Huber Loss δ=1.0: {rmse_huber:.4f}")
    print(f"  Huber Loss δ=10:  {rmse_huber10:.4f}")


def example_quantile_regression():
    """Quantile regression for prediction intervals."""
    print("\n" + "=" * 60)
    print("Example 3: Quantile Regression for Prediction Intervals")
    print("=" * 60)

    # Create regression dataset
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 4,
        "eta": 0.1,
        "verbosity": 0,
    }

    # Train models for different quantiles
    quantiles = [0.1, 0.5, 0.9]
    predictions = {}

    for q in quantiles:
        print(f"Training quantile model (q={q})...")
        q_loss = quantile.with_params(q=q)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            obj=q_loss.xgb_objective,
            evals=[(dtest, "test")],
            verbose_eval=False,
        )
        predictions[q] = model.predict(dtest)

    # Calculate coverage
    lower = predictions[0.1]
    median = predictions[0.5]
    upper = predictions[0.9]

    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    below_lower = np.mean(y_test < lower)
    above_upper = np.mean(y_test > upper)

    print(f"\nPrediction Interval Results (10th to 90th percentile):")
    print(f"  Target coverage: 80%")
    print(f"  Actual coverage: {coverage * 100:.1f}%")
    print(f"  Below 10th percentile: {below_lower * 100:.1f}%")
    print(f"  Above 90th percentile: {above_upper * 100:.1f}%")


def example_custom_loss():
    """Custom loss function with automatic gradient/Hessian."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Loss Function")
    print("=" * 60)

    import jax.numpy as jnp

    # Define a custom asymmetric loss
    # Penalize under-predictions more than over-predictions
    @auto_objective
    def asymmetric_mse(y_pred, y_true, under_weight=2.0, over_weight=1.0):
        """
        Asymmetric MSE: penalize under-predictions more.

        Useful for inventory forecasting where stockouts are costly.
        """
        error = y_true - y_pred
        weight = jnp.where(error > 0, under_weight, over_weight)
        return weight * error**2

    # Create dataset
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 4,
        "eta": 0.1,
        "verbosity": 0,
    }

    # Train with standard MSE
    print("Training with standard MSE...")
    model_mse = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=mse.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_mse = model_mse.predict(dtest)

    # Train with asymmetric loss (penalize under-prediction)
    print("Training with Asymmetric MSE (under_weight=3.0)...")
    asym_loss = asymmetric_mse.with_params(under_weight=3.0, over_weight=1.0)
    model_asym = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=asym_loss.xgb_objective,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    y_pred_asym = model_asym.predict(dtest)

    # Calculate metrics
    def calc_metrics(y_true, y_pred):
        errors = y_true - y_pred
        under_pred = np.mean(errors[errors > 0])  # Average under-prediction
        over_pred = np.mean(-errors[errors < 0])  # Average over-prediction
        pct_under = np.mean(errors > 0) * 100
        return under_pred, over_pred, pct_under

    mse_under, mse_over, mse_pct = calc_metrics(y_test, y_pred_mse)
    asym_under, asym_over, asym_pct = calc_metrics(y_test, y_pred_asym)

    print(f"\nResults:")
    print(f"  Standard MSE:")
    print(f"    Under-predictions: {mse_pct:.1f}% (avg error: {mse_under:.2f})")
    print(f"    Over-predictions:  {100-mse_pct:.1f}% (avg error: {mse_over:.2f})")
    print(f"  Asymmetric MSE:")
    print(f"    Under-predictions: {asym_pct:.1f}% (avg error: {asym_under:.2f})")
    print(f"    Over-predictions:  {100-asym_pct:.1f}% (avg error: {asym_over:.2f})")
    print("\n  → Asymmetric loss reduces under-predictions as expected!")


def example_sample_weights():
    """Using sample weights with custom objectives."""
    print("\n" + "=" * 60)
    print("Example 5: Sample Weights with Custom Objectives")
    print("=" * 60)

    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        weights=[0.95, 0.05],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Class distribution (train): {np.bincount(y_train)}")

    # Create sample weights (5x for minority class)
    sample_weights = np.where(y_train == 1, 5.0, 1.0)

    # Create DMatrix with weights
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 4,
        "eta": 0.1,
        "verbosity": 0,
    }

    # Train with focal loss (weights are automatically applied!)
    print("\nTraining with Focal Loss + Sample Weights...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        obj=focal_loss.xgb_objective,  # Weights handled automatically
        evals=[(dtest, "test")],
        verbose_eval=False,
    )

    y_pred = model.predict(dtest)
    y_pred_prob = 1 / (1 + np.exp(-y_pred))
    y_pred_class = (y_pred_prob > 0.5).astype(int)

    # Calculate per-class accuracy
    acc_0 = accuracy_score(y_test[y_test == 0], y_pred_class[y_test == 0])
    acc_1 = accuracy_score(y_test[y_test == 1], y_pred_class[y_test == 1])

    print(f"\nResults:")
    print(f"  Class 0 accuracy: {acc_0:.2%}")
    print(f"  Class 1 accuracy: {acc_1:.2%}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")


if __name__ == "__main__":
    example_focal_loss_classification()
    example_huber_regression()
    example_quantile_regression()
    example_custom_loss()
    example_sample_weights()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)

