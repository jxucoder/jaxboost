"""
Benchmark: jaxboost objective functions vs manual implementations.

Compares:
1. Speed: jaxboost auto-generated vs hand-coded gradients
2. Accuracy: Gradient/Hessian correctness via numerical differentiation
3. Training time: Full XGBoost training comparison

Requirements:
    pip install xgboost scikit-learn
"""

import time
from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")

from jaxboost.objective import (
    auto_objective,
    focal_loss,
    huber,
    mse,
    quantile,
)


def timer(func):
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        return result, end - start

    return wrapper


# =============================================================================
# Manual Implementations for Comparison
# =============================================================================


def manual_mse_grad_hess(y_pred, y_true):
    """Hand-coded MSE gradient and Hessian."""
    grad = 2 * (y_pred - y_true)
    hess = np.full_like(y_pred, 2.0)
    return grad, hess


def manual_huber_grad_hess(y_pred, y_true, delta=1.0):
    """Hand-coded Huber gradient and Hessian."""
    error = y_pred - y_true
    abs_error = np.abs(error)

    grad = np.where(abs_error <= delta, 2 * error, 2 * delta * np.sign(error))
    hess = np.where(abs_error <= delta, 2.0, 0.0)

    return grad, hess


def manual_focal_grad_hess(y_pred, y_true, gamma=2.0, alpha=0.25):
    """Hand-coded Focal Loss gradient and Hessian (approximate)."""
    p = 1 / (1 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)

    # Focal loss terms
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)

    # Gradient (simplified)
    ce_grad = p - y_true
    focal_weight = alpha_t * (1 - p_t) ** gamma

    # Additional terms from focal weight derivative
    focal_grad_term = gamma * alpha_t * (1 - p_t) ** (gamma - 1) * p * (1 - p)
    log_term = np.where(y_true == 1, -np.log(p + 1e-10), -np.log(1 - p + 1e-10))

    grad = focal_weight * ce_grad - focal_grad_term * log_term * np.sign(y_true - 0.5)

    # Hessian (approximate - use constant for stability)
    hess = np.maximum(2 * focal_weight * p * (1 - p), 1e-6)

    return grad.astype(np.float64), hess.astype(np.float64)


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_gradient_speed(n_samples_list=[1000, 10000, 100000, 500000]):
    """Benchmark gradient computation speed."""
    print("=" * 70)
    print("Benchmark 1: Gradient Computation Speed")
    print("=" * 70)
    print(f"{'n_samples':<12} {'jaxboost (ms)':<15} {'manual (ms)':<15} {'speedup':<10}")
    print("-" * 70)

    for n_samples in n_samples_list:
        y_pred = np.random.randn(n_samples).astype(np.float32)
        y_true = np.random.randn(n_samples).astype(np.float32)

        # Warmup for JIT compilation
        _ = mse.grad_hess(y_pred[:100], y_true[:100])

        # Benchmark jaxboost
        n_runs = 10
        jaxboost_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            grad_jax, hess_jax = mse.grad_hess(y_pred, y_true)
            jaxboost_times.append(time.perf_counter() - start)
        jaxboost_time = np.median(jaxboost_times) * 1000

        # Benchmark manual
        manual_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            grad_manual, hess_manual = manual_mse_grad_hess(y_pred, y_true)
            manual_times.append(time.perf_counter() - start)
        manual_time = np.median(manual_times) * 1000

        speedup = manual_time / jaxboost_time if jaxboost_time > 0 else float("inf")

        print(f"{n_samples:<12} {jaxboost_time:<15.3f} {manual_time:<15.3f} {speedup:<10.2f}x")

    print()


def benchmark_gradient_accuracy():
    """Verify gradient accuracy using numerical differentiation."""
    print("=" * 70)
    print("Benchmark 2: Gradient Accuracy (vs Numerical Differentiation)")
    print("=" * 70)

    n_samples = 100
    y_pred = np.random.randn(n_samples).astype(np.float64)
    y_true = np.random.randn(n_samples).astype(np.float64)

    eps = 1e-5

    def numerical_grad(loss_fn, y_pred, y_true, **kwargs):
        """Compute numerical gradient."""
        grad = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            y_pred_plus = y_pred.copy()
            y_pred_minus = y_pred.copy()
            y_pred_plus[i] += eps
            y_pred_minus[i] -= eps

            loss_plus = np.sum(loss_fn(y_pred_plus, y_true, **kwargs))
            loss_minus = np.sum(loss_fn(y_pred_minus, y_true, **kwargs))

            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        return grad

    # Test MSE
    print("\nMSE Loss:")
    grad_jax, _ = mse.grad_hess(y_pred, y_true)

    def mse_loss_np(y_pred, y_true):
        return (y_pred - y_true) ** 2

    grad_numerical = numerical_grad(mse_loss_np, y_pred, y_true)
    max_diff = np.max(np.abs(grad_jax - grad_numerical))
    print(f"  Max gradient difference: {max_diff:.2e}")
    print(f"  Status: {'✓ PASS' if max_diff < 1e-4 else '✗ FAIL'}")

    # Test Huber
    print("\nHuber Loss:")
    grad_jax, _ = huber.grad_hess(y_pred, y_true)

    def huber_loss_np(y_pred, y_true, delta=1.0):
        error = y_pred - y_true
        abs_error = np.abs(error)
        return np.where(abs_error <= delta, 0.5 * error**2, delta * (abs_error - 0.5 * delta))

    grad_numerical = numerical_grad(huber_loss_np, y_pred, y_true)
    max_diff = np.max(np.abs(grad_jax - grad_numerical))
    print(f"  Max gradient difference: {max_diff:.2e}")
    print(f"  Status: {'✓ PASS' if max_diff < 1e-4 else '✗ FAIL'}")

    # Test Focal Loss (binary classification)
    print("\nFocal Loss:")
    y_true_binary = (np.random.rand(n_samples) > 0.5).astype(np.float64)
    grad_jax, _ = focal_loss.grad_hess(y_pred, y_true_binary)

    def focal_loss_np(y_pred, y_true, gamma=2.0, alpha=0.25):
        p = 1 / (1 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        ce_loss = -y_true * np.log(p) - (1 - y_true) * np.log(1 - p)
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * (1 - p_t) ** gamma * ce_loss

    grad_numerical = numerical_grad(focal_loss_np, y_pred, y_true_binary)
    max_diff = np.max(np.abs(grad_jax - grad_numerical))
    print(f"  Max gradient difference: {max_diff:.2e}")
    print(f"  Status: {'✓ PASS' if max_diff < 1e-3 else '✗ FAIL'}")

    print()


def benchmark_xgboost_training():
    """Benchmark full XGBoost training with different objectives."""
    print("=" * 70)
    print("Benchmark 3: XGBoost Training Time")
    print("=" * 70)

    # Create dataset
    n_samples = 50000
    X, y = make_regression(n_samples=n_samples, n_features=20, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 0,
    }
    num_rounds = 100

    print(f"\nDataset: {n_samples} samples, 20 features")
    print(f"Training: {num_rounds} boosting rounds")
    print("-" * 70)

    results = []

    # XGBoost built-in MSE
    print("Training with XGBoost built-in MSE...")
    start = time.perf_counter()
    model_builtin = xgb.train(
        {**params, "objective": "reg:squarederror"},
        dtrain,
        num_boost_round=num_rounds,
    )
    time_builtin = time.perf_counter() - start
    pred_builtin = model_builtin.predict(dtest)
    rmse_builtin = np.sqrt(np.mean((pred_builtin - y_test) ** 2))
    results.append(("XGBoost built-in MSE", time_builtin, rmse_builtin))

    # jaxboost MSE
    print("Training with jaxboost MSE...")
    start = time.perf_counter()
    model_jaxboost = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=mse.xgb_objective,
    )
    time_jaxboost = time.perf_counter() - start
    pred_jaxboost = model_jaxboost.predict(dtest)
    rmse_jaxboost = np.sqrt(np.mean((pred_jaxboost - y_test) ** 2))
    results.append(("jaxboost MSE", time_jaxboost, rmse_jaxboost))

    # jaxboost Huber
    print("Training with jaxboost Huber...")
    start = time.perf_counter()
    model_huber = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=huber.xgb_objective,
    )
    time_huber = time.perf_counter() - start
    pred_huber = model_huber.predict(dtest)
    rmse_huber = np.sqrt(np.mean((pred_huber - y_test) ** 2))
    results.append(("jaxboost Huber", time_huber, rmse_huber))

    # Custom loss
    @auto_objective
    def custom_smooth_l1(y_pred, y_true, beta=1.0):
        """Smooth L1 loss."""
        diff = jnp.abs(y_pred - y_true)
        return jnp.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)

    print("Training with custom Smooth L1...")
    start = time.perf_counter()
    model_custom = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=custom_smooth_l1.xgb_objective,
    )
    time_custom = time.perf_counter() - start
    pred_custom = model_custom.predict(dtest)
    rmse_custom = np.sqrt(np.mean((pred_custom - y_test) ** 2))
    results.append(("Custom Smooth L1", time_custom, rmse_custom))

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Objective':<25} {'Time (s)':<12} {'RMSE':<12} {'Overhead':<12}")
    print("-" * 70)

    baseline_time = results[0][1]
    for name, train_time, rmse in results:
        overhead = (train_time / baseline_time - 1) * 100
        overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"
        print(f"{name:<25} {train_time:<12.3f} {rmse:<12.4f} {overhead_str:<12}")

    print()


def benchmark_classification():
    """Benchmark classification objectives."""
    print("=" * 70)
    print("Benchmark 4: Classification Training Time")
    print("=" * 70)

    # Create imbalanced dataset
    n_samples = 50000
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        weights=[0.9, 0.1],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 0,
    }
    num_rounds = 100

    print(f"\nDataset: {n_samples} samples (90/10 imbalanced), 20 features")
    print(f"Training: {num_rounds} boosting rounds")
    print("-" * 70)

    from sklearn.metrics import roc_auc_score

    results = []

    # XGBoost built-in logistic
    print("Training with XGBoost built-in logistic...")
    start = time.perf_counter()
    model_builtin = xgb.train(
        {**params, "objective": "binary:logistic"},
        dtrain,
        num_boost_round=num_rounds,
    )
    time_builtin = time.perf_counter() - start
    pred_builtin = model_builtin.predict(dtest)
    auc_builtin = roc_auc_score(y_test, pred_builtin)
    results.append(("XGBoost logistic", time_builtin, auc_builtin))

    # jaxboost Focal Loss
    print("Training with jaxboost Focal Loss...")
    start = time.perf_counter()
    model_focal = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=focal_loss.xgb_objective,
    )
    time_focal = time.perf_counter() - start
    pred_focal = model_focal.predict(dtest)
    pred_focal_prob = 1 / (1 + np.exp(-pred_focal))
    auc_focal = roc_auc_score(y_test, pred_focal_prob)
    results.append(("jaxboost Focal (γ=2)", time_focal, auc_focal))

    # jaxboost Focal Loss with different gamma
    print("Training with jaxboost Focal Loss (γ=5)...")
    focal_g5 = focal_loss.with_params(gamma=5.0)
    start = time.perf_counter()
    model_focal5 = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=focal_g5.xgb_objective,
    )
    time_focal5 = time.perf_counter() - start
    pred_focal5 = model_focal5.predict(dtest)
    pred_focal5_prob = 1 / (1 + np.exp(-pred_focal5))
    auc_focal5 = roc_auc_score(y_test, pred_focal5_prob)
    results.append(("jaxboost Focal (γ=5)", time_focal5, auc_focal5))

    # Print results
    print("\n" + "-" * 70)
    print(f"{'Objective':<25} {'Time (s)':<12} {'AUC-ROC':<12} {'Overhead':<12}")
    print("-" * 70)

    baseline_time = results[0][1]
    for name, train_time, auc in results:
        overhead = (train_time / baseline_time - 1) * 100
        overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"
        print(f"{name:<25} {train_time:<12.3f} {auc:<12.4f} {overhead_str:<12}")

    print()


def benchmark_jit_warmup():
    """Show JIT compilation overhead."""
    print("=" * 70)
    print("Benchmark 5: JIT Compilation Warmup")
    print("=" * 70)

    n_samples = 10000
    y_pred = np.random.randn(n_samples).astype(np.float32)
    y_true = np.random.randn(n_samples).astype(np.float32)

    print(f"\nDataset: {n_samples} samples")
    print("-" * 70)

    # First call (includes JIT compilation)
    start = time.perf_counter()
    grad1, hess1 = huber.grad_hess(y_pred, y_true)
    first_call = (time.perf_counter() - start) * 1000

    # Subsequent calls (JIT-compiled)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        grad, hess = huber.grad_hess(y_pred, y_true)
        times.append((time.perf_counter() - start) * 1000)

    avg_subsequent = np.mean(times)

    print(f"First call (with JIT compilation): {first_call:.2f} ms")
    print(f"Subsequent calls (JIT-compiled):   {avg_subsequent:.2f} ms")
    print(f"Speedup after warmup:              {first_call / avg_subsequent:.1f}x")
    print()


def benchmark_batch_sizes():
    """Benchmark performance across different batch sizes."""
    print("=" * 70)
    print("Benchmark 6: Performance vs Batch Size")
    print("=" * 70)

    batch_sizes = [100, 1000, 10000, 50000, 100000, 500000]

    print(f"\n{'Batch Size':<12} {'jaxboost (ms)':<15} {'Throughput (M/s)':<18}")
    print("-" * 70)

    # Warmup
    y_pred_warmup = np.random.randn(1000).astype(np.float32)
    y_true_warmup = np.random.randn(1000).astype(np.float32)
    _ = focal_loss.grad_hess(y_pred_warmup, y_true_warmup)

    for batch_size in batch_sizes:
        y_pred = np.random.randn(batch_size).astype(np.float32)
        y_true = (np.random.rand(batch_size) > 0.5).astype(np.float32)

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            grad, hess = focal_loss.grad_hess(y_pred, y_true)
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times) * 1000
        throughput = batch_size / (avg_time / 1000) / 1e6  # Million samples per second

        print(f"{batch_size:<12} {avg_time:<15.3f} {throughput:<18.2f}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("JAXBOOST OBJECTIVE FUNCTION BENCHMARKS")
    print("=" * 70 + "\n")

    benchmark_gradient_speed()
    benchmark_gradient_accuracy()
    benchmark_jit_warmup()
    benchmark_batch_sizes()
    benchmark_xgboost_training()
    benchmark_classification()

    print("=" * 70)
    print("All benchmarks completed!")
    print("=" * 70)

