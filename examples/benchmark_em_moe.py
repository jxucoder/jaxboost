"""EM-MOE Benchmark: When does MOE help?

Analyzes cluster separation, pattern count, sample efficiency, and uncertainty.
"""

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_separable_clusters(n_samples: int, n_clusters: int, separation: float):
    """Generate clusters with controllable separation."""
    np.random.seed(42)
    samples_per_cluster = n_samples // n_clusters
    
    X_list, y_list = [], []
    for c in range(n_clusters):
        center = np.zeros(10)
        center[0] = c * separation  # Control separation
        
        X_c = np.random.randn(samples_per_cluster, 10) * 0.5 + center
        
        # Different function per cluster
        funcs = [
            lambda x: 3 * x[:, 1] + 2 * x[:, 2],
            lambda x: x[:, 1]**2 + x[:, 2]**2,
            lambda x: 5 * np.sin(x[:, 1]) + x[:, 2],
            lambda x: x[:, 1] * x[:, 2] + x[:, 3],
            lambda x: np.exp(0.3 * x[:, 1]) - x[:, 2]**2,
            lambda x: x[:, 1]**3 - 3 * x[:, 1] * x[:, 2]**2,
            lambda x: np.log(np.abs(x[:, 1]) + 1) + x[:, 2],
            lambda x: np.tanh(x[:, 1]) * x[:, 2],
        ]
        y_c = funcs[c % len(funcs)](X_c)
        
        X_list.append(X_c)
        y_list.append(y_c)
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    actual_n = len(y)
    y = y + np.random.randn(actual_n) * 0.2
    
    idx = np.random.permutation(actual_n)
    return X[idx], y[idx]


def benchmark_cluster_separation():
    """How does cluster separation affect MOE advantage?"""
    print("=" * 70)
    print("CLUSTER SEPARATION ANALYSIS")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost required")
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    print(f"\n{'Separation':<12} {'XGBoost MSE':>12} {'EM-MOE MSE':>12} {'Winner':>12} {'MOE Gain':>12}")
    print("-" * 60)
    
    results = []
    for separation in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        X, y = generate_separable_clusters(3000, 4, separation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_mse = mean_squared_error(y_test, xgb_model.predict(X_test))
        
        # EM-MOE
        experts = [
            create_xgboost_expert(task="regression", n_estimators=50, max_depth=5, n_jobs=-1)
            for _ in range(4)
        ]
        config = EMConfig(num_experts=4, em_iterations=10, expert_init_strategy="cluster")
        moe = EMMOE(experts, config=config)
        moe.fit(X_train, y_train, verbose=False)
        moe_mse = mean_squared_error(y_test, moe.predict(X_test))
        
        winner = "EM-MOE" if moe_mse < xgb_mse else "XGBoost"
        gain = (xgb_mse - moe_mse) / xgb_mse * 100
        
        print(f"{separation:<12.1f} {xgb_mse:>12.4f} {moe_mse:>12.4f} {winner:>12} {gain:>11.1f}%")
        results.append((separation, xgb_mse, moe_mse, gain))
    
    print("\n→ MOE wins when clusters are well-separated (gain > 0)")
    return results


def benchmark_num_patterns():
    """How does number of patterns affect MOE?"""
    print("\n" + "=" * 70)
    print("NUMBER OF PATTERNS ANALYSIS")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    print(f"\n{'#Patterns':<12} {'XGBoost MSE':>12} {'EM-MOE MSE':>12} {'Winner':>12}")
    print("-" * 50)
    
    for n_patterns in [2, 4, 6, 8]:
        X, y = generate_separable_clusters(4000, n_patterns, separation=5.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_mse = mean_squared_error(y_test, xgb_model.predict(X_test))
        
        # EM-MOE with matching experts
        experts = [
            create_xgboost_expert(task="regression", n_estimators=50, max_depth=5, n_jobs=-1)
            for _ in range(n_patterns)
        ]
        config = EMConfig(num_experts=n_patterns, em_iterations=10, expert_init_strategy="cluster")
        moe = EMMOE(experts, config=config)
        moe.fit(X_train, y_train, verbose=False)
        moe_mse = mean_squared_error(y_test, moe.predict(X_test))
        
        winner = "EM-MOE" if moe_mse < xgb_mse else "XGBoost"
        print(f"{n_patterns:<12} {xgb_mse:>12.4f} {moe_mse:>12.4f} {winner:>12}")


def benchmark_sample_efficiency():
    """Does MOE help with limited data?"""
    print("\n" + "=" * 70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # Fixed test set
    X_full, y_full = generate_separable_clusters(5000, 4, separation=5.0)
    X_test, y_test = X_full[-1000:], y_full[-1000:]
    
    print(f"\n{'Train Size':<12} {'XGBoost MSE':>12} {'EM-MOE MSE':>12} {'Winner':>12}")
    print("-" * 50)
    
    for train_size in [200, 500, 1000, 2000, 4000]:
        X_train, y_train = X_full[:train_size], y_full[:train_size]
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_mse = mean_squared_error(y_test, xgb_model.predict(X_test))
        
        # EM-MOE
        experts = [
            create_xgboost_expert(task="regression", n_estimators=25, max_depth=4, n_jobs=-1)
            for _ in range(4)
        ]
        config = EMConfig(num_experts=4, em_iterations=8, expert_init_strategy="cluster")
        moe = EMMOE(experts, config=config)
        moe.fit(X_train, y_train, verbose=False)
        moe_mse = mean_squared_error(y_test, moe.predict(X_test))
        
        winner = "EM-MOE" if moe_mse < xgb_mse else "XGBoost"
        print(f"{train_size:<12} {xgb_mse:>12.4f} {moe_mse:>12.4f} {winner:>12}")


def benchmark_uncertainty():
    """Test uncertainty estimation quality."""
    print("\n" + "=" * 70)
    print("UNCERTAINTY ESTIMATION")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # Train on limited range, test with extrapolation
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (800, 5))
    y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + X_train[:, 2]
    y_train += np.random.randn(800) * 0.1
    
    # Test includes in-distribution and out-of-distribution
    X_in = np.random.uniform(-2, 2, (200, 5))  # In-distribution
    X_out = np.random.uniform(3, 5, (200, 5))  # Out-of-distribution
    
    y_in = np.sin(X_in[:, 0]) * np.cos(X_in[:, 1]) + X_in[:, 2]
    y_out = np.sin(X_out[:, 0]) * np.cos(X_out[:, 1]) + X_out[:, 2]
    
    # Train EM-MOE
    experts = [
        create_xgboost_expert(task="regression", n_estimators=50, max_depth=4, n_jobs=-1)
        for _ in range(4)
    ]
    config = EMConfig(num_experts=4, em_iterations=10)
    moe = EMMOE(experts, config=config)
    moe.fit(X_train, y_train, verbose=False)
    
    # Predict with uncertainty
    mean_in, std_in = moe.predict_with_uncertainty(X_in)
    mean_out, std_out = moe.predict_with_uncertainty(X_out)
    
    mse_in = mean_squared_error(y_in, mean_in)
    mse_out = mean_squared_error(y_out, mean_out)
    
    print(f"\nIn-Distribution (x ∈ [-2, 2]):")
    print(f"  MSE: {mse_in:.4f}")
    print(f"  Avg Uncertainty (std): {std_in.mean():.4f}")
    
    print(f"\nOut-of-Distribution (x ∈ [3, 5]):")
    print(f"  MSE: {mse_out:.4f}")
    print(f"  Avg Uncertainty (std): {std_out.mean():.4f}")
    
    print(f"\n→ Uncertainty ratio (OOD/ID): {std_out.mean() / std_in.mean():.2f}x")
    print("  (Higher is better - means model knows when it's uncertain)")


def benchmark_expert_specialization():
    """Visualize expert specialization."""
    print("\n" + "=" * 70)
    print("EXPERT SPECIALIZATION ANALYSIS")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # 4 clear clusters
    X, y = generate_separable_clusters(4000, 4, separation=8.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # True cluster labels based on x[0]
    true_clusters = (X_test[:, 0] / 8.0).round().astype(int).clip(0, 3)
    
    experts = [
        create_xgboost_expert(task="regression", n_estimators=50, max_depth=5, n_jobs=-1)
        for _ in range(4)
    ]
    config = EMConfig(num_experts=4, em_iterations=15, expert_init_strategy="cluster")
    moe = EMMOE(experts, config=config)
    moe.fit(X_train, y_train, verbose=False)
    
    # Get expert weights
    weights = moe.get_expert_weights(X_test)
    dominant_expert = weights.argmax(axis=1)
    
    print("\nExpert assignment vs true cluster:")
    print("-" * 50)
    
    # Confusion matrix
    confusion = np.zeros((4, 4), dtype=int)
    for true_c, pred_e in zip(true_clusters, dominant_expert):
        confusion[true_c, pred_e] += 1
    
    print(f"{'':>8}", end="")
    for e in range(4):
        print(f"Expert{e:>3}", end="")
    print()
    
    for c in range(4):
        print(f"Cluster{c}", end="")
        for e in range(4):
            print(f"{confusion[c, e]:>8}", end="")
        print()
    
    # Specialization score
    total = confusion.sum()
    diagonal = np.diag(confusion).sum()
    specialization = diagonal / total
    
    print(f"\nSpecialization score: {specialization:.1%}")
    print("(100% = perfect expert-to-cluster mapping)")
    
    # Per-expert accuracy
    print("\nPer-cluster expert dominance:")
    for c in range(4):
        dominant = confusion[c].argmax()
        pct = confusion[c, dominant] / confusion[c].sum() * 100
        print(f"  Cluster {c}: Expert {dominant} handles {pct:.0f}%")


def benchmark_init_strategies():
    """Compare expert initialization strategies."""
    print("\n" + "=" * 70)
    print("INITIALIZATION STRATEGY COMPARISON")
    print("=" * 70)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    X, y = generate_separable_clusters(3000, 4, separation=5.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    strategies = ["cluster", "random", "bootstrap", "uniform"]
    
    print(f"\n{'Strategy':<12} {'MSE':>10} {'R²':>10} {'Time':>10}")
    print("-" * 45)
    
    for strategy in strategies:
        experts = [
            create_xgboost_expert(task="regression", n_estimators=50, max_depth=5, n_jobs=-1)
            for _ in range(4)
        ]
        config = EMConfig(
            num_experts=4,
            em_iterations=10,
            expert_init_strategy=strategy,
        )
        moe = EMMOE(experts, config=config)
        
        t0 = time.time()
        moe.fit(X_train, y_train, verbose=False)
        train_time = time.time() - t0
        
        y_pred = moe.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{strategy:<12} {mse:>10.4f} {r2:>10.4f} {train_time:>10.2f}s")


def final_summary():
    """Print final recommendations."""
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     WHEN TO USE EM-MOE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ USE EM-MOE WHEN:                                                │
│     • Data has natural clusters/subpopulations                      │
│     • Different regions need different model types                  │
│     • You want interpretable expert specialization                  │
│     • Uncertainty estimation is important                           │
│     • Data is heterogeneous (mixed patterns)                        │
│                                                                     │
│  ❌ PREFER SINGLE MODEL WHEN:                                       │
│     • Data is homogeneous (single pattern)                          │
│     • Training time is critical                                     │
│     • You have very limited data (< 500 samples per cluster)        │
│     • Patterns overlap significantly                                │
│                                                                     │
│  📊 CONFIGURATION TIPS:                                             │
│     • Set num_experts ≈ number of natural clusters                  │
│     • Use expert_init_strategy="cluster" for best results           │
│     • 5-10 EM iterations is usually sufficient                      │
│     • Expert trees × num_experts ≈ single model trees               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    print("EM-MOE Benchmark\n")
    
    benchmark_cluster_separation()
    benchmark_num_patterns()
    benchmark_sample_efficiency()
    benchmark_uncertainty()
    benchmark_expert_specialization()
    benchmark_init_strategies()
    final_summary()

