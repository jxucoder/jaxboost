"""EM-MOE Demo: External GBDT Experts trained via EM algorithm.

This example demonstrates using XGBoost, LightGBM, and CatBoost models
as experts in a Mixture of Experts architecture trained via EM.

The EM algorithm:
    E-step: Compute posterior responsibilities γ_{ik} ∝ g_k(x_i) · p(y_i|x_i, k)
    M-step: Retrain experts with sample_weight = γ, update gating to predict γ

Requirements:
    pip install xgboost lightgbm catboost scikit-learn
"""

import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def demo_em_moe():
    """Demo EM-trained MOE with XGBoost experts."""
    print("=" * 60)
    print("EM-MOE with XGBoost Experts")
    print("=" * 60)
    
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # Generate synthetic data with distinct patterns
    np.random.seed(42)
    n_samples = 2000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # 4 distinct patterns based on input regions
    y = np.zeros(n_samples)
    q1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    q2 = (X[:, 0] <= 0) & (X[:, 1] > 0)
    q3 = (X[:, 0] <= 0) & (X[:, 1] <= 0)
    q4 = (X[:, 0] > 0) & (X[:, 1] <= 0)
    
    y[q1] = 2 * X[q1, 2] + 3 * X[q1, 3]  # Linear
    y[q2] = X[q2, 4] ** 2                 # Quadratic
    y[q3] = X[q3, 5] * X[q3, 6]           # Interaction
    y[q4] = 3 * np.sin(X[q4, 7])          # Sinusoidal
    
    y += np.random.randn(n_samples) * 0.3
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create experts
    experts = [
        create_xgboost_expert(
            task="regression",
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42 + i,
        )
        for i in range(4)
    ]
    
    # EM configuration
    config = EMConfig(
        num_experts=4,
        em_iterations=10,  # Run 10 EM iterations
        gating_hidden_dims=(32, 16),
        gating_epochs_per_iter=100,
        expert_init_strategy="cluster",  # K-means initialization
        noise_variance=0.5,  # Assumed noise level
    )
    
    # Train
    moe = EMMOE(experts, config=config)
    moe.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    y_pred = moe.predict(X_test)
    print(f"\nResults:")
    print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"  R²:  {r2_score(y_test, y_pred):.4f}")
    
    # Compare to single XGBoost
    single_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5)
    single_xgb.fit(X_train, y_train)
    y_pred_single = single_xgb.predict(X_test)
    print(f"\nSingle XGBoost baseline:")
    print(f"  MSE: {mean_squared_error(y_test, y_pred_single):.4f}")
    print(f"  R²:  {r2_score(y_test, y_pred_single):.4f}")
    
    # Analyze expert specialization
    print("\nExpert Specialization Analysis:")
    weights = moe.get_expert_weights(X_test)
    
    test_masks = [q1[len(X_train):], q2[len(X_train):], q3[len(X_train):], q4[len(X_train):]]
    region_names = ["Q1 (linear)", "Q2 (quadratic)", "Q3 (interaction)", "Q4 (sinusoidal)"]
    
    for name, mask in zip(region_names, test_masks):
        if mask.sum() > 0:
            region_weights = weights[mask].mean(axis=0)
            dominant = region_weights.argmax()
            print(f"  {name}: Expert {dominant} dominant ({region_weights[dominant]:.2f})")
    
    return moe


def demo_em_convergence():
    """Demo EM convergence and log-likelihood improvement."""
    print("\n" + "=" * 60)
    print("EM Convergence Analysis")
    print("=" * 60)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # Simple data
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    
    experts = [
        create_xgboost_expert(task="regression", n_estimators=30, max_depth=3)
        for _ in range(4)
    ]
    
    config = EMConfig(
        num_experts=4,
        em_iterations=15,
        expert_init_strategy="cluster",
    )
    
    moe = EMMOE(experts, config=config)
    moe.fit(X, y, verbose=True)
    
    # Plot convergence
    ll_history = moe.history["log_likelihood"]
    
    print(f"\nLog-likelihood improved from {ll_history[0]:.4f} to {ll_history[-1]:.4f}")
    print(f"Total improvement: {ll_history[-1] - ll_history[0]:.4f}")
    
    # Expert usage over iterations
    print("\nExpert usage evolution:")
    for i, usage in enumerate(moe.history["expert_usage"]):
        usage_str = ", ".join(f"{u:.2f}" for u in usage)
        print(f"  Iter {i+1}: [{usage_str}]")


def demo_hard_em():
    """Demo Hard-EM variant (hard assignments)."""
    print("\n" + "=" * 60)
    print("Hard-EM MOE (Crisp Assignments)")
    print("=" * 60)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, HardEMMOE, create_xgboost_expert
    
    # Data with clear clusters
    np.random.seed(42)
    n_per_cluster = 300
    
    # 4 clusters with different patterns
    X_list, y_list = [], []
    for i in range(4):
        center = np.array([i * 3, 0] + [0] * 8)
        X_c = np.random.randn(n_per_cluster, 10) + center
        # Different function per cluster
        if i == 0:
            y_c = X_c[:, 0]
        elif i == 1:
            y_c = X_c[:, 0] ** 2
        elif i == 2:
            y_c = np.sin(X_c[:, 0])
        else:
            y_c = X_c[:, 0] * X_c[:, 1]
        X_list.append(X_c)
        y_list.append(y_c)
    
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32) + np.random.randn(len(X)) * 0.1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Soft EM
    experts_soft = [
        create_xgboost_expert(task="regression", n_estimators=30, max_depth=3)
        for _ in range(4)
    ]
    config = EMConfig(num_experts=4, em_iterations=10)
    soft_moe = EMMOE(experts_soft, config=config)
    soft_moe.fit(X_train, y_train, verbose=False)
    
    # Hard EM  
    experts_hard = [
        create_xgboost_expert(task="regression", n_estimators=30, max_depth=3)
        for _ in range(4)
    ]
    hard_moe = HardEMMOE(experts_hard, config=config)
    hard_moe.fit(X_train, y_train, verbose=False)
    
    print("Results on clustered data:")
    y_pred_soft = soft_moe.predict(X_test)
    y_pred_hard = hard_moe.predict(X_test)
    
    print(f"  Soft EM: MSE={mean_squared_error(y_test, y_pred_soft):.4f}, "
          f"R²={r2_score(y_test, y_pred_soft):.4f}")
    print(f"  Hard EM: MSE={mean_squared_error(y_test, y_pred_hard):.4f}, "
          f"R²={r2_score(y_test, y_pred_hard):.4f}")


def demo_sparse_em():
    """Demo Sparse EM-MOE with top-k routing."""
    print("\n" + "=" * 60)
    print("Sparse EM-MOE (Top-K Routing)")
    print("=" * 60)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, SparseEMMOE, create_xgboost_expert
    
    # Data
    X, y = make_regression(n_samples=1000, n_features=15, noise=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    config = EMConfig(num_experts=8, em_iterations=10)  # More experts
    
    results = {}
    
    # Full routing (all 8 experts)
    experts_full = [
        create_xgboost_expert(task="regression", n_estimators=30, max_depth=3)
        for _ in range(8)
    ]
    moe_full = EMMOE(experts_full, config=config)
    moe_full.fit(X_train, y_train, verbose=False)
    y_pred_full = moe_full.predict(X_test)
    results["All 8"] = mean_squared_error(y_test, y_pred_full)
    
    # Sparse routing (top-k)
    for k in [4, 2, 1]:
        experts_sparse = [
            create_xgboost_expert(task="regression", n_estimators=30, max_depth=3)
            for _ in range(8)
        ]
        moe_sparse = SparseEMMOE(experts_sparse, config=config, top_k=k)
        moe_sparse.fit(X_train, y_train, verbose=False)
        y_pred_sparse = moe_sparse.predict(X_test)
        results[f"Top-{k}"] = mean_squared_error(y_test, y_pred_sparse)
    
    print("Sparse routing comparison:")
    for name, mse in results.items():
        print(f"  {name}: MSE = {mse:.4f}")


def demo_uncertainty():
    """Demo uncertainty estimation from expert disagreement."""
    print("\n" + "=" * 60)
    print("Uncertainty from Expert Disagreement")
    print("=" * 60)
    
    try:
        import xgboost as xgb
    except ImportError:
        return
    
    from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert
    
    # Training data in limited range
    np.random.seed(42)
    X_train = np.random.uniform(-2, 2, (500, 1))
    y_train = np.sin(X_train[:, 0]) + np.random.randn(500) * 0.1
    
    # Test includes extrapolation
    X_test = np.linspace(-4, 4, 200).reshape(-1, 1)
    y_test_true = np.sin(X_test[:, 0])
    
    experts = [
        create_xgboost_expert(task="regression", n_estimators=50, max_depth=3)
        for _ in range(4)
    ]
    config = EMConfig(num_experts=4, em_iterations=10)
    
    moe = EMMOE(experts, config=config)
    moe.fit(X_train, y_train, verbose=False)
    
    # Predict with uncertainty
    mean, std = moe.predict_with_uncertainty(X_test)
    
    print("Uncertainty analysis:")
    in_range = (X_test[:, 0] >= -2) & (X_test[:, 0] <= 2)
    out_range = ~in_range
    
    print(f"  In-distribution (|x| <= 2):")
    print(f"    Avg uncertainty: {std[in_range].mean():.4f}")
    print(f"    MSE: {mean_squared_error(y_test_true[in_range], mean[in_range]):.4f}")
    
    print(f"  Out-of-distribution (|x| > 2):")
    print(f"    Avg uncertainty: {std[out_range].mean():.4f}")
    print(f"    MSE: {mean_squared_error(y_test_true[out_range], mean[out_range]):.4f}")


def demo_mixed_library():
    """Demo with mixed XGBoost/LightGBM/CatBoost experts."""
    print("\n" + "=" * 60)
    print("Mixed Library Experts")
    print("=" * 60)
    
    try:
        import xgboost
        import lightgbm
        from catboost import CatBoostRegressor
    except ImportError as e:
        print(f"Missing library: {e}")
        return
    
    from jaxboost.ensemble import (
        EMMOE, EMConfig,
        create_xgboost_expert,
        create_lightgbm_expert,
        create_catboost_expert,
    )
    
    # California housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Diverse experts
    experts = [
        create_xgboost_expert(task="regression", n_estimators=50, max_depth=4),
        create_xgboost_expert(task="regression", n_estimators=50, max_depth=6),
        create_lightgbm_expert(task="regression", n_estimators=50, num_leaves=31),
        create_catboost_expert(task="regression", iterations=50, depth=4),
    ]
    
    config = EMConfig(
        num_experts=4,
        em_iterations=8,
        expert_init_strategy="cluster",
    )
    
    print("Training EM-MOE with mixed experts...")
    moe = EMMOE(experts, config=config)
    moe.fit(X_train, y_train, verbose=True)
    
    y_pred = moe.predict(X_test)
    print(f"\nEM-MOE: MSE={mean_squared_error(y_test, y_pred):.4f}, "
          f"R²={r2_score(y_test, y_pred):.4f}")
    
    # Expert weights
    weights = moe.get_expert_weights(X_test)
    expert_names = ["XGB-d4", "XGB-d6", "LightGBM", "CatBoost"]
    print("\nExpert usage:")
    for name, w in zip(expert_names, weights.mean(axis=0)):
        print(f"  {name}: {w:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("EM-MOE Demo: External GBDT Experts via EM Algorithm")
    print("=" * 60)
    print()
    
    demo_em_moe()
    demo_em_convergence()
    demo_hard_em()
    demo_sparse_em()
    demo_uncertainty()
    demo_mixed_library()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
