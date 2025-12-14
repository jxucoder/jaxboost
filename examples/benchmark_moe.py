"""Benchmark MOE vs Baseline Models."""

import time
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import make_classification, make_friedman1, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from jaxboost import GBMTrainer, TrainerConfig
from jaxboost.ensemble import MOEEnsemble


@dataclass
class Result:
    name: str
    dataset: str
    metric: float
    train_time: float


def make_multipattern(n=1500):
    np.random.seed(42)
    X = np.random.randn(n, 20)
    p = (X[:, 0] > 0).astype(int) * 2 + (X[:, 1] > 0).astype(int)
    y = np.zeros(n)
    y[p == 0] = 2 * X[p == 0, 2] + X[p == 0, 3]
    y[p == 1] = X[p == 1, 4] ** 2
    y[p == 2] = X[p == 2, 6] * X[p == 2, 7]
    y[p == 3] = np.sin(2 * X[p == 3, 8])
    y += np.random.randn(n) * 0.1
    return X, y


def bench_regression(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_mean, y_std = y_tr.mean(), y_tr.std()
    y_tr_n = (y_tr - y_mean) / y_std
    results = []

    # Baseline
    print("  GBMTrainer...", end=" ", flush=True)
    cfg = TrainerConfig(n_trees=20, depth=4, epochs=300, patience=50, verbose=False)
    t0 = time.time()
    model = GBMTrainer(task="regression", config=cfg).fit(X_tr, y_tr)
    dt = time.time() - t0
    mse = mean_squared_error(y_te, model.predict(X_te))
    results.append(Result("GBMTrainer", name, mse, dt))
    print(f"MSE={mse:.4f} ({dt:.1f}s)")

    # MOE variants
    for mname, gating, topk in [("MOE-Linear", "linear", None), ("MOE-MLP", "mlp", None), 
                                 ("MOE-Tree", "tree", None), ("MOE-TopK", "tree", 2)]:
        print(f"  {mname}...", end=" ", flush=True)
        moe = MOEEnsemble(num_experts=4, trees_per_expert=5, tree_depth=4, gating=gating, top_k=topk, task="regression")
        t0 = time.time()
        params = moe.fit(X_tr, y_tr_n, epochs=300, learning_rate=0.02, patience=50, verbose=False)
        dt = time.time() - t0
        pred = np.array(moe.predict(params, X_te)) * y_std + y_mean
        mse = mean_squared_error(y_te, pred)
        results.append(Result(mname, name, mse, dt))
        print(f"MSE={mse:.4f} ({dt:.1f}s)")
    return results


def bench_classification(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    # Baseline
    print("  GBMTrainer...", end=" ", flush=True)
    cfg = TrainerConfig(n_trees=20, depth=4, epochs=300, patience=50, verbose=False)
    t0 = time.time()
    model = GBMTrainer(task="classification", config=cfg).fit(X_tr, y_tr)
    dt = time.time() - t0
    acc = accuracy_score(y_te, model.predict_class(X_te))
    results.append(Result("GBMTrainer", name, acc, dt))
    print(f"Acc={acc:.4f} ({dt:.1f}s)")

    # MOE variants
    for mname, gating, topk in [("MOE-Linear", "linear", None), ("MOE-MLP", "mlp", None),
                                 ("MOE-Tree", "tree", None), ("MOE-TopK", "tree", 2)]:
        print(f"  {mname}...", end=" ", flush=True)
        moe = MOEEnsemble(num_experts=4, trees_per_expert=5, tree_depth=4, gating=gating, top_k=topk, task="classification")
        t0 = time.time()
        params = moe.fit(X_tr, y_tr.astype(float), epochs=300, learning_rate=0.02, patience=50, verbose=False)
        dt = time.time() - t0
        pred = (np.array(moe.predict(params, X_te)) > 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        results.append(Result(mname, name, acc, dt))
        print(f"Acc={acc:.4f} ({dt:.1f}s)")
    return results


def main():
    print("=" * 60)
    print("MOE Benchmark")
    print("=" * 60)
    all_results = []

    print("\n[1] Multi-Pattern Regression")
    X, y = make_multipattern()
    all_results.extend(bench_regression(X, y, "multipattern"))

    print("\n[2] Friedman #1")
    X, y = make_friedman1(n_samples=1500, n_features=20, noise=1.0, random_state=42)
    all_results.extend(bench_regression(X, y, "friedman1"))

    print("\n[3] Classification")
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=10, n_clusters_per_class=3, random_state=42)
    all_results.extend(bench_classification(X, y, "classification"))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds in sorted(set(r.dataset for r in all_results)):
        print(f"\n{ds}:")
        for r in sorted([x for x in all_results if x.dataset == ds], key=lambda x: x.metric, reverse=(ds == "classification")):
            print(f"  {r.name:<15} {r.metric:.4f}  ({r.train_time:.1f}s)")

    # Expert analysis
    print("\n" + "=" * 60)
    print("Expert Specialization Analysis")
    print("=" * 60)
    X, y = make_multipattern()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    y_tr_n = (y_tr - y_tr.mean()) / y_tr.std()
    moe = MOEEnsemble(num_experts=4, trees_per_expert=5, tree_depth=4, gating="tree")
    params = moe.fit(X_tr, y_tr_n, epochs=300, verbose=False)
    pattern = (X_te[:, 0] > 0).astype(int) * 2 + (X_te[:, 1] > 0).astype(int)
    weights = np.array(moe.get_expert_weights(params, X_te))
    print("\nExpert weights by pattern:")
    for p in range(4):
        w = weights[pattern == p].mean(axis=0)
        print(f"  Pattern {p}: [{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}] -> Expert {np.argmax(w)}")


if __name__ == "__main__":
    main()
