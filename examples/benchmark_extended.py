"""
Benchmark: jaxboost vs XGBoost
==============================

High-quality datasets for fair comparison:
- Real-world datasets (not synthetic)
- Commonly used in ML benchmarks
- Clean data (minimal preprocessing)
- 5-fold cross-validation
- Same hyperparameters for both models
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xgboost as xgb
from sklearn.datasets import (
    load_breast_cancer,
    load_wine,
    load_digits,
    load_diabetes,
    fetch_california_housing,
    fetch_openml,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from jaxboost import GBMTrainer, TrainerConfig


# =============================================================================
# SKLEARN WRAPPERS
# =============================================================================

class JaxboostClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper for jaxboost classification."""
    
    def __init__(self, n_trees=20, depth=4, epochs=300):
        self.n_trees = n_trees
        self.depth = depth
        self.epochs = epochs
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        trainer = GBMTrainer(
            task="classification",
            config=TrainerConfig(
                n_trees=self.n_trees,
                depth=self.depth,
                epochs=self.epochs,
                verbose=False,
            ),
        )
        self.forest_ = trainer.fit(X.astype(np.float32), y.astype(np.float32))
        return self
    
    def predict(self, X):
        return self.forest_.predict_class(X.astype(np.float32))


class JaxboostRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible wrapper for jaxboost regression."""
    
    def __init__(self, n_trees=20, depth=4, epochs=500):
        self.n_trees = n_trees
        self.depth = depth
        self.epochs = epochs
    
    def fit(self, X, y):
        trainer = GBMTrainer(
            task="regression",
            config=TrainerConfig(
                n_trees=self.n_trees,
                depth=self.depth,
                epochs=self.epochs,
                verbose=False,
            ),
        )
        self.forest_ = trainer.fit(X.astype(np.float32), y.astype(np.float32))
        return self
    
    def predict(self, X):
        return self.forest_.predict(X.astype(np.float32))


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_spambase():
    """Spambase: email spam detection."""
    data = fetch_openml(data_id=44, as_frame=False, parser="auto")
    X, y = data.data, data.target
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X.astype(np.float32), y.astype(np.int32)


def load_phoneme():
    """Phoneme: speech recognition."""
    data = fetch_openml(data_id=1489, as_frame=False, parser="auto")
    X, y = data.data, data.target
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X.astype(np.float32), y.astype(np.int32)


def load_ionosphere():
    """Ionosphere: radar signal classification."""
    data = fetch_openml(data_id=59, as_frame=False, parser="auto")
    X, y = data.data, data.target
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X.astype(np.float32), y.astype(np.int32)


def load_sonar():
    """Sonar: rock vs mine classification."""
    data = fetch_openml(data_id=40, as_frame=False, parser="auto")
    X, y = data.data, data.target
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return X.astype(np.float32), y.astype(np.int32)


# =============================================================================
# DATASETS
# =============================================================================

DATASETS = [
    # Classification - sklearn built-in
    ("Breast Cancer", lambda: (load_breast_cancer().data, load_breast_cancer().target), "clf",
     "Tumor classification (569 samples, 30 features)"),
    
    ("Wine", lambda: (load_wine().data, (load_wine().target == 0).astype(int)), "clf",
     "Wine origin (178 samples, 13 features)"),
    
    ("Digits", lambda: (load_digits().data, (load_digits().target == 0).astype(int)), "clf",
     "Handwritten digits (1.8K samples, 64 features)"),
    
    # Classification - OpenML
    ("Spambase", load_spambase, "clf",
     "Spam email detection (4.6K samples, 57 features)"),
    
    ("Phoneme", load_phoneme, "clf",
     "Speech phoneme (5.4K samples, 5 features)"),
    
    ("Ionosphere", load_ionosphere, "clf",
     "Radar returns (351 samples, 34 features)"),
    
    ("Sonar", load_sonar, "clf",
     "Rock vs mine (208 samples, 60 features)"),
    
    # Regression - sklearn built-in
    ("California Housing", lambda: (fetch_california_housing().data, fetch_california_housing().target), "reg",
     "House prices (20.6K samples, 8 features)"),
    
    ("Diabetes", lambda: (load_diabetes().data, load_diabetes().target), "reg",
     "Disease progression (442 samples, 10 features)"),
]


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_dataset(name, X, y, task, n_folds=5):
    """Run 5-fold CV benchmark on a single dataset."""
    # Limit large datasets
    if len(X) > 10000:
        idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
        X, y = X[idx], y[idx]
    
    # Same hyperparameters
    N_TREES, DEPTH = 20, 4
    
    if task == "clf":
        xgb_model = xgb.XGBClassifier(
            n_estimators=N_TREES, max_depth=DEPTH, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        jax_model = JaxboostClassifier(n_trees=N_TREES, depth=DEPTH)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = "accuracy"
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=N_TREES, max_depth=DEPTH, learning_rate=0.1,
            random_state=42, verbosity=0
        )
        jax_model = JaxboostRegressor(n_trees=N_TREES, depth=DEPTH)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scoring = "r2"
    
    # Pipelines
    xgb_pipe = Pipeline([("scaler", StandardScaler()), ("model", xgb_model)])
    jax_pipe = Pipeline([("scaler", StandardScaler()), ("model", jax_model)])
    
    # Cross-validation
    xgb_scores = cross_val_score(xgb_pipe, X, y, cv=cv, scoring=scoring)
    jax_scores = cross_val_score(jax_pipe, X, y, cv=cv, scoring=scoring)
    
    return {
        "name": name,
        "task": task,
        "n": len(X),
        "p": X.shape[1],
        "xgb_mean": xgb_scores.mean(),
        "xgb_std": xgb_scores.std(),
        "jax_mean": jax_scores.mean(),
        "jax_std": jax_scores.std(),
    }


def main():
    print("=" * 70)
    print(" jaxboost vs XGBoost Benchmark")
    print(" (5-fold CV, same hyperparameters: n_trees=20, depth=4)")
    print("=" * 70)
    
    results = []
    
    for i, (name, loader, task, desc) in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] {name}")
        print(f"    {desc}")
        
        try:
            X, y = loader()
            r = benchmark_dataset(name, X, y, task)
            results.append(r)
            
            diff = r["jax_mean"] - r["xgb_mean"]
            marker = "*" if diff > 0.005 else ""
            
            print(f"    XGBoost:  {r['xgb_mean']:.4f} ± {r['xgb_std']:.3f}")
            print(f"    jaxboost: {r['jax_mean']:.4f} ± {r['jax_std']:.3f} {marker}")
            
        except Exception as e:
            print(f"    FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    print("\n┌──────────────────┬──────┬─────────────────┬─────────────────┬────────┐")
    print("│ Dataset          │ Task │ XGBoost         │ jaxboost        │ Winner │")
    print("├──────────────────┼──────┼─────────────────┼─────────────────┼────────┤")
    
    jax_wins, xgb_wins, ties = 0, 0, 0
    
    for r in results:
        diff = r["jax_mean"] - r["xgb_mean"]
        
        if diff > 0.005:
            winner, jax_wins = "jax", jax_wins + 1
        elif diff < -0.005:
            winner, xgb_wins = "xgb", xgb_wins + 1
        else:
            winner, ties = "tie", ties + 1
        
        print(f"│ {r['name']:16s} │ {r['task']:4s} │ "
              f"{r['xgb_mean']:.4f} ± {r['xgb_std']:.2f} │ "
              f"{r['jax_mean']:.4f} ± {r['jax_std']:.2f} │ {winner:6s} │")
    
    print("└──────────────────┴──────┴─────────────────┴─────────────────┴────────┘")
    
    total = len(results)
    win_rate = (jax_wins + ties * 0.5) / total * 100
    
    print(f"""
┌───────────────────────────────┐
│ FINAL SCORE                   │
├───────────────────────────────┤
│ jaxboost wins:  {jax_wins:2d} / {total}         │
│ XGBoost wins:   {xgb_wins:2d} / {total}         │
│ Ties:           {ties:2d} / {total}         │
├───────────────────────────────┤
│ Win rate:       {win_rate:5.1f}%        │
└───────────────────────────────┘
""")
    
    if jax_wins > xgb_wins:
        print("jaxboost wins overall")
    elif xgb_wins > jax_wins:
        print("XGBoost wins overall")
    else:
        print("Tie")


if __name__ == "__main__":
    main()
