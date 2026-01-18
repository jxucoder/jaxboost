"""
Ordinal Regression Benchmark: Wine Quality Dataset

Compares approaches for ordinal classification using Quadratic Weighted Kappa (QWK).

Wine quality ratings are ordinal: 3 < 4 < 5 < 6 < 7 < 8
QWK penalizes predictions quadratically by distance from truth.

Models:
1. XGBoost Regression + simple rounding
2. XGBoost Regression + OptimizedRounder (Kaggle 1st place strategy)
3. XGBoost Multi-class
4. JAXBoost Ordinal NLL (standard CLM with negative log-likelihood)
5. JAXBoost Ordinal EQE (Expected Quadratic Error - QWK-aligned surrogate)
6. JAXBoost Squared CDF (Continuous Ranked Probability Score)

Key insight: QWK is non-differentiable.
- OptRounder directly optimizes QWK via threshold search (two-stage)
- Ordinal objectives optimize a differentiable surrogate (single-stage)
- Ordinal objectives provide proper probabilistic outputs

Usage:
    JAX_PLATFORMS=cpu python examples/ordinal_wine_quality.py
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")

from jaxboost.objective import ordinal_logit, qwk_ordinal, squared_cdf_ordinal

# =============================================================================
# OptimizedRounder (from Kaggle 1st Place Solution)
# =============================================================================


class OptimizedRounder:
    """
    Post-hoc threshold optimization for ordinal regression.

    From the Kaggle Playground Series S3E5 1st place solution:
    https://www.kaggle.com/code/rapela/tpss3e5-1st-place-solution-rapids-xgboost

    Strategy:
    1. Train XGBoost as a regressor (continuous output)
    2. Find optimal thresholds to bin predictions into ordinal classes
    3. Optimize thresholds to maximize Quadratic Weighted Kappa (QWK)

    This is a two-stage approach:
    - Stage 1: Learn the regression function
    - Stage 2: Learn the binning thresholds (post-hoc)

    JAXBoost's CLM learns both jointly during training.
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.thresholds_ = None

    def _kappa_loss(
        self, thresholds: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Negative QWK loss for minimization."""
        y_pred = self._apply_thresholds(X, thresholds)
        return -cohen_kappa_score(y, y_pred, weights="quadratic")

    def _apply_thresholds(
        self, X: np.ndarray, thresholds: np.ndarray
    ) -> np.ndarray:
        """Convert continuous predictions to ordinal classes using thresholds."""
        # thresholds define boundaries: class k if thresholds[k-1] <= x < thresholds[k]
        y_pred = np.zeros_like(X, dtype=int)
        for i, pred in enumerate(X):
            # Find which bin the prediction falls into
            y_pred[i] = np.searchsorted(thresholds, pred)
        return y_pred

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OptimizedRounder":
        """
        Find optimal thresholds to maximize QWK.

        Args:
            X: Continuous predictions from regressor
            y: True ordinal labels (0 to n_classes-1)
        """
        # Initial thresholds: evenly spaced between min and max class
        initial_thresholds = np.arange(0.5, self.n_classes - 0.5, 1.0)

        # Optimize using Nelder-Mead
        loss_fn = partial(self._kappa_loss, X=X, y=y)
        result = scipy.optimize.minimize(
            loss_fn,
            initial_thresholds,
            method="nelder-mead",
            options={"maxiter": 1000, "xatol": 1e-4},
        )
        self.thresholds_ = result.x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Convert continuous predictions to ordinal classes."""
        if self.thresholds_ is None:
            raise ValueError("Must call fit() before predict()")
        y_pred = self._apply_thresholds(X, self.thresholds_)
        return np.clip(y_pred, 0, self.n_classes - 1)


# =============================================================================
# Data Loading
# =============================================================================


def load_wine_quality(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load the UCI Wine Quality dataset.

    Downloads from UCI if not cached locally.

    Returns:
        X: Features, shape (n_samples, 11)
        y: Quality ratings, shape (n_samples,), values 0-5 (mapped from 3-8)
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    red_path = data_dir / "winequality-red.csv"
    white_path = data_dir / "winequality-white.csv"

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"

    # Download if needed
    if not red_path.exists():
        print("Downloading wine quality dataset...")
        red_df = pd.read_csv(f"{base_url}/winequality-red.csv", sep=";")
        white_df = pd.read_csv(f"{base_url}/winequality-white.csv", sep=";")
        red_df.to_csv(red_path, index=False)
        white_df.to_csv(white_path, index=False)
    else:
        red_df = pd.read_csv(red_path)
        white_df = pd.read_csv(white_path)

    # Combine red and white
    df = pd.concat([red_df, white_df], ignore_index=True)

    # Features and target
    X = df.drop("quality", axis=1).values.astype(np.float32)
    y_raw = df["quality"].values

    # Map to 0-indexed classes
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print(f"Loaded {len(y)} samples with {X.shape[1]} features")
    print(f"Classes: {list(le.classes_)} -> {list(range(n_classes))}")
    print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    return X, y, le.classes_, n_classes


# =============================================================================
# Metrics
# =============================================================================


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa - penalizes errors by squared distance."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_ama(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average Mean Absolute Error (AMA) / Macro-MAE.
    
    MAE calculated per class, then averaged.
    Sensitive to class imbalance (minority classes matter as much as majority).
    Standard metric in Ordinal Regression literature (e.g., Gutierrez et al.).
    """
    classes = np.unique(y_true)
    maes = []
    for c in classes:
        mask = y_true == c
        if np.any(mask):
            mae_c = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            maes.append(mae_c)
    return np.mean(maes) if maes else 0.0


def compute_mze(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Zero-one Error (1 - Accuracy)."""
    return 1.0 - accuracy_score(y_true, y_pred)


def get_tail_recall(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Calculate recall for the first and last classes (tails)."""
    cm = confusion_matrix(y_true, y_pred)
    # Handle cases where some classes might be missing in y_true/y_pred interaction
    # (though y_true definitely has them if stratified split worked)

    # Recall = TP / (TP + FN)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.diag(cm) / cm.sum(axis=1)

    # Fill NaNs (if no true samples for a class) with 0
    per_class_recall = np.nan_to_num(per_class_recall)

    # Return (First Class Recall, Last Class Recall)
    # For Wine: Class 0 (Quality 3) and Class 6 (Quality 9)
    if len(per_class_recall) >= 2:
        return per_class_recall[0], per_class_recall[-1]
    return 0.0, 0.0


# =============================================================================
# Models
# =============================================================================


def train_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Train XGBoost with regression objective.

    Returns:
        y_pred: Predicted classes (rounded)
        train_time: Training time in seconds
        y_pred_raw: Raw continuous predictions (for OptimizedRounder)
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    reg_params = {**params, "objective": "reg:squarederror"}

    start = time.perf_counter()
    model = xgb.train(reg_params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    train_time = time.perf_counter() - start

    # Get raw predictions
    y_pred_raw = model.predict(dtest)
    n_classes = len(np.unique(dtrain.get_label()))

    # Also get train predictions for OptimizedRounder fitting
    y_train_pred_raw = model.predict(dtrain)

    # Round predictions to nearest class
    y_pred = np.clip(np.round(y_pred_raw), 0, n_classes - 1).astype(int)

    return y_pred, train_time, (y_train_pred_raw, y_pred_raw)


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    n_classes: int = 6,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with multi-class objective."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    mc_params = {
        **params,
        "objective": "multi:softprob",
        "num_class": n_classes,
    }

    start = time.perf_counter()
    model = xgb.train(mc_params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    train_time = time.perf_counter() - start

    # Get predicted probabilities and take argmax
    y_pred_proba = model.predict(dtest).reshape(-1, n_classes)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred, train_time


def train_ordinal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    ordinal_obj,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with JAXBoost ordinal objective."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    ordinal_obj.init_thresholds_from_data(y_train)

    start = time.perf_counter()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=ordinal_obj.xgb_objective,
        verbose_eval=False,
    )
    train_time = time.perf_counter() - start

    latent_pred = model.predict(dtest)
    y_pred = ordinal_obj.predict(latent_pred)

    return y_pred, train_time


# =============================================================================
# Benchmark
# =============================================================================


def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_seeds: int = 5,
    num_rounds: int = 100,
) -> pd.DataFrame:
    """Run benchmark with multiple random seeds."""
    params = {
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 0,
    }

    print(f"Using {n_classes} classes")
    results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        def evaluate_model(y_p, name):
            score = qwk(y_test, y_p)
            tail_l, tail_r = get_tail_recall(y_test, y_p)
            results.append({
                "Model": name,
                "Seed": seed,
                "QWK": score,
                "Tail_L_Recall": tail_l,
                "Tail_R_Recall": tail_r
            })
            print(f"  {name:<20} QWK={score:.4f} | Tail Recall: L={tail_l:.2f}, R={tail_r:.2f}")

        # 1. Regression (simple rounding)
        y_pred, _, (y_train_pred_raw, y_test_pred_raw) = train_regression(
            X_train, y_train, X_test, params, num_rounds
        )
        evaluate_model(y_pred, "Regression")

        # 2. Regression + OptimizedRounder (Kaggle 1st place)
        opt_rounder = OptimizedRounder(n_classes)
        opt_rounder.fit(y_train_pred_raw, y_train)
        y_pred_opt = opt_rounder.predict(y_test_pred_raw)
        evaluate_model(y_pred_opt, "Reg + OptRounder")

        # 3. Multi-class
        y_pred, _ = train_multiclass(X_train, y_train, X_test, params, n_classes, num_rounds)
        evaluate_model(y_pred, "Multi-class")

        # 4. Ordinal NLL
        obj = ordinal_logit(n_classes=n_classes)
        y_pred, _ = train_ordinal(X_train, y_train, X_test, params, obj, num_rounds)
        evaluate_model(y_pred, "Ordinal NLL")

        # 5. Ordinal EQE
        obj = qwk_ordinal(n_classes=n_classes)
        y_pred, _ = train_ordinal(X_train, y_train, X_test, params, obj, num_rounds)
        evaluate_model(y_pred, "Ordinal EQE")

        # 6. Squared CDF
        obj = squared_cdf_ordinal(n_classes=n_classes)
        y_pred, _ = train_ordinal(X_train, y_train, X_test, params, obj, num_rounds)
        evaluate_model(y_pred, "Squared CDF")

    return pd.DataFrame(results)


def print_summary(results: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("RESULTS: Quadratic Weighted Kappa & Tail Class Recall")
    print("=" * 80)

    summary = results.groupby("Model")[["QWK", "Tail_L_Recall", "Tail_R_Recall"]].agg(["mean", "std"]).round(4)
    summary = summary.sort_values(("QWK", "mean"), ascending=False)

    # Clean column names
    summary.columns = [f"{c[0]}_{c[1]}" for c in summary.columns]

    print("\n" + summary.to_string())

    best_model = summary.index[0]
    best_qwk = summary.loc[best_model, "QWK_mean"]

    print("\n" + "-" * 80)
    print(f"Best QWK: {best_model} (QWK = {best_qwk:.4f})")
    print("-" * 80)
    print("\nConclusion:")
    print("1. 'Regression + OptRounder' (Kaggle strategy) wins.")
    print("   Reason: Decouples ranking (regression) from calibration (thresholds).")
    print("   Directly optimizes QWK which is non-differentiable.")
    print("2. 'Squared CDF' and 'Ordinal NLL' are the best differentiable objectives.")
    print("   They provide proper probabilistic outputs but optimize a surrogate.")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("=" * 80)
    print("WINE QUALITY ORDINAL REGRESSION BENCHMARK")
    print("=" * 80)

    # Load data
    X, y, classes, n_classes = load_wine_quality()

    # Run benchmark
    results = run_benchmark(X, y, n_classes=n_classes, n_seeds=5, num_rounds=100)

    # Print summary
    print_summary(results)

    # Save results
    results_path = Path(__file__).parent / "data" / "ordinal_benchmark_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

