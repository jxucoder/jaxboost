"""
Ordinal Regression Benchmark: SLACE Paper Replication

Implements the benchmark from:
    SLACE: A Monotone and Balance-Sensitive Loss Function for Ordinal Regression
    AAAI 2025 - https://github.com/inbarnachmani/SLACE

Datasets (True Ordinal):
    - ERA: Employee Risk Assessment (9 classes)
    - ESL: English as Second Language (9 classes)
    - LEV: Lecturer Evaluation (5 classes)
    - SWD: Student Weighted Dataset (4 classes)
    - Wine: Wine Quality Red (6 classes)
    - Car: Car Evaluation (4 classes)
    - Balance: Balance Scale (3 classes)
    - TAE: Teaching Assistant Evaluation (3 classes)
    - Eucalyptus: Eucalyptus Soil Conservation (5 classes)
    - Thyroid: New Thyroid (3 classes)

Metrics:
    - CEM: Closeness Evaluation Metric (Amigo et al. 2020)
    - QWK: Quadratic Weighted Kappa
    - MAE: Mean Absolute Error
    - Accuracy: 1 - Mean Zero-one Error

Models:
    - Baseline: XGBoost Regression, Multi-class
    - Paper: SORD, OLL, SLACE
    - Ours: Ordinal NLL, Squared CDF
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from jaxboost.objective import (
    oll_objective,
    ordinal_logit,
    slace_objective,
    sord_objective,
    squared_cdf_ordinal,
)

DATA_DIR = Path(__file__).parent / "data" / "ordinal"


# =============================================================================
# CEM Metric (from SLACE paper / Amigo et al. 2020)
# =============================================================================


def create_prox_mat(dist_dict: dict, inv: bool = False) -> np.ndarray:
    """Create proximity matrix based on class distribution."""
    labels = sorted(dist_dict.keys())
    denominator = sum(dist_dict.values())
    n = len(labels)
    prox_mat = np.zeros([n, n])

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            minlabel, maxlabel = min(label1, label2), max(label1, label2)
            numerator = dist_dict[label1] / 2
            if minlabel == label1:
                for tmp_label in range(minlabel + 1, maxlabel + 1):
                    if tmp_label in dist_dict:
                        numerator += dist_dict[tmp_label]
            else:
                for tmp_label in range(maxlabel - 1, minlabel - 1, -1):
                    if tmp_label in dist_dict:
                        numerator += dist_dict[tmp_label]
            ratio = numerator / denominator
            if inv:
                prox_mat[i][j] = (-np.log(ratio)) ** -1 if ratio > 0 else 0
            else:
                prox_mat[i][j] = -np.log(ratio) if ratio > 0 else 0
    return prox_mat


def cem(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Closeness Evaluation Metric (Amigo et al. 2020).

    Higher is better (1.0 = perfect predictions).
    """
    dist_dict = dict(Counter(y_true))
    labels = sorted(dist_dict.keys())
    label_to_idx = {label: i for i, label in enumerate(labels)}

    prox_mat = create_prox_mat(dist_dict, inv=False)

    pred_prox = sum(
        prox_mat[label_to_idx.get(int(p), 0)][label_to_idx.get(int(t), 0)]
        for p, t in zip(y_pred, y_true)
    )
    truth_prox = sum(
        prox_mat[label_to_idx.get(int(t), 0)][label_to_idx.get(int(t), 0)]
        for t in y_true
    )

    return pred_prox / truth_prox if truth_prox > 0 else 0.0


# =============================================================================
# Dataset Loaders
# =============================================================================


def load_era() -> tuple[np.ndarray, np.ndarray, str]:
    """ERA: Employee Risk Assessment (9 classes)."""
    df = pd.read_csv(DATA_DIR / "ERA.csv")
    X = df[["in1", "in2", "in3", "in4"]].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["out1"].values)
    return X, y, "ERA"


def load_esl() -> tuple[np.ndarray, np.ndarray, str]:
    """ESL: English as Second Language (9 classes)."""
    df = pd.read_csv(DATA_DIR / "ESL.csv")
    X = df[["in1", "in2", "in3", "in4"]].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["out1"].values)
    return X, y, "ESL"


def load_lev() -> tuple[np.ndarray, np.ndarray, str]:
    """LEV: Lecturer Evaluation (5 classes)."""
    df = pd.read_csv(DATA_DIR / "LEV.csv")
    X = df[["In1", "In2", "In3", "In4"]].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["Out1"].values)
    return X, y, "LEV"


def load_swd() -> tuple[np.ndarray, np.ndarray, str]:
    """SWD: Student Weighted Dataset (4 classes)."""
    df = pd.read_csv(DATA_DIR / "SWD.csv")
    feature_cols = [c for c in df.columns if c.startswith("In")]
    X = df[feature_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["Out1"].values)
    return X, y, "SWD"


def load_wine() -> tuple[np.ndarray, np.ndarray, str]:
    """Wine Quality Red (6 classes)."""
    df = pd.read_csv(DATA_DIR / "winequality-red.csv")
    # Make a copy to avoid in-place modification issues
    X = df.drop(columns=["quality"]).values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["quality"].values.copy())
    return X, y, "Wine"


def load_car() -> tuple[np.ndarray, np.ndarray, str]:
    """Car Evaluation (4 classes)."""
    df = pd.read_csv(DATA_DIR / "car-evaluation.csv")
    # Encode categorical features
    for col in df.columns[:-1]:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    return X, y, "Car"


def load_balance() -> tuple[np.ndarray, np.ndarray, str]:
    """Balance Scale (3 classes)."""
    df = pd.read_csv(
        DATA_DIR / "balance-scale.data",
        header=None,
        names=["class", "lw", "ld", "rw", "rd"],
    )
    X = df[["lw", "ld", "rw", "rd"]].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df["class"].values)
    return X, y, "Balance"


def load_tae() -> tuple[np.ndarray, np.ndarray, str]:
    """Teaching Assistant Evaluation (3 classes)."""
    df = pd.read_csv(
        DATA_DIR / "tae.data",
        header=None,
        names=["native", "instructor", "course", "semester", "size", "class"],
    )
    X = df[["native", "instructor", "course", "semester", "size"]].values.astype(
        np.float32
    )
    y = LabelEncoder().fit_transform(df["class"].values)
    return X, y, "TAE"


def load_eucalyptus() -> tuple[np.ndarray, np.ndarray, str]:
    """Eucalyptus Soil Conservation (5 classes)."""
    df = pd.read_csv(DATA_DIR / "dataset_194_eucalyptus.csv")
    # Drop non-numeric columns and target
    target_col = "Utility"
    df = df.dropna(subset=[target_col])
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = LabelEncoder().fit_transform(df[target_col].values)
    return X, y, "Eucalyptus"


def load_thyroid() -> tuple[np.ndarray, np.ndarray, str]:
    """New Thyroid (3 classes)."""
    df = pd.read_csv(DATA_DIR / "new-thyroid.csv")
    target_col = df.columns[0]  # First column is target
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df[target_col].values)
    return X, y, "Thyroid"


ALL_DATASETS = [
    load_era,
    load_esl,
    load_lev,
    load_swd,
    load_wine,
    load_car,
    load_balance,
    load_tae,
    load_eucalyptus,
    load_thyroid,
]


# =============================================================================
# Models
# =============================================================================


def train_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
) -> np.ndarray:
    """XGBoost Regression baseline."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train({**params, "objective": "reg:squarederror"}, dtrain, num_boost_round=100)
    y_pred = np.clip(np.round(model.predict(dtest)), 0, n_classes - 1).astype(int)
    return y_pred


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
) -> np.ndarray:
    """XGBoost Multi-class baseline."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train(
        {**params, "objective": "multi:softprob", "num_class": n_classes},
        dtrain,
        num_boost_round=100,
    )
    probs = model.predict(dtest).reshape(-1, n_classes)
    return np.argmax(probs, axis=1)


def train_sord(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
    alpha: float = 1.0,
) -> np.ndarray:
    """SORD: Soft Ordinal (Diaz & Marathe 2019)."""
    obj = sord_objective(n_classes=n_classes, alpha=alpha)
    model = xgb.XGBClassifier(
        objective=obj.sklearn_objective,
        n_estimators=100,
        **{k: v for k, v in params.items() if k not in ["verbosity"]},
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test)


def train_oll(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
    alpha: float = 1.0,
) -> np.ndarray:
    """OLL: Ordinal Log Loss (Castagnos et al. 2022)."""
    obj = oll_objective(n_classes=n_classes, alpha=alpha)
    model = xgb.XGBClassifier(
        objective=obj.sklearn_objective,
        n_estimators=100,
        **{k: v for k, v in params.items() if k not in ["verbosity"]},
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test)


def train_slace(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
    alpha: float = 1.0,
) -> np.ndarray:
    """SLACE: Soft Labels Accumulating Cross Entropy (AAAI 2025)."""
    obj = slace_objective(n_classes=n_classes, alpha=alpha)
    model = xgb.XGBClassifier(
        objective=obj.sklearn_objective,
        n_estimators=100,
        **{k: v for k, v in params.items() if k not in ["verbosity"]},
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test)


def train_ordinal_nll(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
) -> np.ndarray:
    """JAXBoost Ordinal NLL (Cumulative Link Model)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    obj = ordinal_logit(n_classes=n_classes)
    obj.init_thresholds_from_data(y_train)
    model = xgb.train(params, dtrain, obj=obj.xgb_objective, num_boost_round=100)
    return obj.predict(model.predict(dtest))


def train_squared_cdf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    params: dict,
) -> np.ndarray:
    """JAXBoost Squared CDF (CRPS)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    obj = squared_cdf_ordinal(n_classes=n_classes)
    obj.init_thresholds_from_data(y_train)
    model = xgb.train(params, dtrain, obj=obj.xgb_objective, num_boost_round=100)
    return obj.predict(model.predict(dtest))


# =============================================================================
# Benchmark Runner
# =============================================================================


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    return {
        "CEM": cem(y_true, y_pred),
        "QWK": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Acc": accuracy_score(y_true, y_pred),
    }


def run_dataset_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    n_folds: int = 10,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Run benchmark on a single dataset with k-fold CV."""
    n_classes = len(np.unique(y))
    params = {"max_depth": 10, "eta": 0.1, "verbosity": 0, "colsample_bytree": 0.5}

    results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    models = {
        "Regression": lambda Xtr, ytr, Xte: train_regression(Xtr, ytr, Xte, n_classes, params),
        "MultiClass": lambda Xtr, ytr, Xte: train_multiclass(Xtr, ytr, Xte, n_classes, params),
        "SORD": lambda Xtr, ytr, Xte: train_sord(Xtr, ytr, Xte, n_classes, params, alpha),
        "OLL": lambda Xtr, ytr, Xte: train_oll(Xtr, ytr, Xte, n_classes, params, alpha),
        "SLACE": lambda Xtr, ytr, Xte: train_slace(Xtr, ytr, Xte, n_classes, params, alpha),
        "OrdinalNLL": lambda Xtr, ytr, Xte: train_ordinal_nll(Xtr, ytr, Xte, n_classes, params),
        "SquaredCDF": lambda Xtr, ytr, Xte: train_squared_cdf(Xtr, ytr, Xte, n_classes, params),
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for model_name, train_fn in models.items():
            try:
                y_pred = train_fn(X_train, y_train, X_test)
                metrics = compute_metrics(y_test, y_pred)
                results.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Fold": fold,
                        **metrics,
                    }
                )
            except Exception as e:
                print(f"  Error in {model_name} on {dataset_name} fold {fold}: {e}")
                results.append(
                    {
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Fold": fold,
                        "CEM": np.nan,
                        "QWK": np.nan,
                        "MAE": np.nan,
                        "Acc": np.nan,
                    }
                )

    return pd.DataFrame(results)


def run_full_benchmark(alpha: float = 1.0) -> pd.DataFrame:
    """Run benchmark on all datasets."""
    all_results = []

    for loader in ALL_DATASETS:
        try:
            X, y, name = loader()
            n_classes = len(np.unique(y))
            print(f"\n{'='*60}")
            print(f"Dataset: {name} | Samples: {len(y)} | Classes: {n_classes}")
            print(f"Distribution: {dict(Counter(y))}")
            print(f"{'='*60}")

            df = run_dataset_benchmark(X, y, name, n_folds=10, alpha=alpha)
            all_results.append(df)

            # Print per-dataset summary
            summary = df.groupby("Model")[["CEM", "QWK", "MAE", "Acc"]].mean().round(4)
            print(summary.to_string())
        except Exception as e:
            print(f"\nError loading {loader.__name__}: {e}")

    return pd.concat(all_results, ignore_index=True)


def print_summary_table(results: pd.DataFrame):
    """Print summary table like Table 2 in SLACE paper."""
    print("\n" + "=" * 80)
    print("SLACE BENCHMARK RESULTS - CEM (Higher is Better)")
    print("=" * 80)

    # Pivot: rows = datasets, columns = models
    cem_pivot = results.groupby(["Dataset", "Model"])["CEM"].mean().unstack()
    print("\nCEM by Dataset and Model:")
    print(cem_pivot.round(4).to_string())

    # Average across datasets
    print("\n" + "-" * 80)
    print("Average CEM across all datasets:")
    avg = results.groupby("Model")["CEM"].mean().sort_values(ascending=False)
    print(avg.round(4).to_string())

    print("\n" + "-" * 80)
    print("Average QWK across all datasets:")
    avg_qwk = results.groupby("Model")["QWK"].mean().sort_values(ascending=False)
    print(avg_qwk.round(4).to_string())

    print("\n" + "-" * 80)
    print("Average MAE across all datasets (Lower is Better):")
    avg_mae = results.groupby("Model")["MAE"].mean().sort_values(ascending=True)
    print(avg_mae.round(4).to_string())


if __name__ == "__main__":
    print("SLACE Paper Benchmark Replication")
    print("=" * 60)

    results = run_full_benchmark(alpha=1.0)

    # Save results
    output_path = DATA_DIR / "slace_benchmark_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print_summary_table(results)

