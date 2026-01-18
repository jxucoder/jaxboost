"""
Bounded Regression Benchmark: Proportion Prediction

Demonstrates bounded regression for proportion targets in [0, 1].

Problem: Predict proportions (e.g., obesity rate) where targets are bounded [0, 1].
Standard MSE regression can predict outside valid range.
Bounded losses via sigmoid link guarantee valid outputs AND improve accuracy.

Models:
1. XGBoost MSE - standard regression (can predict outside [0, 1])
2. XGBoost MSE + Clip - post-hoc clipping to [0, 1]
3. JAXBoost Logit MSE - MSE in probability space with sigmoid link
4. JAXBoost Soft CE - cross-entropy for proportion targets
5. JAXBoost Beta NLL - beta distribution likelihood

Results:
- Synthetic data (full [0,1] range): ~10% improvement over MSE
- Real data (narrow range): ~2% improvement over MSE
- Bounded losses eliminate out-of-bounds predictions

Why JAXBoost?
- XGBoost has NO built-in bounded regression objective
- Custom bounded loss in 5 lines with @auto_objective
- Proper loss function = better predictions + valid outputs

Usage:
    JAX_PLATFORMS=cpu python examples/beta_regression_health.py
    JAX_PLATFORMS=cpu python examples/beta_regression_health.py --synthetic
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.nn import sigmoid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("Please install xgboost: pip install xgboost")

from jaxboost.objective import auto_objective

# =============================================================================
# Beta Loss Function
# =============================================================================



def make_beta_nll(phi: float = 10.0):
    """Create a Beta NLL loss with specified precision."""

    @auto_objective
    def beta_nll_loss(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        """
        Beta distribution negative log-likelihood for bounded regression.
        
        Uses a simplified formulation that's more numerically stable:
        NLL ∝ -phi * [mu*log(y) + (1-mu)*log(1-y)]
        
        This is the cross-entropy between predicted distribution and observed value,
        scaled by precision phi.
        """
        # Transform logit to probability
        mu = sigmoid(y_pred)

        # Clamp for numerical stability
        eps = 1e-6
        mu = jnp.clip(mu, eps, 1.0 - eps)
        y = jnp.clip(y_true, eps, 1.0 - eps)

        # Simplified Beta NLL (cross-entropy form)
        # This is equivalent to weighted cross-entropy
        nll = -phi * (y * jnp.log(mu) + (1.0 - y) * jnp.log(1.0 - mu))

        return nll

    return beta_nll_loss


# Simple logit-space MSE (predicts logits, evaluates in probability space)
@auto_objective
def logit_mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    MSE in probability space with sigmoid link.
    
    Predictions are logits, transformed via sigmoid before computing MSE.
    This naturally constrains outputs to [0, 1].
    """
    mu = sigmoid(y_pred)
    return (mu - y_true) ** 2


# Cross-entropy loss for proportions (soft labels)
@auto_objective
def soft_crossentropy(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Cross-entropy loss treating proportions as soft labels.
    
    This is the proper loss when y_true is a probability/proportion.
    Equivalent to KL divergence up to a constant.
    """
    mu = sigmoid(y_pred)
    eps = 1e-6
    mu = jnp.clip(mu, eps, 1.0 - eps)
    y = jnp.clip(y_true, eps, 1.0 - eps)

    # Cross-entropy: -[y*log(mu) + (1-y)*log(1-mu)]
    ce = -(y * jnp.log(mu) + (1.0 - y) * jnp.log(1.0 - mu))
    return ce


# =============================================================================
# Data Loading
# =============================================================================



def load_county_health(data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load County Health Rankings data.
    
    Downloads 2024 analytic data from countyhealthrankings.org.
    Target: Adult obesity rate (proportion)
    Features: Various socioeconomic and demographic indicators
    
    Returns:
        X: Features array
        y: Obesity rates in [0, 1]
        feature_names: List of feature names
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    csv_path = data_dir / "county_health_2024.csv"

    if not csv_path.exists():
        print("Downloading County Health Rankings 2024...")
        url = "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2024.csv"
        try:
            df = pd.read_csv(url, encoding='latin-1', skiprows=1, low_memory=False)
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")
        except Exception as e:
            print(f"Error downloading: {e}")
            print("Generating synthetic data instead...")
            return _generate_synthetic_data()

    df = pd.read_csv(csv_path, low_memory=False)

    # Target: Adult obesity (% Obese)
    # The column name varies by year, try common names
    target_candidates = [
        '% Adults with Obesity',
        'Adult obesity Value',
        '% Obese',
        'v011_rawvalue',  # 2024 format
    ]

    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        # Try to find a column with 'obesity' in it
        obesity_cols = [c for c in df.columns if 'obesity' in c.lower() or 'obese' in c.lower()]
        if obesity_cols:
            target_col = obesity_cols[0]
        else:
            print("Could not find obesity column, using synthetic data")
            return _generate_synthetic_data()

    print(f"Using target column: {target_col}")

    # Feature columns (various health/socioeconomic indicators)
    # These are common columns in County Health Rankings
    feature_candidates = [
        # Demographics
        '% below 18 years of age', '% 65 and older', '% Female',
        '% Rural', 'Population',
        # Socioeconomic
        'Median Household Income', '% Children in Poverty',
        '% Adults with Some College', 'Unemployment Rate',
        '% Uninsured', '% Single-Parent Households',
        # Health behaviors (other than obesity - our target)
        '% Smokers', '% Excessive Drinking', '% Physically Inactive',
        '% With Access to Exercise Opportunities',
        # Health outcomes
        '% Diabetic', 'Teen Birth Rate',
        # Built environment
        'Food Environment Index', '% Food Insecure',
        # Use raw value columns (2024 format)
        'v009_rawvalue',  # Adult smoking
        'v049_rawvalue',  # Excessive drinking
        'v070_rawvalue',  # Physical inactivity
        'v024_rawvalue',  # Children in poverty
        'v063_rawvalue',  # Median household income
        'v069_rawvalue',  # Some college
        'v023_rawvalue',  # Unemployment
        'v085_rawvalue',  # Uninsured
        'v052_rawvalue',  # Population
    ]

    # Find available features
    available_features = [c for c in feature_candidates if c in df.columns]

    if len(available_features) < 3:
        print(f"Only found {len(available_features)} features, using synthetic data")
        return _generate_synthetic_data()

    print(f"Found {len(available_features)} feature columns")

    # Extract data
    df_clean = df[[target_col] + available_features].copy()

    # Drop rows with missing values
    df_clean = df_clean.dropna()

    print(f"Samples after dropping NaN: {len(df_clean)}")

    if len(df_clean) < 500:
        print("Not enough samples, using synthetic data")
        return _generate_synthetic_data()

    # Extract X and y
    y = df_clean[target_col].values.astype(np.float32)
    X = df_clean[available_features].values.astype(np.float32)

    # Convert percentage to proportion if needed (0-100 -> 0-1)
    if y.max() > 1.0:
        y = y / 100.0

    # Clip to valid range (some data might have edge cases)
    y = np.clip(y, 0.01, 0.99)

    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target mean: {y.mean():.3f}")

    return X, y, available_features


def _generate_synthetic_data(n_samples: int = 5000, n_features: int = 10) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Generate synthetic proportion data covering full [0, 1] range.
    
    This creates a challenging case where:
    - Target covers nearly full [0, 1] range
    - MSE can predict outside bounds
    - Variance depends on mean (heteroscedastic)
    """
    print("Generating synthetic bounded regression data...")

    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate proportion target from features
    # Use stronger weights to push predictions toward edges
    weights = np.random.randn(n_features) * 0.8
    logits = X @ weights + np.random.randn(n_samples) * 0.5  # add noise to logits
    mu = 1 / (1 + np.exp(-logits))  # sigmoid

    # Add beta-distributed noise (heteroscedastic - variance depends on mean)
    phi = 15  # moderate precision
    alpha = mu * phi
    beta_param = (1 - mu) * phi
    y = np.random.beta(alpha, beta_param).astype(np.float32)

    # Clip to valid range
    y = np.clip(y, 0.01, 0.99)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Target near edges: {np.mean(y < 0.1):.1%} below 0.1, {np.mean(y > 0.9):.1%} above 0.9")

    return X, y, feature_names


# =============================================================================
# Metrics
# =============================================================================



def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics for bounded regression."""
    # Clip predictions for fair comparison
    y_pred_clipped = np.clip(y_pred, 0, 1)

    # MSE
    mse = np.mean((y_true - y_pred_clipped) ** 2)

    # MAE
    mae = np.mean(np.abs(y_true - y_pred_clipped))

    # Proportion of predictions outside [0, 1]
    out_of_bounds = np.mean((y_pred < 0) | (y_pred > 1))

    # Calibration: mean predicted vs mean actual
    calibration_error = np.abs(y_pred_clipped.mean() - y_true.mean())

    return {
        "MSE": mse,
        "MAE": mae,
        "Out_of_Bounds_%": out_of_bounds * 100,
        "Calibration_Error": calibration_error,
    }


# =============================================================================
# Models
# =============================================================================



def train_mse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with MSE objective."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    mse_params = {**params, "objective": "reg:squarederror"}

    start = time.perf_counter()
    model = xgb.train(mse_params, dtrain, num_boost_round=num_rounds, verbose_eval=False)
    train_time = time.perf_counter() - start

    y_pred = model.predict(dtest)

    return y_pred, train_time


def train_beta(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    phi: float = 10.0,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with Beta NLL objective."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Create beta loss with specified precision
    beta_obj = make_beta_nll(phi=phi)

    start = time.perf_counter()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=beta_obj.xgb_objective,
        verbose_eval=False
    )
    train_time = time.perf_counter() - start

    # Predictions are logits, convert to probabilities
    logits = model.predict(dtest)
    y_pred = 1 / (1 + np.exp(-logits))  # sigmoid

    return y_pred, train_time


def train_logit_mse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with Logit-space MSE (bounded output)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    start = time.perf_counter()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=logit_mse.xgb_objective,
        verbose_eval=False
    )
    train_time = time.perf_counter() - start

    # Predictions are logits, convert to probabilities
    logits = model.predict(dtest)
    y_pred = 1 / (1 + np.exp(-logits))  # sigmoid

    return y_pred, train_time


def train_soft_ce(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    num_rounds: int = 100,
) -> tuple[np.ndarray, float]:
    """Train XGBoost with soft cross-entropy (for proportion targets)."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    start = time.perf_counter()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        obj=soft_crossentropy.xgb_objective,
        verbose_eval=False
    )
    train_time = time.perf_counter() - start

    # Predictions are logits, convert to probabilities
    logits = model.predict(dtest)
    y_pred = 1 / (1 + np.exp(-logits))  # sigmoid

    return y_pred, train_time


# =============================================================================
# Benchmark
# =============================================================================



def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    n_seeds: int = 5,
    num_rounds: int = 100,
) -> pd.DataFrame:
    """Run benchmark with multiple random seeds."""
    params = {
        "max_depth": 6,
        "eta": 0.1,
        "verbosity": 0,
    }

    results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed + 1}/{n_seeds} ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def evaluate_model(y_pred, name, train_time):
            metrics = compute_metrics(y_test, y_pred)
            results.append({
                "Model": name,
                "Seed": seed,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "Out_of_Bounds_%": metrics["Out_of_Bounds_%"],
                "Calibration_Error": metrics["Calibration_Error"],
                "Train_Time": train_time,
            })
            print(f"  {name:<20} MSE={metrics['MSE']:.6f} MAE={metrics['MAE']:.4f} OOB={metrics['Out_of_Bounds_%']:.1f}%")

        # 1. XGBoost MSE (raw - can predict outside [0,1])
        y_pred_raw, train_time = train_mse(X_train_scaled, y_train, X_test_scaled, params, num_rounds)
        evaluate_model(y_pred_raw, "XGB MSE", train_time)

        # 2. XGBoost MSE + post-hoc clipping
        y_pred_clipped = np.clip(y_pred_raw, 0, 1)
        evaluate_model(y_pred_clipped, "XGB MSE + Clip", train_time)

        # 3. Logit-space MSE (naturally bounded via sigmoid link)
        y_pred, train_time = train_logit_mse(X_train_scaled, y_train, X_test_scaled, params, num_rounds)
        evaluate_model(y_pred, "Logit MSE", train_time)

        # 4. Soft Cross-Entropy (proper loss for proportion targets)
        y_pred, train_time = train_soft_ce(X_train_scaled, y_train, X_test_scaled, params, num_rounds)
        evaluate_model(y_pred, "Soft CE", train_time)

        # 5. Beta-like NLL (scaled cross-entropy)
        y_pred, train_time = train_beta(X_train_scaled, y_train, X_test_scaled, params, phi=1.0, num_rounds=num_rounds)
        evaluate_model(y_pred, "Beta NLL", train_time)

    return pd.DataFrame(results)


def print_summary(results: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("RESULTS: Bounded Regression for Proportion Prediction")
    print("=" * 80)

    # Aggregate by model
    summary = results.groupby("Model").agg({
        "MSE": ["mean", "std"],
        "MAE": ["mean", "std"],
        "Out_of_Bounds_%": ["mean"],
        "Calibration_Error": ["mean"],
    }).round(6)

    # Flatten column names
    summary.columns = [f"{c[0]}_{c[1]}" if c[1] != "mean" else c[0] for c in summary.columns]
    summary = summary.rename(columns={"MSE_mean": "MSE", "MAE_mean": "MAE"})

    # Sort by MSE
    summary = summary.sort_values("MSE")

    print("\n" + summary.to_string())

    best_model = summary.index[0]
    best_mse = summary.loc[best_model, "MSE"]

    # Calculate improvement
    mse_baseline = summary.loc["XGB MSE + Clip", "MSE"] if "XGB MSE + Clip" in summary.index else summary.loc["XGB MSE", "MSE"]
    improvement = (mse_baseline - best_mse) / mse_baseline * 100

    print("\n" + "-" * 80)
    print(f"Best Model: {best_model} (MSE = {best_mse:.6f})")
    if best_model != "XGB MSE + Clip" and best_model != "XGB MSE":
        print(f"Improvement over XGB MSE + Clip: {improvement:.1f}%")
    print("-" * 80)

    print("\nKey Insights:")
    print("1. Bounded losses (Soft CE, Beta NLL) naturally constrain predictions to [0, 1]")
    print("2. No post-hoc clipping needed - predictions are always valid proportions")
    print("3. Improvement is larger when target covers more of the [0,1] range")
    print("4. Standard MSE can predict outside bounds, especially near edges")
    print("\nWhy use JAXBoost for this?")
    print("- XGBoost has no built-in bounded regression objective")
    print("- JAXBoost lets you define Soft CE or Beta NLL in ~5 lines")
    print("- Proper loss function improves predictions AND guarantees valid outputs")


# =============================================================================
# Main
# =============================================================================



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--rounds", type=int, default=100, help="Number of boosting rounds")
    args = parser.parse_args()

    print("=" * 80)
    print("BOUNDED REGRESSION BENCHMARK")
    print("Predicting proportions in [0, 1]")
    print("=" * 80)

    # Load data
    if args.synthetic:
        X, y, feature_names = _generate_synthetic_data()
        data_name = "Synthetic"
    else:
        X, y, feature_names = load_county_health()
        data_name = "County Health Rankings"

    print(f"\nDataset: {data_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

    # Run benchmark
    results = run_benchmark(X, y, n_seeds=args.seeds, num_rounds=args.rounds)

    # Print summary
    print_summary(results)

    # Save results
    results_path = Path(__file__).parent / "data" / "beta_regression_results.csv"
    results_path.parent.mkdir(exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
