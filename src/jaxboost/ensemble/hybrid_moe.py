"""Hybrid Mixture of Experts with External GBDT Experts.

This module provides MOE ensembles that use external boosting libraries
(XGBoost, CatBoost, LightGBM) as experts, trained via EM algorithm.

EM Algorithm for MOE:
    The MOE model assumes: p(y|x) = Σ_k g_k(x) · p(y|x, expert_k)
    
    E-step: Compute posterior responsibility of expert k for sample i:
        γ_{ik} ∝ g_k(x_i) · p(y_i | x_i, expert_k)
        
    M-step: 
        - Retrain experts on weighted samples using γ
        - Update gating to predict responsibilities

This trades end-to-end differentiability for:
- Much faster expert training (native implementations)
- Better individual expert performance  
- Access to advanced features (GPU, categorical handling, etc.)
"""

from __future__ import annotations

from typing import Any, Literal, Protocol
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


class ExternalGBDTExpert(Protocol):
    """Protocol for external GBDT experts (XGBoost, CatBoost, LightGBM)."""
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Train the expert."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...


@dataclass  
class EMConfig:
    """Configuration for EM-based MOE training.
    
    Attributes:
        num_experts: Number of expert models.
        em_iterations: Number of EM iterations.
        gating_hidden_dims: Hidden layer dimensions for gating MLP.
        gating_epochs_per_iter: Gating training epochs per EM iteration.
        gating_lr: Learning rate for gating network.
        noise_variance: Assumed noise variance for likelihood (regression).
        expert_init_strategy: How to initialize experts before EM.
            - "random": Random partition
            - "cluster": K-means clustering  
            - "bootstrap": Bootstrap sampling
            - "uniform": Train all on full data (no partition)
        min_responsibility: Minimum responsibility threshold for expert training.
        temperature: Softmax temperature for gating.
    """
    num_experts: int = 4
    em_iterations: int = 10
    gating_hidden_dims: tuple[int, ...] = (32, 16)
    gating_epochs_per_iter: int = 100
    gating_lr: float = 0.01
    noise_variance: float = 1.0
    expert_init_strategy: Literal["random", "cluster", "bootstrap", "uniform"] = "cluster"
    min_responsibility: float = 0.01
    temperature: float = 1.0


def create_xgboost_expert(**kwargs) -> ExternalGBDTExpert:
    """Create an XGBoost expert wrapper."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")
    
    class XGBExpert:
        def __init__(self, task: str = "regression", **params):
            self.task = task
            self.params = params
            self.model = None
            
        def fit(self, X, y, sample_weight=None):
            if self.task == "regression":
                self.model = xgb.XGBRegressor(**self.params)
            else:
                self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(X, y, sample_weight=sample_weight)
            
        def predict(self, X):
            if self.model is None:
                raise ValueError("Expert not fitted")
            if self.task == "regression":
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 1]
    
    return XGBExpert(**kwargs)


def create_lightgbm_expert(**kwargs) -> ExternalGBDTExpert:
    """Create a LightGBM expert wrapper."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm")
    
    class LGBMExpert:
        def __init__(self, task: str = "regression", **params):
            self.task = task
            self.params = {**params, "verbosity": -1}
            self.model = None
            
        def fit(self, X, y, sample_weight=None):
            if self.task == "regression":
                self.model = lgb.LGBMRegressor(**self.params)
            else:
                self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X, y, sample_weight=sample_weight)
            
        def predict(self, X):
            if self.model is None:
                raise ValueError("Expert not fitted")
            if self.task == "regression":
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 1]
    
    return LGBMExpert(**kwargs)


def create_catboost_expert(**kwargs) -> ExternalGBDTExpert:
    """Create a CatBoost expert wrapper."""
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
    except ImportError:
        raise ImportError("catboost is required: pip install catboost")
    
    class CatBoostExpert:
        def __init__(self, task: str = "regression", **params):
            self.task = task
            self.params = {**params, "verbose": False}
            self.model = None
            
        def fit(self, X, y, sample_weight=None):
            if self.task == "regression":
                self.model = CatBoostRegressor(**self.params)
            else:
                self.model = CatBoostClassifier(**self.params)
            self.model.fit(X, y, sample_weight=sample_weight)
            
        def predict(self, X):
            if self.model is None:
                raise ValueError("Expert not fitted")
            if self.task == "regression":
                return self.model.predict(X)
            else:
                return self.model.predict_proba(X)[:, 1]
    
    return CatBoostExpert(**kwargs)


class GatingNetwork:
    """MLP gating network for routing inputs to experts."""
    
    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (32, 16),
        temperature: float = 1.0,
    ):
        self.hidden_dims = hidden_dims
        self.temperature = temperature
    
    def init_params(self, key: Array, num_features: int, num_experts: int) -> dict:
        """Initialize gating network parameters."""
        params = {"layers": []}
        dims = [num_features] + list(self.hidden_dims) + [num_experts]
        keys = jax.random.split(key, len(dims) - 1)
        
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            scale = jnp.sqrt(2.0 / in_dim)
            params["layers"].append({
                "W": jax.random.normal(keys[i], (in_dim, out_dim)) * scale,
                "b": jnp.zeros(out_dim),
            })
        return params
    
    def __call__(self, params: dict, x: Array, temperature: float | None = None) -> Array:
        """Forward pass: returns expert weights (batch, num_experts)."""
        temp = temperature if temperature is not None else self.temperature
        h = x
        for i, layer in enumerate(params["layers"]):
            h = h @ layer["W"] + layer["b"]
            if i < len(params["layers"]) - 1:
                h = jax.nn.relu(h)
        return jax.nn.softmax(h / temp, axis=-1)


class EMMOE:
    """EM-trained Mixture of Experts with external GBDT experts.
    
    This implementation uses proper EM algorithm:
    
    **Model**: p(y|x) = Σ_k g_k(x) · p(y|x, expert_k)
    
    **E-step**: Compute responsibilities (posterior over experts):
        γ_{ik} = g_k(x_i) · N(y_i; f_k(x_i), σ²) / Σ_j g_j(x_i) · N(y_i; f_j(x_i), σ²)
        
    **M-step**: 
        - Train expert k on all data with sample_weight = γ_{:,k}
        - Train gating to minimize KL(γ || g(x)) or maximize responsibility prediction
    
    Example:
        >>> experts = [
        ...     create_xgboost_expert(task="regression", n_estimators=100)
        ...     for _ in range(4)
        ... ]
        >>> moe = EMMOE(experts, config=EMConfig(em_iterations=10))
        >>> moe.fit(X_train, y_train)
        >>> y_pred = moe.predict(X_test)
    """
    
    def __init__(
        self,
        experts: list[ExternalGBDTExpert],
        config: EMConfig | None = None,
        task: Literal["regression", "classification"] = "regression",
    ):
        self.experts = experts
        self.config = config or EMConfig(num_experts=len(experts))
        self.task = task
        
        if len(experts) != self.config.num_experts:
            raise ValueError(
                f"Number of experts ({len(experts)}) must match "
                f"config.num_experts ({self.config.num_experts})"
            )
        
        self.gating = GatingNetwork(
            hidden_dims=self.config.gating_hidden_dims,
            temperature=self.config.temperature,
        )
        self.gating_params = None
        self._fitted = False
        self.history = {"log_likelihood": [], "expert_usage": []}
    
    def _init_experts(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize experts before EM iterations."""
        n_samples = X.shape[0]
        K = self.config.num_experts
        strategy = self.config.expert_init_strategy
        
        if strategy == "random":
            indices = np.random.permutation(n_samples)
            splits = np.array_split(indices, K)
            for k, idx in enumerate(splits):
                if len(idx) > 0:
                    self.experts[k].fit(X[idx], y[idx])
                    
        elif strategy == "cluster":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    self.experts[k].fit(X[mask], y[mask])
                    
        elif strategy == "bootstrap":
            for k in range(K):
                rng = np.random.default_rng(42 + k)
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                self.experts[k].fit(X[idx], y[idx])
                
        elif strategy == "uniform":
            # All experts start from same data
            for expert in self.experts:
                expert.fit(X, y)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _get_expert_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all experts: (n_samples, num_experts)."""
        preds = []
        for expert in self.experts:
            try:
                preds.append(expert.predict(X))
            except ValueError:
                preds.append(np.zeros(X.shape[0]))
        return np.stack(preds, axis=-1)
    
    def _compute_responsibilities(
        self,
        X: np.ndarray,
        y: np.ndarray,
        expert_preds: np.ndarray,
        gating_weights: np.ndarray,
    ) -> np.ndarray:
        """E-step: Compute posterior responsibilities γ_{ik}."""
        sigma2 = self.config.noise_variance
        
        if self.task == "regression":
            # Gaussian likelihood: p(y|x,k) ∝ exp(-0.5 * (y - f_k(x))² / σ²)
            errors = y[:, None] - expert_preds  # (n, K)
            log_likelihood = -0.5 * errors**2 / sigma2  # (n, K)
        else:
            # Bernoulli likelihood for classification
            eps = 1e-7
            probs = np.clip(expert_preds, eps, 1 - eps)
            log_likelihood = y[:, None] * np.log(probs) + (1 - y[:, None]) * np.log(1 - probs)
        
        # Log posterior: log(γ_ik) = log(g_k) + log(p(y|x,k)) - log(Σ_j ...)
        log_weights = np.log(gating_weights + 1e-10)
        log_joint = log_weights + log_likelihood  # (n, K)
        
        # Normalize via log-sum-exp for numerical stability
        log_sum = np.logaddexp.reduce(log_joint, axis=-1, keepdims=True)
        log_responsibilities = log_joint - log_sum
        responsibilities = np.exp(log_responsibilities)
        
        return responsibilities
    
    def _m_step_experts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        responsibilities: np.ndarray,
    ) -> None:
        """M-step for experts: Retrain with responsibility-weighted samples."""
        for k in range(self.config.num_experts):
            weights = responsibilities[:, k]
            # Only train if expert has significant total responsibility
            if weights.sum() > self.config.min_responsibility * len(X):
                self.experts[k].fit(X, y, sample_weight=weights)
    
    def _m_step_gating(
        self,
        X: np.ndarray,
        responsibilities: np.ndarray,
        epochs: int,
        learning_rate: float,
    ) -> float:
        """M-step for gating: Train to predict responsibilities.
        
        Minimize cross-entropy between gating outputs and responsibilities:
            L = -Σ_i Σ_k γ_{ik} · log(g_k(x_i))
        """
        X_jax = jnp.array(X, dtype=jnp.float32)
        resp_jax = jnp.array(responsibilities, dtype=jnp.float32)
        
        def loss_fn(params):
            g = self.gating(params, X_jax)
            # Cross-entropy: -Σ γ_k · log(g_k)
            return -jnp.mean(jnp.sum(resp_jax * jnp.log(g + 1e-10), axis=-1))
        
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.gating_params)
        
        @jax.jit
        def train_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        final_loss = 0.0
        for _ in range(epochs):
            self.gating_params, opt_state, final_loss = train_step(
                self.gating_params, opt_state
            )
        
        return float(final_loss)
    
    def _compute_log_likelihood(
        self,
        y: np.ndarray,
        expert_preds: np.ndarray,
        gating_weights: np.ndarray,
    ) -> float:
        """Compute log-likelihood: log p(y|x) = log Σ_k g_k · p(y|x,k)."""
        sigma2 = self.config.noise_variance
        
        if self.task == "regression":
            errors = y[:, None] - expert_preds
            log_likelihood = -0.5 * errors**2 / sigma2 - 0.5 * np.log(2 * np.pi * sigma2)
        else:
            eps = 1e-7
            probs = np.clip(expert_preds, eps, 1 - eps)
            log_likelihood = y[:, None] * np.log(probs) + (1 - y[:, None]) * np.log(1 - probs)
        
        # log p(y|x) = log Σ_k g_k · exp(log_lik_k)
        log_joint = np.log(gating_weights + 1e-10) + log_likelihood
        log_marginal = np.logaddexp.reduce(log_joint, axis=-1)
        
        return float(np.mean(log_marginal))
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> "EMMOE":
        """Train MOE using EM algorithm.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            verbose: Whether to print progress.
            
        Returns:
            Self for method chaining.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Initialize gating network
        key = jax.random.PRNGKey(42)
        self.gating_params = self.gating.init_params(
            key, X.shape[1], self.config.num_experts
        )
        
        # Initialize experts
        if verbose:
            print(f"Initializing {self.config.num_experts} experts "
                  f"(strategy: {self.config.expert_init_strategy})...")
        self._init_experts(X, y)
        
        # EM iterations
        for em_iter in range(self.config.em_iterations):
            # Get current predictions
            expert_preds = self._get_expert_predictions(X)
            X_jax = jnp.array(X, dtype=jnp.float32)
            gating_weights = np.array(self.gating(self.gating_params, X_jax))
            
            # Compute log-likelihood before update
            ll = self._compute_log_likelihood(y, expert_preds, gating_weights)
            self.history["log_likelihood"].append(ll)
            
            # E-step: Compute responsibilities
            responsibilities = self._compute_responsibilities(
                X, y, expert_preds, gating_weights
            )
            
            # Track expert usage
            expert_usage = responsibilities.mean(axis=0)
            self.history["expert_usage"].append(expert_usage.tolist())
            
            if verbose:
                usage_str = ", ".join(f"{u:.2f}" for u in expert_usage)
                print(f"EM iter {em_iter + 1}/{self.config.em_iterations}: "
                      f"LL={ll:.4f}, usage=[{usage_str}]")
            
            # M-step: Update experts
            self._m_step_experts(X, y, responsibilities)
            
            # M-step: Update gating
            self._m_step_gating(
                X, responsibilities,
                epochs=self.config.gating_epochs_per_iter,
                learning_rate=self.config.gating_lr,
            )
        
        # Final log-likelihood
        expert_preds = self._get_expert_predictions(X)
        gating_weights = np.array(self.gating(self.gating_params, jnp.array(X, dtype=jnp.float32)))
        final_ll = self._compute_log_likelihood(y, expert_preds, gating_weights)
        self.history["log_likelihood"].append(final_ll)
        
        if verbose:
            print(f"Final LL: {final_ll:.4f}")
        
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions: weighted sum of expert outputs."""
        if not self._fitted:
            raise ValueError("MOE not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        X_jax = jnp.array(X, dtype=jnp.float32)
        
        expert_preds = self._get_expert_predictions(X)
        gating_weights = np.array(self.gating(self.gating_params, X_jax))
        
        return np.sum(gating_weights * expert_preds, axis=-1)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with epistemic uncertainty from expert disagreement.
        
        Returns:
            mean: Weighted mean predictions
            std: Weighted standard deviation across experts
        """
        X = np.asarray(X, dtype=np.float32)
        X_jax = jnp.array(X, dtype=jnp.float32)
        
        expert_preds = self._get_expert_predictions(X)
        weights = np.array(self.gating(self.gating_params, X_jax))
        
        # Weighted mean
        mean = np.sum(weights * expert_preds, axis=-1)
        
        # Weighted variance: Var = E[f²] - E[f]²
        mean_sq = np.sum(weights * expert_preds**2, axis=-1)
        variance = mean_sq - mean**2
        std = np.sqrt(np.maximum(variance, 0))
        
        return mean, std
    
    def get_responsibilities(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get expert responsibilities for given samples."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        expert_preds = self._get_expert_predictions(X)
        X_jax = jnp.array(X, dtype=jnp.float32)
        gating_weights = np.array(self.gating(self.gating_params, X_jax))
        
        return self._compute_responsibilities(X, y, expert_preds, gating_weights)
    
    def get_expert_weights(self, X: np.ndarray) -> np.ndarray:
        """Get gating weights (prior over experts) for inputs."""
        X_jax = jnp.array(X, dtype=jnp.float32)
        return np.array(self.gating(self.gating_params, X_jax))
    
    def get_expert_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get individual expert predictions."""
        return self._get_expert_predictions(np.asarray(X, dtype=np.float32))


# Backward compatibility aliases
HybridMOE = EMMOE
HybridMOEConfig = EMConfig


class HardEMMOE(EMMOE):
    """Hard-EM variant where each sample is assigned to one expert.
    
    Instead of soft responsibilities, uses hard assignment:
        z_i = argmax_k γ_{ik}
        
    Then trains each expert only on samples assigned to it.
    """
    
    def _m_step_experts(
        self,
        X: np.ndarray,
        y: np.ndarray,
        responsibilities: np.ndarray,
    ) -> None:
        """M-step with hard assignments."""
        assignments = responsibilities.argmax(axis=-1)  # (n_samples,)
        
        for k in range(self.config.num_experts):
            mask = assignments == k
            if mask.sum() > 0:
                self.experts[k].fit(X[mask], y[mask])


class SparseEMMOE(EMMOE):
    """EM-MOE with sparse top-k routing.
    
    Only top-k experts are activated per sample, reducing computation
    at inference time.
    """
    
    def __init__(
        self,
        experts: list[ExternalGBDTExpert],
        config: EMConfig | None = None,
        task: Literal["regression", "classification"] = "regression",
        top_k: int = 2,
    ):
        super().__init__(experts, config, task)
        self.top_k = top_k
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using only top-k experts per sample."""
        if not self._fitted:
            raise ValueError("MOE not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        X_jax = jnp.array(X, dtype=jnp.float32)
        
        expert_preds = self._get_expert_predictions(X)
        gating_weights = np.array(self.gating(self.gating_params, X_jax))
        
        # Keep only top-k weights
        n_samples = X.shape[0]
        sparse_weights = np.zeros_like(gating_weights)
        
        for i in range(n_samples):
            top_k_idx = np.argsort(gating_weights[i])[-self.top_k:]
            sparse_weights[i, top_k_idx] = gating_weights[i, top_k_idx]
            sparse_weights[i] /= sparse_weights[i].sum()
        
        return np.sum(sparse_weights * expert_preds, axis=-1)
