"""Gradient boosting trainer with soft oblivious trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from jaxboost.aggregation import boosting_aggregate
from jaxboost.losses import mse_loss, sigmoid_binary_cross_entropy
from jaxboost.routing import soft_routing
from jaxboost.splits import HyperplaneSplit
from jaxboost.structures import ObliviousTree


@dataclass
class TrainerConfig:
    """Configuration for forest training.
    
    Attributes:
        n_trees: Number of trees in the forest.
        depth: Maximum depth of each tree.
        learning_rate: Base learning rate for optimizer.
        tree_weight: Weight for each tree's contribution.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        temp_start: Starting temperature for soft routing.
        temp_end: Ending temperature (None = auto based on data size).
        val_fraction: Fraction of training data for validation (if no val set provided).
        normalize_target: Whether to normalize regression targets.
        verbose: Whether to print training progress.
    """
    n_trees: int = 20
    depth: int = 4
    learning_rate: float = 0.01
    tree_weight: float = 0.1
    epochs: int = 500
    patience: int = 50
    temp_start: float = 1.0
    temp_end: float | None = None  # Auto-detect
    val_fraction: float = 0.15
    normalize_target: bool = True
    verbose: bool = False


@dataclass
class TrainedGBM:
    """A trained gradient boosting model."""
    params: list
    config: TrainerConfig
    target_mean: float
    target_std: float
    task: str
    split_fn: Any
    tree: Any
    
    def predict(self, X: np.ndarray | Array) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features, shape (n_samples, n_features).
            
        Returns:
            Predictions. For regression: continuous values.
            For classification: probabilities.
        """
        X = jnp.array(X)
        
        def routing_fn(score):
            return soft_routing(score, self.config.temp_end)
        
        preds = []
        for tree_params in self.params:
            pred = self.tree.forward(tree_params, X, self.split_fn, routing_fn)
            preds.append(pred)
        
        preds = jnp.stack(preds, axis=0)
        weights = jnp.full((len(self.params),), self.config.tree_weight)
        output = boosting_aggregate(preds, weights)
        
        if self.task == "regression":
            # Denormalize
            output = output * self.target_std + self.target_mean
        else:
            # Return probabilities
            output = jax.nn.sigmoid(output)
        
        return np.array(output)
    
    def predict_class(self, X: np.ndarray | Array, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels for classification.
        
        Args:
            X: Features.
            threshold: Classification threshold.
            
        Returns:
            Class labels (0 or 1).
        """
        probs = self.predict(X)
        return (probs > threshold).astype(np.int32)


class GBMTrainer:
    """Gradient boosting trainer with soft oblivious trees.
    
    Example:
        >>> trainer = GBMTrainer(task="regression")
        >>> model = trainer.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        config: TrainerConfig | None = None,
    ):
        """Initialize trainer.
        
        Args:
            task: 'regression' or 'classification'.
            config: Training configuration. If None, uses defaults.
        """
        self.task = task
        self.config = config or TrainerConfig()
        self.split_fn = HyperplaneSplit()
        self.tree = ObliviousTree()
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainedGBM:
        """Train a forest on the given data.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.
            
        Returns:
            Trained model.
        """
        config = self.config
        n_samples, n_features = X.shape
        
        # Auto-adjust hyperparameters based on data size
        config = self._adapt_config(config, n_samples)
        
        # Create validation split if not provided
        if X_val is None:
            split_idx = int(n_samples * (1 - config.val_fraction))
            indices = np.random.RandomState(42).permutation(n_samples)
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
        
        # Normalize target for regression
        if self.task == "regression" and config.normalize_target:
            target_mean, target_std = float(y_train.mean()), float(y_train.std())
            y_train = (y_train - target_mean) / target_std
            y_val = (y_val - target_mean) / target_std
        else:
            target_mean, target_std = 0.0, 1.0
        
        # Convert to JAX arrays
        X_train = jnp.array(X_train, dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.float32)
        X_val = jnp.array(X_val, dtype=jnp.float32)
        y_val = jnp.array(y_val, dtype=jnp.float32)
        
        # Initialize trees
        key = jax.random.PRNGKey(42)
        forest_params = []
        for _ in range(config.n_trees):
            key, subkey = jax.random.split(key)
            params = self.tree.init_params(
                subkey, config.depth, n_features, self.split_fn
            )
            forest_params.append(params)
        
        # Setup optimizer
        schedule = optax.cosine_decay_schedule(
            config.learning_rate, config.epochs, alpha=0.01
        )
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(forest_params)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_params = forest_params
        no_improve = 0
        
        @jax.jit
        def train_step(params, opt_state, X, y, temperature):
            loss, grads = jax.value_and_grad(self._loss_fn)(
                params, X, y, temperature, config
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        for epoch in range(config.epochs):
            progress = epoch / config.epochs
            temperature = config.temp_start + progress * (config.temp_end - config.temp_start)
            
            forest_params, opt_state, train_loss = train_step(
                forest_params, opt_state, X_train, y_train, temperature
            )
            
            # Validation check
            if epoch % 10 == 0:
                val_loss = self._loss_fn(
                    forest_params, X_val, y_val, temperature, config
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = forest_params
                    no_improve = 0
                else:
                    no_improve += 10
                
                if config.verbose and epoch % 50 == 0:
                    print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                if no_improve >= config.patience:
                    if config.verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break
        
        return TrainedGBM(
            params=best_params,
            config=config,
            target_mean=target_mean,
            target_std=target_std,
            task=self.task,
            split_fn=self.split_fn,
            tree=self.tree,
        )
    
    def _adapt_config(self, config: TrainerConfig, n_samples: int) -> TrainerConfig:
        """Adapt configuration based on dataset size."""
        if config.temp_end is None:
            if n_samples < 500:
                temp_end = 3.0
            elif n_samples < 5000:
                temp_end = 5.0
            else:
                temp_end = 10.0
        else:
            temp_end = config.temp_end
        
        depth = config.depth
        lr = config.learning_rate
        if n_samples < 500:
            depth = min(depth, 3)
            lr = min(lr, 0.01)
        
        return TrainerConfig(
            n_trees=config.n_trees,
            depth=depth,
            learning_rate=lr,
            tree_weight=config.tree_weight,
            epochs=config.epochs,
            patience=config.patience,
            temp_start=config.temp_start,
            temp_end=temp_end,
            val_fraction=config.val_fraction,
            normalize_target=config.normalize_target,
            verbose=config.verbose,
        )
    
    def _loss_fn(
        self,
        params: list,
        X: Array,
        y: Array,
        temperature: float,
        config: TrainerConfig,
    ) -> Array:
        """Compute loss for the forest."""
        def routing_fn(score):
            return soft_routing(score, temperature)
        
        preds = []
        for tree_params in params:
            pred = self.tree.forward(tree_params, X, self.split_fn, routing_fn)
            preds.append(pred)
        
        preds = jnp.stack(preds, axis=0)
        weights = jnp.full((config.n_trees,), config.tree_weight)
        output = boosting_aggregate(preds, weights)
        
        if self.task == "regression":
            return mse_loss(output, y)
        else:
            return sigmoid_binary_cross_entropy(output, y)

