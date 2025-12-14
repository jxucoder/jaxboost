"""Mixture of Experts Ensemble with GBDT experts.

Each expert is a gradient boosted ensemble of soft decision trees.
A gating network routes inputs to appropriate experts.

Key features:
- Multiple GBDT experts for capacity
- Flexible gating (Linear, MLP, Tree)
- Optional sparse top-k routing for efficiency
- Load balancing to prevent expert collapse
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import Array

from jaxboost.aggregation import boosting_aggregate
from jaxboost.losses import mse_loss, sigmoid_binary_cross_entropy
from jaxboost.routing import soft_routing
from jaxboost.splits import HyperplaneSplit
from jaxboost.structures import ObliviousTree

from .gating import (
    GatingFn,
    LinearGating,
    MLPGating,
    TreeGating,
    load_balance_loss,
    sparse_top_k,
)


class MOEParams(NamedTuple):
    """Parameters for MOE ensemble.
    
    Attributes:
        gating_params: Parameters for the gating network.
        expert_params: List of expert parameters. Each expert has a list
            of tree parameters (one per tree in the expert).
    """
    gating_params: Any
    expert_params: list[list[Any]]  # [expert_idx][tree_idx]


class MOEEnsemble:
    """Mixture of Experts with GBDT experts.
    
    Architecture:
        Input x → Gating Network → Expert weights g(x)
                → Expert_1(x), Expert_2(x), ..., Expert_K(x)
                → Output = Σ g_k(x) * Expert_k(x)
    
    Each expert is a boosted ensemble of soft oblivious trees.
    
    Example:
        >>> moe = MOEEnsemble(
        ...     num_experts=4,
        ...     trees_per_expert=10,
        ...     tree_depth=4,
        ...     gating="tree",
        ... )
        >>> 
        >>> # Initialize
        >>> key = jax.random.PRNGKey(42)
        >>> params = moe.init_params(key, num_features=10)
        >>> 
        >>> # Forward pass
        >>> predictions = moe.forward(params, X)
        >>> 
        >>> # Training
        >>> params = moe.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        trees_per_expert: int = 10,
        tree_depth: int = 4,
        tree_weight: float = 0.1,
        gating: Literal["linear", "mlp", "tree"] | GatingFn = "tree",
        gating_temperature: float = 1.0,
        top_k: int | None = None,
        load_balance_weight: float = 0.01,
        task: Literal["regression", "classification"] = "regression",
    ) -> None:
        """Initialize MOE ensemble.
        
        Args:
            num_experts: Number of expert GBDTs.
            trees_per_expert: Number of trees in each expert GBDT.
            tree_depth: Depth of each tree.
            tree_weight: Weight for each tree's contribution (learning rate).
            gating: Gating type or custom GatingFn instance.
                - "linear": LinearGating
                - "mlp": MLPGating with hidden_dim=32
                - "tree": TreeGating (depth auto-computed from num_experts)
            gating_temperature: Temperature for gating softmax.
            top_k: If set, only top-k experts are activated (sparse routing).
            load_balance_weight: Weight for load balancing auxiliary loss.
            task: "regression" or "classification".
        """
        self.num_experts = num_experts
        self.trees_per_expert = trees_per_expert
        self.tree_depth = tree_depth
        self.tree_weight = tree_weight
        self.gating_temperature = gating_temperature
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.task = task
        
        # Initialize gating
        if isinstance(gating, str):
            if gating == "linear":
                self.gating = LinearGating()
            elif gating == "mlp":
                self.gating = MLPGating(hidden_dim=32)
            elif gating == "tree":
                # Compute depth from num_experts
                tree_gating_depth = int(jnp.ceil(jnp.log2(num_experts)))
                if 2 ** tree_gating_depth != num_experts:
                    raise ValueError(
                        f"TreeGating requires num_experts to be a power of 2, "
                        f"got {num_experts}. Use {2 ** tree_gating_depth} experts "
                        f"or choose a different gating type."
                    )
                self.gating = TreeGating(depth=tree_gating_depth)
            else:
                raise ValueError(f"Unknown gating type: {gating}")
        else:
            self.gating = gating
        
        # Tree components (shared across all experts)
        self.split_fn = HyperplaneSplit()
        self.tree = ObliviousTree()
    
    def init_params(
        self,
        key: Array,
        num_features: int,
    ) -> MOEParams:
        """Initialize all parameters.
        
        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            
        Returns:
            Initialized MOE parameters.
        """
        keys = jax.random.split(key, self.num_experts + 1)
        
        # Initialize gating
        gating_params = self.gating.init_params(
            keys[0], num_features, self.num_experts
        )
        
        # Initialize experts
        expert_params = []
        for k in range(self.num_experts):
            expert_keys = jax.random.split(keys[k + 1], self.trees_per_expert)
            trees = [
                self.tree.init_params(
                    expert_keys[t], 
                    self.tree_depth, 
                    num_features, 
                    self.split_fn,
                )
                for t in range(self.trees_per_expert)
            ]
            expert_params.append(trees)
        
        return MOEParams(gating_params=gating_params, expert_params=expert_params)
    
    def forward(
        self,
        params: MOEParams,
        x: Array,
        routing_temperature: float = 1.0,
    ) -> Array:
        """Forward pass through MOE.
        
        Args:
            params: MOE parameters.
            x: Input features, shape (batch, num_features).
            routing_temperature: Temperature for soft routing in trees.
            
        Returns:
            Predictions, shape (batch,).
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x[None, :]
        
        # Compute gating weights
        gate_weights = self.gating(
            params.gating_params, x, self.gating_temperature
        )  # (batch, num_experts)
        
        # Apply sparse top-k if specified
        if self.top_k is not None:
            gate_weights, _ = sparse_top_k(gate_weights, self.top_k)
        
        # Compute expert predictions
        expert_preds = self._compute_expert_predictions(
            params.expert_params, x, routing_temperature
        )  # (batch, num_experts)
        
        # Weighted combination
        output = jnp.sum(gate_weights * expert_preds, axis=-1)  # (batch,)
        
        return output[0] if single_sample else output
    
    def _compute_expert_predictions(
        self,
        expert_params: list[list[Any]],
        x: Array,
        routing_temperature: float,
    ) -> Array:
        """Compute predictions from all experts.
        
        Args:
            expert_params: List of expert parameters.
            x: Input features, shape (batch, num_features).
            routing_temperature: Temperature for soft routing.
            
        Returns:
            Expert predictions, shape (batch, num_experts).
        """
        def routing_fn(score):
            return soft_routing(score, routing_temperature)
        
        expert_preds = []
        for k in range(self.num_experts):
            # Each expert is a boosted ensemble
            tree_preds = []
            for tree_params in expert_params[k]:
                pred = self.tree.forward(tree_params, x, self.split_fn, routing_fn)
                tree_preds.append(pred)
            
            tree_preds = jnp.stack(tree_preds, axis=0)  # (trees_per_expert, batch)
            weights = jnp.full((self.trees_per_expert,), self.tree_weight)
            expert_pred = boosting_aggregate(tree_preds, weights)  # (batch,)
            expert_preds.append(expert_pred)
        
        return jnp.stack(expert_preds, axis=-1)  # (batch, num_experts)
    
    def loss(
        self,
        params: MOEParams,
        x: Array,
        y: Array,
        routing_temperature: float = 1.0,
    ) -> Array:
        """Compute total loss including auxiliary losses.
        
        Args:
            params: MOE parameters.
            x: Input features, shape (batch, num_features).
            y: Targets, shape (batch,).
            routing_temperature: Temperature for soft routing.
            
        Returns:
            Total loss (scalar).
        """
        # Forward pass
        pred = self.forward(params, x, routing_temperature)
        
        # Task loss
        if self.task == "regression":
            task_loss = mse_loss(pred, y)
        else:
            task_loss = sigmoid_binary_cross_entropy(pred, y)
        
        # Load balancing loss
        gate_weights = self.gating(
            params.gating_params, x, self.gating_temperature
        )
        lb_loss = load_balance_loss(gate_weights)
        
        return task_loss + self.load_balance_weight * lb_loss
    
    def fit(
        self,
        X: Array,
        y: Array,
        X_val: Array | None = None,
        y_val: Array | None = None,
        epochs: int = 300,
        learning_rate: float = 0.01,
        temp_start: float = 1.0,
        temp_end: float = 5.0,
        patience: int = 50,
        verbose: bool = True,
    ) -> MOEParams:
        """Train the MOE ensemble.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            X_val: Validation features (optional).
            y_val: Validation targets (optional).
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimizer.
            temp_start: Starting temperature for routing.
            temp_end: Ending temperature for routing.
            patience: Early stopping patience.
            verbose: Whether to print progress.
            
        Returns:
            Trained parameters.
        """
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32)
        
        # Create validation split if not provided
        if X_val is None:
            n_samples = X.shape[0]
            split_idx = int(n_samples * 0.85)
            indices = jax.random.permutation(jax.random.PRNGKey(42), n_samples)
            train_idx, val_idx = indices[:split_idx], indices[split_idx:]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val = jnp.array(X_val, dtype=jnp.float32)
            y_val = jnp.array(y_val, dtype=jnp.float32)
        
        # Initialize
        key = jax.random.PRNGKey(42)
        num_features = X_train.shape[1]
        params = self.init_params(key, num_features)
        
        # Setup optimizer
        schedule = optax.cosine_decay_schedule(learning_rate, epochs, alpha=0.01)
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(params)
        
        # Training loop
        best_val_loss = float('inf')
        best_params = params
        no_improve = 0
        
        @jax.jit
        def train_step(params, opt_state, temperature):
            loss_val, grads = jax.value_and_grad(self.loss)(
                params, X_train, y_train, temperature
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val
        
        for epoch in range(epochs):
            # Temperature annealing
            progress = epoch / epochs
            temperature = temp_start + progress * (temp_end - temp_start)
            
            # Train step
            params, opt_state, train_loss = train_step(params, opt_state, temperature)
            
            # Validation
            if epoch % 10 == 0:
                val_loss = self.loss(params, X_val, y_val, temperature)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = params
                    no_improve = 0
                else:
                    no_improve += 10
                
                if verbose and epoch % 50 == 0:
                    # Compute metrics
                    pred_train = self.forward(params, X_train, temperature)
                    pred_val = self.forward(params, X_val, temperature)
                    
                    if self.task == "regression":
                        train_metric = jnp.mean((pred_train - y_train) ** 2)
                        val_metric = jnp.mean((pred_val - y_val) ** 2)
                        metric_name = "MSE"
                    else:
                        train_metric = jnp.mean((pred_train > 0) == y_train)
                        val_metric = jnp.mean((pred_val > 0) == y_val)
                        metric_name = "Acc"
                    
                    print(
                        f"Epoch {epoch}: loss={train_loss:.4f}, "
                        f"train_{metric_name}={train_metric:.4f}, "
                        f"val_{metric_name}={val_metric:.4f}"
                    )
                
                if no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
        
        return best_params
    
    def predict(
        self,
        params: MOEParams,
        X: Array,
        routing_temperature: float = 5.0,
    ) -> Array:
        """Make predictions.
        
        Args:
            params: Trained parameters.
            X: Features, shape (n_samples, n_features).
            routing_temperature: Temperature for routing (use trained temp).
            
        Returns:
            Predictions. For regression: values. For classification: probabilities.
        """
        X = jnp.array(X, dtype=jnp.float32)
        output = self.forward(params, X, routing_temperature)
        
        if self.task == "classification":
            return jax.nn.sigmoid(output)
        return output
    
    def get_expert_weights(
        self,
        params: MOEParams,
        X: Array,
    ) -> Array:
        """Get expert weights for analysis.
        
        Args:
            params: MOE parameters.
            X: Input features, shape (n_samples, n_features).
            
        Returns:
            Expert weights, shape (n_samples, num_experts).
        """
        X = jnp.array(X, dtype=jnp.float32)
        gate_weights = self.gating(
            params.gating_params, X, self.gating_temperature
        )
        if self.top_k is not None:
            gate_weights, _ = sparse_top_k(gate_weights, self.top_k)
        return gate_weights
    
    def get_expert_predictions(
        self,
        params: MOEParams,
        X: Array,
        routing_temperature: float = 5.0,
    ) -> Array:
        """Get individual expert predictions for analysis.
        
        Args:
            params: MOE parameters.
            X: Input features, shape (n_samples, n_features).
            routing_temperature: Temperature for routing.
            
        Returns:
            Expert predictions, shape (n_samples, num_experts).
        """
        X = jnp.array(X, dtype=jnp.float32)
        return self._compute_expert_predictions(
            params.expert_params, X, routing_temperature
        )


class TaskAwareMOEEnsemble(MOEEnsemble):
    """MOE with task-aware gating for multi-task learning.
    
    Extends MOEEnsemble to accept task IDs, allowing the gating network
    to route based on both input features and task identity.
    
    Example:
        >>> moe = TaskAwareMOEEnsemble(num_experts=4, num_tasks=3)
        >>> params = moe.init_params(key, num_features=10)
        >>> 
        >>> # Forward with task ID
        >>> predictions = moe.forward(params, X, task_ids=task_ids)
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        num_tasks: int = 1,
        task_embed_dim: int = 8,
        **kwargs,
    ) -> None:
        """Initialize task-aware MOE.
        
        Args:
            num_experts: Number of experts.
            num_tasks: Number of tasks.
            task_embed_dim: Dimension of task embeddings.
            **kwargs: Additional arguments for MOEEnsemble.
        """
        # Force linear gating for task-aware version
        kwargs["gating"] = "linear"  # Will be overridden
        super().__init__(num_experts=num_experts, **kwargs)
        
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim
        
        # Override gating with task-aware version
        self.gating = _TaskAwareLinearGating(
            num_tasks=num_tasks,
            task_embed_dim=task_embed_dim,
        )
    
    def forward(
        self,
        params: MOEParams,
        x: Array,
        task_ids: Array | None = None,
        routing_temperature: float = 1.0,
    ) -> Array:
        """Forward pass with task awareness.
        
        Args:
            params: MOE parameters.
            x: Input features, shape (batch, num_features).
            task_ids: Task IDs, shape (batch,). If None, uses task 0.
            routing_temperature: Temperature for routing.
            
        Returns:
            Predictions, shape (batch,).
        """
        if task_ids is None:
            task_ids = jnp.zeros(x.shape[0], dtype=jnp.int32)
        
        single_sample = x.ndim == 1
        if single_sample:
            x = x[None, :]
            task_ids = jnp.array([task_ids])
        
        # Compute gating weights with task info
        gate_weights = self.gating(
            params.gating_params, x, task_ids, self.gating_temperature
        )
        
        if self.top_k is not None:
            gate_weights, _ = sparse_top_k(gate_weights, self.top_k)
        
        expert_preds = self._compute_expert_predictions(
            params.expert_params, x, routing_temperature
        )
        
        output = jnp.sum(gate_weights * expert_preds, axis=-1)
        
        return output[0] if single_sample else output


class _TaskAwareLinearGating:
    """Linear gating with task embeddings."""
    
    def __init__(self, num_tasks: int, task_embed_dim: int) -> None:
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim
    
    def init_params(
        self,
        key: Array,
        num_features: int,
        num_experts: int,
    ) -> dict:
        k1, k2, k3 = jax.random.split(key, 3)
        
        return {
            'task_embeddings': jax.random.normal(
                k1, (self.num_tasks, self.task_embed_dim)
            ) * 0.1,
            'W': jax.random.normal(
                k2, (num_experts, num_features + self.task_embed_dim)
            ) * 0.01,
            'b': jnp.zeros(num_experts),
        }
    
    def __call__(
        self,
        params: dict,
        x: Array,
        task_ids: Array,
        temperature: float = 1.0,
    ) -> Array:
        # Get task embeddings
        task_emb = params['task_embeddings'][task_ids]  # (batch, task_embed_dim)
        
        # Concatenate features and task embedding
        x_combined = jnp.concatenate([x, task_emb], axis=-1)
        
        # Linear gating
        logits = x_combined @ params['W'].T + params['b']
        return jax.nn.softmax(logits / temperature, axis=-1)

