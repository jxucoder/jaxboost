"""Information Bottleneck Trees.

Theoretical Foundation:
- Trees compress input X into leaf assignment Z
- Prediction uses Z to estimate Y
- Information Bottleneck: max I(Y;Z) - β·I(X;Z)

Key Insight:
- I(X;Z) measures routing complexity (how input-dependent is leaf assignment)
- I(Y;Z) measures prediction quality (how much info about Y in leaves)
- β controls tradeoff: high β = simpler trees, low β = complex trees

Why This Should Work Better Than XGBoost:
1. Principled regularization (not just max_depth heuristic)
2. Automatic capacity control via β
3. Provably robust to label noise
4. Better generalization on small data

Mathematical Framework:
    Loss = -I(Y;Z) + β·I(X;Z)
    
Variational Approximation (differentiable):
    Loss ≈ E[-log q(y|z)] + β·KL[p(z|x) || r(z)]
    
    Where:
    - q(y|z) = leaf predictors
    - p(z|x) = soft routing distribution
    - r(z) = marginal prior (uniform or learned)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
import optax

from jaxboost.splits import HyperplaneSplit
from jaxboost.structures import ObliviousTree
from jaxboost.routing import soft_routing


class IBTreeParams(NamedTuple):
    """Parameters for IB Tree.
    
    Attributes:
        split_params: Parameters for splits at each depth.
        leaf_values: Prediction values at leaves.
        leaf_log_var: Log variance at each leaf (for uncertainty).
        prior_logits: Prior distribution over leaves (learned).
    """
    split_params: list[Any]
    leaf_values: Array  # (num_leaves,)
    leaf_log_var: Array  # (num_leaves,) - for uncertainty
    prior_logits: Array  # (num_leaves,) - learned prior


class IBTree:
    """Information Bottleneck Tree.
    
    Optimizes: max I(Y;Z) - β·I(X;Z)
    
    Key features:
    1. Soft routing gives p(z|x) - probability distribution over leaves
    2. KL regularization on routing (the I(X;Z) term)
    3. Uncertainty-aware predictions
    4. Automatic complexity control
    
    Example:
        >>> tree = IBTree(depth=4, beta=0.1)
        >>> params = tree.init_params(key, num_features=10)
        >>> 
        >>> # Training
        >>> loss = tree.loss(params, X, y)
        >>> 
        >>> # Prediction with uncertainty
        >>> mean, var = tree.predict_with_uncertainty(params, X)
    """
    
    def __init__(
        self,
        depth: int = 4,
        beta: float = 0.1,
        temperature: float = 1.0,
        learn_prior: bool = True,
    ) -> None:
        """Initialize IB Tree.
        
        Args:
            depth: Tree depth.
            beta: IB tradeoff parameter. Higher = simpler trees.
                  Recommended: 0.01 (complex) to 1.0 (simple)
            temperature: Soft routing temperature.
            learn_prior: Whether to learn the prior p(z) or use uniform.
        """
        self.depth = depth
        self.beta = beta
        self.temperature = temperature
        self.learn_prior = learn_prior
        self.num_leaves = 2 ** depth
        
        self.split_fn = HyperplaneSplit()
        self.tree = ObliviousTree()
    
    def init_params(
        self,
        key: Array,
        num_features: int,
    ) -> IBTreeParams:
        """Initialize parameters."""
        keys = jax.random.split(key, 4)
        
        # Split parameters (one per depth)
        split_params = [
            self.split_fn.init_params(keys[0], num_features)
            for _ in range(self.depth)
        ]
        # Re-initialize with different keys
        for d in range(self.depth):
            key, subkey = jax.random.split(keys[0])
            split_params[d] = self.split_fn.init_params(subkey, num_features)
            keys = jax.random.split(key, 2)
        
        # Leaf values (small init)
        leaf_values = jax.random.normal(keys[1], (self.num_leaves,)) * 0.01
        
        # Leaf variances (log scale, init to reasonable value)
        leaf_log_var = jnp.zeros((self.num_leaves,))  # var = 1.0
        
        # Prior logits (uniform init)
        prior_logits = jnp.zeros((self.num_leaves,))
        
        return IBTreeParams(
            split_params=split_params,
            leaf_values=leaf_values,
            leaf_log_var=leaf_log_var,
            prior_logits=prior_logits,
        )
    
    def _compute_leaf_probs(self, params: IBTreeParams, x: Array) -> Array:
        """Compute p(z|x) - soft routing probabilities to each leaf.
        
        Args:
            params: Model parameters.
            x: Input, shape (batch, num_features).
            
        Returns:
            Leaf probabilities, shape (batch, num_leaves).
        """
        batch_size = x.shape[0]
        
        # Get routing probabilities at each depth
        p_rights = []
        for d in range(self.depth):
            score = self.split_fn.compute_score(params.split_params[d], x)
            p_right = soft_routing(score, self.temperature)
            p_rights.append(p_right)
        p_rights = jnp.stack(p_rights, axis=0)  # (depth, batch)
        
        # Compute leaf probabilities (vectorized)
        leaf_indices = jnp.arange(self.num_leaves)
        depth_indices = jnp.arange(self.depth)
        
        # bit_mask[leaf, depth] = 1 if go right, 0 if go left
        bit_mask = (leaf_indices[:, None] >> depth_indices) & 1
        
        # p_rights: (depth, batch) -> (batch, depth)
        p_rights_T = p_rights.T
        
        # Broadcast and compute
        p_rights_exp = p_rights_T[:, None, :]  # (batch, 1, depth)
        bit_mask_exp = bit_mask[None, :, :]   # (1, num_leaves, depth)
        
        routing_probs = (
            bit_mask_exp * p_rights_exp +
            (1 - bit_mask_exp) * (1 - p_rights_exp)
        )  # (batch, num_leaves, depth)
        
        leaf_probs = jnp.prod(routing_probs, axis=-1)  # (batch, num_leaves)
        
        return leaf_probs
    
    def _get_prior(self, params: IBTreeParams) -> Array:
        """Get prior distribution p(z)."""
        if self.learn_prior:
            return jax.nn.softmax(params.prior_logits)
        else:
            return jnp.ones(self.num_leaves) / self.num_leaves
    
    def forward(self, params: IBTreeParams, x: Array) -> tuple[Array, Array]:
        """Forward pass returning prediction and leaf probabilities.
        
        Args:
            params: Model parameters.
            x: Input, shape (batch, num_features).
            
        Returns:
            Tuple of (predictions, leaf_probs).
        """
        leaf_probs = self._compute_leaf_probs(params, x)
        predictions = leaf_probs @ params.leaf_values
        return predictions, leaf_probs
    
    def predict(self, params: IBTreeParams, x: Array) -> Array:
        """Predict (mean only)."""
        pred, _ = self.forward(params, x)
        return pred
    
    def predict_with_uncertainty(
        self,
        params: IBTreeParams,
        x: Array,
    ) -> tuple[Array, Array]:
        """Predict with uncertainty (variance).
        
        Variance comes from two sources:
        1. Aleatoric: leaf variances (irreducible noise)
        2. Epistemic: routing uncertainty (model uncertainty)
        
        Returns:
            Tuple of (mean, variance).
        """
        leaf_probs = self._compute_leaf_probs(params, x)
        
        # Mean prediction
        mean = leaf_probs @ params.leaf_values
        
        # Variance = E[Var] + Var[E] (law of total variance)
        leaf_var = jnp.exp(params.leaf_log_var)
        
        # E[Var]: expected variance given leaf
        aleatoric = leaf_probs @ leaf_var
        
        # Var[E]: variance of predictions across leaves
        leaf_values_sq = params.leaf_values ** 2
        epistemic = leaf_probs @ leaf_values_sq - mean ** 2
        
        total_var = aleatoric + epistemic
        
        return mean, total_var
    
    def kl_divergence(
        self,
        params: IBTreeParams,
        x: Array,
    ) -> Array:
        """Compute KL[p(z|x) || p(z)] - the I(X;Z) approximation.
        
        This measures how much the routing depends on input.
        High KL = complex routing, Low KL = simple routing.
        """
        leaf_probs = self._compute_leaf_probs(params, x)  # p(z|x)
        prior = self._get_prior(params)  # p(z)
        
        # KL divergence (per sample, then average)
        kl = jnp.sum(
            leaf_probs * (jnp.log(leaf_probs + 1e-8) - jnp.log(prior + 1e-8)),
            axis=-1
        )
        
        return jnp.mean(kl)
    
    def prediction_loss(
        self,
        params: IBTreeParams,
        x: Array,
        y: Array,
        task: str = "regression",
    ) -> Array:
        """Compute prediction loss -I(Y;Z) ≈ -log p(y|z).
        
        For regression: Gaussian likelihood
        For classification: Cross-entropy
        """
        pred, leaf_probs = self.forward(params, x)
        
        if task == "regression":
            # Gaussian negative log likelihood
            # -log p(y|z) ∝ (y - pred)² / (2σ²) + log(σ)
            # Use predicted variance
            _, var = self.predict_with_uncertainty(params, x)
            nll = 0.5 * ((y - pred) ** 2 / (var + 1e-6) + jnp.log(var + 1e-6))
            return jnp.mean(nll)
        else:
            # Binary cross-entropy
            logits = pred
            return jnp.mean(
                -y * jax.nn.log_sigmoid(logits) 
                - (1 - y) * jax.nn.log_sigmoid(-logits)
            )
    
    def loss(
        self,
        params: IBTreeParams,
        x: Array,
        y: Array,
        task: str = "regression",
    ) -> Array:
        """Total IB loss: -I(Y;Z) + β·I(X;Z).
        
        Args:
            params: Model parameters.
            x: Input features.
            y: Targets.
            task: "regression" or "classification".
            
        Returns:
            Total loss.
        """
        pred_loss = self.prediction_loss(params, x, y, task)
        kl_loss = self.kl_divergence(params, x)
        
        return pred_loss + self.beta * kl_loss
    
    def get_effective_num_leaves(self, params: IBTreeParams, x: Array) -> float:
        """Compute effective number of leaves used (for interpretability).
        
        Based on entropy of leaf distribution.
        Max = num_leaves (uniform), Min = 1 (deterministic)
        """
        leaf_probs = self._compute_leaf_probs(params, x)
        avg_probs = jnp.mean(leaf_probs, axis=0)
        entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-8))
        return jnp.exp(entropy)  # Effective number of categories
    
    def get_routing_entropy(self, params: IBTreeParams, x: Array) -> Array:
        """Get entropy of routing per sample (uncertainty measure)."""
        leaf_probs = self._compute_leaf_probs(params, x)
        entropy = -jnp.sum(leaf_probs * jnp.log(leaf_probs + 1e-8), axis=-1)
        return entropy


class IBTreeEnsemble:
    """Ensemble of IB Trees with boosting.
    
    Combines Information Bottleneck regularization with gradient boosting.
    """
    
    def __init__(
        self,
        n_trees: int = 10,
        depth: int = 4,
        beta: float = 0.1,
        tree_weight: float = 0.1,
        task: str = "regression",
    ) -> None:
        self.n_trees = n_trees
        self.depth = depth
        self.beta = beta
        self.tree_weight = tree_weight
        self.task = task
        
        self.trees = [IBTree(depth=depth, beta=beta) for _ in range(n_trees)]
    
    def init_params(self, key: Array, num_features: int) -> list[IBTreeParams]:
        keys = jax.random.split(key, self.n_trees)
        return [
            tree.init_params(keys[i], num_features)
            for i, tree in enumerate(self.trees)
        ]
    
    def forward(self, params_list: list[IBTreeParams], x: Array) -> Array:
        preds = []
        for tree, params in zip(self.trees, params_list):
            pred = tree.predict(params, x)
            preds.append(pred)
        
        preds = jnp.stack(preds, axis=0)  # (n_trees, batch)
        return jnp.sum(preds * self.tree_weight, axis=0)
    
    def loss(
        self,
        params_list: list[IBTreeParams],
        x: Array,
        y: Array,
    ) -> Array:
        # Ensemble prediction
        pred = self.forward(params_list, x)
        
        if self.task == "regression":
            pred_loss = jnp.mean((pred - y) ** 2)
        else:
            pred_loss = jnp.mean(
                -y * jax.nn.log_sigmoid(pred)
                - (1 - y) * jax.nn.log_sigmoid(-pred)
            )
        
        # KL regularization for each tree
        kl_total = 0.0
        for tree, params in zip(self.trees, params_list):
            kl_total += tree.kl_divergence(params, x)
        kl_total /= self.n_trees
        
        return pred_loss + self.beta * kl_total
    
    def fit(
        self,
        X: Array,
        y: Array,
        epochs: int = 300,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> list[IBTreeParams]:
        """Train the ensemble."""
        key = jax.random.PRNGKey(42)
        num_features = X.shape[1]
        
        params_list = self.init_params(key, num_features)
        
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params_list)
        
        @jax.jit
        def step(params, opt_state):
            loss_val, grads = jax.value_and_grad(self.loss)(params, X, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val
        
        for epoch in range(epochs):
            params_list, opt_state, loss_val = step(params_list, opt_state)
            
            if verbose and epoch % 50 == 0:
                pred = self.forward(params_list, X)
                if self.task == "regression":
                    mse = jnp.mean((pred - y) ** 2)
                    print(f"Epoch {epoch}: loss={loss_val:.4f}, MSE={mse:.4f}")
                else:
                    acc = jnp.mean((pred > 0) == y)
                    print(f"Epoch {epoch}: loss={loss_val:.4f}, Acc={acc:.4f}")
        
        return params_list
    
    def predict(self, params_list: list[IBTreeParams], x: Array) -> Array:
        return self.forward(params_list, x)

