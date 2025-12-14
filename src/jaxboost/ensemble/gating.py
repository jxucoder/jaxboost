"""Gating functions for Mixture of Experts.

Gating networks determine how to route inputs to different experts.
Three options provided:
1. LinearGating - Simple linear projection (baseline)
2. MLPGating - Two-layer MLP for non-linear routing
3. TreeGating - Soft decision tree (JaxBoost style, most interpretable)
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from jaxboost.routing import soft_routing
from jaxboost.splits import HyperplaneSplit


@runtime_checkable
class GatingFn(Protocol):
    """Protocol for gating functions.
    
    A gating function computes expert weights for each input sample.
    """
    
    def init_params(self, key: Array, num_features: int, num_experts: int) -> Any:
        """Initialize gating parameters.
        
        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            num_experts: Number of experts.
            
        Returns:
            Gating parameters (any PyTree).
        """
        ...
    
    def __call__(
        self, 
        params: Any, 
        x: Array, 
        temperature: float = 1.0,
    ) -> Array:
        """Compute expert weights.
        
        Args:
            params: Gating parameters.
            x: Input features, shape (batch, num_features).
            temperature: Softmax temperature. Higher = softer routing.
            
        Returns:
            Expert weights, shape (batch, num_experts), softmax normalized.
        """
        ...


# =============================================================================
# Linear Gating
# =============================================================================

class LinearGatingParams(NamedTuple):
    """Parameters for linear gating.
    
    Attributes:
        W: Weight matrix, shape (num_experts, num_features).
        b: Bias vector, shape (num_experts,).
    """
    W: Array
    b: Array


class LinearGating:
    """Linear gating: g(x) = softmax(W @ x + b).
    
    Simple and interpretable. Each row of W represents an expert's
    "preference" for different features.
    
    Example:
        >>> gating = LinearGating()
        >>> params = gating.init_params(key, num_features=10, num_experts=4)
        >>> weights = gating(params, x)  # (batch, 4)
    """
    
    def init_params(
        self,
        key: Array,
        num_features: int,
        num_experts: int,
        init_scale: float = 0.01,
    ) -> LinearGatingParams:
        """Initialize gating parameters.
        
        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            num_experts: Number of experts.
            init_scale: Scale for weight initialization.
            
        Returns:
            Initialized parameters.
        """
        W = jax.random.normal(key, (num_experts, num_features)) * init_scale
        b = jnp.zeros(num_experts)
        return LinearGatingParams(W=W, b=b)
    
    def __call__(
        self,
        params: LinearGatingParams,
        x: Array,
        temperature: float = 1.0,
    ) -> Array:
        """Compute expert weights.
        
        Args:
            params: Gating parameters.
            x: Input features, shape (batch, num_features).
            temperature: Softmax temperature.
            
        Returns:
            Expert weights, shape (batch, num_experts).
        """
        # x: (batch, num_features), W: (num_experts, num_features)
        logits = x @ params.W.T + params.b  # (batch, num_experts)
        return jax.nn.softmax(logits / temperature, axis=-1)


# =============================================================================
# MLP Gating
# =============================================================================

class MLPGatingParams(NamedTuple):
    """Parameters for MLP gating.
    
    Attributes:
        W1: First layer weights, shape (hidden_dim, num_features).
        b1: First layer bias, shape (hidden_dim,).
        W2: Second layer weights, shape (num_experts, hidden_dim).
        b2: Second layer bias, shape (num_experts,).
    """
    W1: Array
    b1: Array
    W2: Array
    b2: Array


class MLPGating:
    """MLP gating: g(x) = softmax(W2 @ relu(W1 @ x + b1) + b2).
    
    Two-layer MLP for non-linear routing decisions.
    More expressive than linear, but may overfit on small data.
    
    Example:
        >>> gating = MLPGating(hidden_dim=32)
        >>> params = gating.init_params(key, num_features=10, num_experts=4)
        >>> weights = gating(params, x)  # (batch, 4)
    """
    
    def __init__(self, hidden_dim: int = 32) -> None:
        """Initialize MLP gating.
        
        Args:
            hidden_dim: Hidden layer dimension.
        """
        self.hidden_dim = hidden_dim
    
    def init_params(
        self,
        key: Array,
        num_features: int,
        num_experts: int,
        init_scale: float = 0.01,
    ) -> MLPGatingParams:
        """Initialize gating parameters.
        
        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            num_experts: Number of experts.
            init_scale: Scale for weight initialization.
            
        Returns:
            Initialized parameters.
        """
        k1, k2 = jax.random.split(key)
        
        # Xavier-style initialization
        std1 = init_scale * jnp.sqrt(2.0 / (num_features + self.hidden_dim))
        std2 = init_scale * jnp.sqrt(2.0 / (self.hidden_dim + num_experts))
        
        W1 = jax.random.normal(k1, (self.hidden_dim, num_features)) * std1
        b1 = jnp.zeros(self.hidden_dim)
        W2 = jax.random.normal(k2, (num_experts, self.hidden_dim)) * std2
        b2 = jnp.zeros(num_experts)
        
        return MLPGatingParams(W1=W1, b1=b1, W2=W2, b2=b2)
    
    def __call__(
        self,
        params: MLPGatingParams,
        x: Array,
        temperature: float = 1.0,
    ) -> Array:
        """Compute expert weights.
        
        Args:
            params: Gating parameters.
            x: Input features, shape (batch, num_features).
            temperature: Softmax temperature.
            
        Returns:
            Expert weights, shape (batch, num_experts).
        """
        # First layer
        h = jax.nn.relu(x @ params.W1.T + params.b1)  # (batch, hidden_dim)
        
        # Second layer
        logits = h @ params.W2.T + params.b2  # (batch, num_experts)
        
        return jax.nn.softmax(logits / temperature, axis=-1)


# =============================================================================
# Tree Gating
# =============================================================================

class TreeGatingParams(NamedTuple):
    """Parameters for tree gating.
    
    Attributes:
        split_params: Split parameters, one per depth level.
    """
    split_params: list[Any]


class TreeGating:
    """Tree-based gating using soft decision tree.
    
    Uses an oblivious soft decision tree where leaf probabilities
    become expert weights. Most interpretable option.
    
    The number of experts is determined by tree depth: num_experts = 2^depth.
    
    Example:
        >>> gating = TreeGating(depth=2)  # 4 experts
        >>> params = gating.init_params(key, num_features=10)
        >>> weights = gating(params, x)  # (batch, 4)
        
    Interpretability:
        Each path through the tree defines an expert's "activation region".
        - Expert 0: left at depth 0, left at depth 1
        - Expert 1: left at depth 0, right at depth 1
        - Expert 2: right at depth 0, left at depth 1
        - Expert 3: right at depth 0, right at depth 1
    """
    
    def __init__(self, depth: int = 2) -> None:
        """Initialize tree gating.
        
        Args:
            depth: Tree depth. Number of experts = 2^depth.
        """
        self.depth = depth
        self.num_experts = 2 ** depth
        self.split_fn = HyperplaneSplit()
    
    def init_params(
        self,
        key: Array,
        num_features: int,
        num_experts: int | None = None,
    ) -> TreeGatingParams:
        """Initialize gating parameters.
        
        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            num_experts: Ignored (determined by depth). Kept for API consistency.
            
        Returns:
            Initialized parameters.
        """
        if num_experts is not None and num_experts != self.num_experts:
            raise ValueError(
                f"TreeGating with depth={self.depth} has {self.num_experts} experts, "
                f"but num_experts={num_experts} was requested. "
                f"Use depth={int(jnp.log2(num_experts))} for {num_experts} experts."
            )
        
        keys = jax.random.split(key, self.depth)
        split_params = [
            self.split_fn.init_params(keys[d], num_features)
            for d in range(self.depth)
        ]
        
        return TreeGatingParams(split_params=split_params)
    
    def __call__(
        self,
        params: TreeGatingParams,
        x: Array,
        temperature: float = 1.0,
    ) -> Array:
        """Compute expert weights (leaf probabilities).
        
        Args:
            params: Gating parameters.
            x: Input features, shape (batch, num_features).
            temperature: Routing temperature.
            
        Returns:
            Expert weights, shape (batch, num_experts).
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x[None, :]
        
        # Compute routing probabilities at each depth
        p_rights = []
        for d in range(self.depth):
            score = self.split_fn.compute_score(params.split_params[d], x)
            p_right = soft_routing(score, temperature)
            p_rights.append(p_right)
        p_rights = jnp.stack(p_rights, axis=0)  # (depth, batch)
        
        # Compute leaf probabilities (expert weights)
        leaf_probs = self._compute_leaf_probs(p_rights)  # (batch, num_experts)
        
        if single_sample:
            return leaf_probs[0]
        return leaf_probs
    
    def _compute_leaf_probs(self, p_rights: Array) -> Array:
        """Compute probability of reaching each leaf (expert).
        
        Uses bit manipulation + broadcasting for efficiency.
        
        Args:
            p_rights: Right routing probabilities, shape (depth, batch).
            
        Returns:
            Leaf probabilities, shape (batch, num_experts).
        """
        # Create binary mask: which way to go at each depth for each leaf
        leaf_indices = jnp.arange(self.num_experts)
        depth_indices = jnp.arange(self.depth)
        
        # bit_mask[leaf, depth] = 1 if leaf goes right at depth, else 0
        bit_mask = (leaf_indices[:, None] >> depth_indices) & 1  # (num_experts, depth)
        
        # p_rights: (depth, batch) -> (batch, depth)
        p_rights_T = p_rights.T  # (batch, depth)
        
        # Broadcast: (batch, 1, depth) vs (1, num_experts, depth)
        p_rights_expanded = p_rights_T[:, None, :]  # (batch, 1, depth)
        bit_mask_expanded = bit_mask[None, :, :]  # (1, num_experts, depth)
        
        # routing_prob = p_right if go_right else (1 - p_right)
        routing_probs = (
            bit_mask_expanded * p_rights_expanded
            + (1 - bit_mask_expanded) * (1 - p_rights_expanded)
        )  # (batch, num_experts, depth)
        
        # Product over depth dimension
        leaf_probs = jnp.prod(routing_probs, axis=-1)  # (batch, num_experts)
        
        return leaf_probs
    
    def get_routing_rules(self, params: TreeGatingParams) -> list[str]:
        """Get human-readable routing rules for each expert.
        
        Useful for interpretability.
        
        Args:
            params: Gating parameters.
            
        Returns:
            List of rule strings, one per expert.
        """
        rules = []
        for expert_idx in range(self.num_experts):
            conditions = []
            for d in range(self.depth):
                go_right = (expert_idx >> d) & 1
                direction = ">" if go_right else "<="
                conditions.append(f"split_{d} {direction} 0")
            rules.append(f"Expert {expert_idx}: " + " AND ".join(conditions))
        return rules


# =============================================================================
# Sparse Gating Utilities
# =============================================================================

def sparse_top_k(
    gate_weights: Array,
    k: int,
) -> tuple[Array, Array]:
    """Apply top-k sparsity to gate weights.
    
    Only the top-k experts are activated, others are zeroed out.
    Weights are renormalized after selection.
    
    Args:
        gate_weights: Dense gate weights, shape (batch, num_experts).
        k: Number of experts to keep.
        
    Returns:
        Tuple of:
        - sparse_weights: Sparse gate weights, shape (batch, num_experts).
        - top_k_indices: Indices of selected experts, shape (batch, k).
    """
    batch_size, num_experts = gate_weights.shape
    
    # Get top-k indices and values
    top_k_values, top_k_indices = jax.lax.top_k(gate_weights, k)
    
    # Renormalize top-k values
    top_k_probs = top_k_values / (jnp.sum(top_k_values, axis=-1, keepdims=True) + 1e-8)
    
    # Scatter back to sparse tensor
    sparse_weights = jnp.zeros_like(gate_weights)
    batch_idx = jnp.arange(batch_size)[:, None]  # (batch, 1)
    sparse_weights = sparse_weights.at[batch_idx, top_k_indices].set(top_k_probs)
    
    return sparse_weights, top_k_indices


def load_balance_loss(gate_weights: Array) -> Array:
    """Compute load balancing loss to encourage uniform expert usage.
    
    Penalizes variance in expert load across the batch.
    
    Args:
        gate_weights: Gate weights, shape (batch, num_experts).
        
    Returns:
        Load balance loss (scalar). Lower = more balanced.
    """
    # Average load per expert
    expert_load = jnp.mean(gate_weights, axis=0)  # (num_experts,)
    
    # Variance penalty (0 when perfectly balanced)
    num_experts = gate_weights.shape[1]
    return jnp.var(expert_load) * num_experts


def router_z_loss(gate_logits: Array) -> Array:
    """Compute router z-loss for training stability.
    
    Penalizes large logits to prevent router from becoming too confident.
    From "ST-MoE: Designing Stable and Transferable Sparse Expert Models".
    
    Args:
        gate_logits: Raw gate logits before softmax, shape (batch, num_experts).
        
    Returns:
        Z-loss (scalar).
    """
    # Log-sum-exp of logits
    log_z = jax.scipy.special.logsumexp(gate_logits, axis=-1)
    return jnp.mean(log_z ** 2)

