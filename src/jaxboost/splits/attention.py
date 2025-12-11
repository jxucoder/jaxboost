"""Attention-based split function (simplified & regularized).

Uses input-dependent attention to dynamically select features for each sample.
Includes built-in regularization to prevent overfitting.

Note: Benchmarks show attention often overfits compared to HyperplaneSplit.
Use only when you have:
- Large datasets (n > 5000)
- Classification with complex feature interactions
- Sufficient regularization in your training loop
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class AttentionSplitParams(NamedTuple):
    """Parameters for attention-based split.

    Attributes:
        W_q: Projects input to query, shape (num_features, head_dim).
        W_k: Projects feature positions to keys, shape (num_features, head_dim).
        W_v: Value weights per feature, shape (num_features,).
        threshold: Split threshold, scalar.
    """

    W_q: Array  # (num_features, head_dim)
    W_k: Array  # (num_features, head_dim)
    W_v: Array  # (num_features,)
    threshold: Array  # scalar


class AttentionSplit:
    """Input-dependent attention split with regularization.
    
    Simplified design with built-in regularization:
    - Smaller head_dim (default 4) to reduce parameters
    - Temperature scaling for sharper/softer attention
    - Entropy regularization to prevent attention collapse
    - L2 regularization helper for training
    
    For each sample:
    1. Generate query from input: q = x @ W_q
    2. Compute attention: attn = softmax((q @ W_k.T) / (sqrt(d) * temperature))
    3. Aggregate: output = sum(x * attn * W_v)
    
    Example:
        >>> split_fn = AttentionSplit(head_dim=4, temperature=2.0)
        >>> params = split_fn.init_params(key, num_features=10)
        >>> score = split_fn.compute_score(params, x)
        >>> 
        >>> # Add regularization to your loss:
        >>> reg_loss = split_fn.l2_regularization(params, weight=0.01)
        >>> entropy_loss = split_fn.entropy_regularization(params, x, weight=0.1)
    """

    def __init__(
        self,
        head_dim: int = 4,
        temperature: float = 2.0,
    ) -> None:
        """Initialize attention split.
        
        Args:
            head_dim: Dimension of query/key space. Smaller = fewer params.
                      Default 4 (was 8). Use 2 for very small datasets.
            temperature: Softmax temperature. Higher = softer attention,
                         better generalization. Default 2.0 (was implicit 1.0).
        """
        self.head_dim = head_dim
        self.temperature = temperature

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.05,  # Smaller default (was 0.1)
    ) -> AttentionSplitParams:
        """Initialize split parameters with small weights."""
        keys = jax.random.split(key, 4)
        
        # Xavier/Glorot-style initialization scaled down
        fan_in = num_features
        fan_out = self.head_dim
        std = init_scale * jnp.sqrt(2.0 / (fan_in + fan_out))
        
        W_q = jax.random.normal(keys[0], (num_features, self.head_dim)) * std
        W_k = jax.random.normal(keys[1], (num_features, self.head_dim)) * std
        W_v = jax.random.normal(keys[2], (num_features,)) * init_scale
        threshold = jnp.zeros(())  # Start at zero
        
        return AttentionSplitParams(W_q=W_q, W_k=W_k, W_v=W_v, threshold=threshold)

    def compute_score(self, params: AttentionSplitParams, x: Array) -> Array:
        """Compute split score with temperature-scaled attention.
        
        Args:
            params: Split parameters.
            x: Input features, shape (..., num_features).
            
        Returns:
            Split scores, shape (...).
        """
        # Query from input
        q = x @ params.W_q  # (..., head_dim)
        
        # Attention logits with temperature scaling
        attn_logits = (q @ params.W_k.T) / (jnp.sqrt(self.head_dim) * self.temperature)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (..., num_features)
        
        # Weighted aggregation
        weighted = x * attn_weights * params.W_v
        aggregated = jnp.sum(weighted, axis=-1)
        
        return aggregated - params.threshold

    def l2_regularization(self, params: AttentionSplitParams, weight: float = 0.01) -> Array:
        """Compute L2 regularization on attention weights.
        
        Add this to your loss to prevent overfitting:
            total_loss = task_loss + split_fn.l2_regularization(params)
        
        Args:
            params: Split parameters.
            weight: Regularization strength. Default 0.01.
            
        Returns:
            L2 penalty (scalar).
        """
        l2 = (
            jnp.sum(params.W_q ** 2) +
            jnp.sum(params.W_k ** 2) +
            jnp.sum(params.W_v ** 2)
        )
        return weight * l2

    def entropy_regularization(
        self,
        params: AttentionSplitParams,
        x: Array,
        weight: float = 0.1,
    ) -> Array:
        """Compute entropy regularization to prevent attention collapse.
        
        Encourages attention to spread across features rather than
        focusing on a single feature (which causes overfitting).
        
        Args:
            params: Split parameters.
            x: Input batch, shape (batch, num_features).
            weight: Regularization strength. Default 0.1.
            
        Returns:
            Negative entropy penalty (scalar). Higher = more spread attention.
        """
        q = x @ params.W_q
        attn_logits = (q @ params.W_k.T) / (jnp.sqrt(self.head_dim) * self.temperature)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        
        # Entropy: -sum(p * log(p)), higher = more uniform
        # We want to maximize entropy, so return negative
        entropy = -jnp.sum(attn_weights * jnp.log(attn_weights + 1e-8), axis=-1)
        mean_entropy = jnp.mean(entropy)
        
        # Return negative because we minimize loss (so minimizing -entropy maximizes entropy)
        return -weight * mean_entropy


class MultiHeadAttentionSplitParams(NamedTuple):
    """Parameters for multi-head attention split.
    
    Attributes:
        W_q: Query projection, shape (num_features, num_heads, head_dim).
        W_k: Key projection, shape (num_features, num_heads, head_dim).
        W_v: Value weights, shape (num_features, num_heads).
        W_o: Output projection, shape (num_heads,).
        threshold: Split threshold, scalar.
    """
    
    W_q: Array  # (num_features, num_heads, head_dim)
    W_k: Array  # (num_features, num_heads, head_dim)
    W_v: Array  # (num_features, num_heads)
    W_o: Array  # (num_heads,)
    threshold: Array  # scalar


class MultiHeadAttentionSplit:
    """Multi-head attention split.
    
    Multiple attention heads capture different feature interaction patterns.
    Use smaller num_heads and head_dim for better generalization.
    """

    def __init__(
        self,
        num_heads: int = 2,  # Reduced from 4
        head_dim: int = 2,   # Reduced from 4
        temperature: float = 2.0,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.temperature = temperature

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.05,
    ) -> MultiHeadAttentionSplitParams:
        """Initialize split parameters."""
        keys = jax.random.split(key, 5)
        
        std = init_scale * jnp.sqrt(2.0 / (num_features + self.head_dim))
        
        W_q = jax.random.normal(keys[0], (num_features, self.num_heads, self.head_dim)) * std
        W_k = jax.random.normal(keys[1], (num_features, self.num_heads, self.head_dim)) * std
        W_v = jax.random.normal(keys[2], (num_features, self.num_heads)) * init_scale
        W_o = jax.random.normal(keys[3], (self.num_heads,)) * init_scale
        threshold = jnp.zeros(())
        
        return MultiHeadAttentionSplitParams(
            W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, threshold=threshold
        )

    def compute_score(self, params: MultiHeadAttentionSplitParams, x: Array) -> Array:
        """Compute split score with multi-head attention."""
        # Query: x @ W_q -> (..., num_heads, head_dim)
        q = jnp.einsum('...f,fhd->...hd', x, params.W_q)
        
        # Attention per head with temperature
        attn_logits = jnp.einsum('...hd,fhd->...hf', q, params.W_k)
        attn_logits = attn_logits / (jnp.sqrt(self.head_dim) * self.temperature)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        
        # Weighted sum per head
        weighted = jnp.einsum('...hf,...f,fh->...h', attn_weights, x, params.W_v)
        
        # Combine heads
        output = jnp.einsum('...h,h->...', weighted, params.W_o)
        
        return output - params.threshold

    def l2_regularization(self, params: MultiHeadAttentionSplitParams, weight: float = 0.01) -> Array:
        """Compute L2 regularization."""
        l2 = (
            jnp.sum(params.W_q ** 2) +
            jnp.sum(params.W_k ** 2) +
            jnp.sum(params.W_v ** 2) +
            jnp.sum(params.W_o ** 2)
        )
        return weight * l2
