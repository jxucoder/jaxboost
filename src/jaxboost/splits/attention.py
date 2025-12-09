"""Attention-based split function.

Uses input-dependent attention to dynamically select features for each sample.
Different samples attend to different features based on their values.
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
    """Input-dependent attention split.
    
    For each sample:
    1. Generate query from input: q = x @ W_q
    2. Compute attention over features: attn = softmax(q @ W_k.T / sqrt(d))
    3. Aggregate: output = sum(x * attn * W_v)
    
    Key insight: each sample gets DIFFERENT attention weights based on its
    feature values, enabling dynamic feature selection.
    """

    def __init__(self, head_dim: int = 8) -> None:
        self.head_dim = head_dim

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> AttentionSplitParams:
        """Initialize split parameters."""
        keys = jax.random.split(key, 4)
        
        W_q = jax.random.normal(keys[0], (num_features, self.head_dim)) * init_scale
        W_k = jax.random.normal(keys[1], (num_features, self.head_dim)) * init_scale
        W_v = jax.random.normal(keys[2], (num_features,)) * init_scale
        threshold = jax.random.normal(keys[3], ()) * 0.1
        
        return AttentionSplitParams(W_q=W_q, W_k=W_k, W_v=W_v, threshold=threshold)

    def compute_score(self, params: AttentionSplitParams, x: Array) -> Array:
        """Compute split score with input-dependent attention.
        
        Args:
            params: Split parameters.
            x: Input features, shape (..., num_features).
            
        Returns:
            Split scores, shape (...).
        """
        # x: (..., num_features)
        # Query from input: each sample generates its own query
        q = x @ params.W_q  # (..., head_dim)
        
        # Keys are learned per feature position
        k = params.W_k  # (num_features, head_dim)
        
        # Attention: q @ k.T / sqrt(d) -> (..., num_features)
        attn_logits = (q @ k.T) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (..., num_features)
        
        # Weighted aggregation: attention * values * input
        weighted = x * attn_weights * params.W_v
        aggregated = jnp.sum(weighted, axis=-1)
        
        return aggregated - params.threshold


class MultiHeadAttentionSplitParams(NamedTuple):
    """Parameters for multi-head attention split."""
    
    W_q: Array  # (num_features, num_heads, head_dim)
    W_k: Array  # (num_features, num_heads, head_dim)
    W_v: Array  # (num_features, num_heads)
    W_o: Array  # (num_heads,) - combine heads
    threshold: Array  # scalar


class MultiHeadAttentionSplit:
    """Multi-head input-dependent attention split.
    
    Multiple attention heads can capture different feature interaction patterns.
    """

    def __init__(self, num_heads: int = 4, head_dim: int = 4) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> MultiHeadAttentionSplitParams:
        """Initialize split parameters."""
        keys = jax.random.split(key, 5)
        
        W_q = jax.random.normal(keys[0], (num_features, self.num_heads, self.head_dim)) * init_scale
        W_k = jax.random.normal(keys[1], (num_features, self.num_heads, self.head_dim)) * init_scale
        W_v = jax.random.normal(keys[2], (num_features, self.num_heads)) * init_scale
        W_o = jax.random.normal(keys[3], (self.num_heads,)) * init_scale
        threshold = jax.random.normal(keys[4], ()) * 0.1
        
        return MultiHeadAttentionSplitParams(
            W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, threshold=threshold
        )

    def compute_score(self, params: MultiHeadAttentionSplitParams, x: Array) -> Array:
        """Compute split score with multi-head attention."""
        # x: (..., num_features)
        batch_shape = x.shape[:-1]
        num_features = x.shape[-1]
        
        # Query: x @ W_q -> (..., num_heads, head_dim)
        q = jnp.einsum('...f,fhd->...hd', x, params.W_q)
        
        # Keys: (num_features, num_heads, head_dim)
        k = params.W_k
        
        # Attention per head: (..., num_heads, num_features)
        attn_logits = jnp.einsum('...hd,fhd->...hf', q, k) / jnp.sqrt(self.head_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        
        # Values: (num_features, num_heads)
        # Weighted sum: (..., num_heads)
        weighted = jnp.einsum('...hf,...f,fh->...h', attn_weights, x, params.W_v)
        
        # Combine heads: (...,)
        output = jnp.einsum('...h,h->...', weighted, params.W_o)
        
        return output - params.threshold
