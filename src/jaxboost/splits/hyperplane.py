"""Hyperplane split function.

Instead of axis-aligned splits (x[j] <= t), hyperplane splits use
a learned linear combination: w @ x <= t

This captures feature interactions directly in a single split!
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class HyperplaneSplitParams(NamedTuple):
    """Parameters for a hyperplane split.

    Attributes:
        weights: Linear combination weights, shape (num_features,).
        threshold: Split threshold, scalar.
    """

    weights: Array  # (num_features,)
    threshold: Array  # scalar


class HyperplaneSplit:
    """Hyperplane split: w @ x - threshold.
    
    More expressive than axis-aligned splits.
    A single hyperplane split can capture linear feature interactions.
    """

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> HyperplaneSplitParams:
        """Initialize split parameters.
        
        Weights are initialized small and normalized.
        """
        keys = jax.random.split(key, 2)
        weights = jax.random.normal(keys[0], (num_features,)) * init_scale
        weights = weights / (jnp.linalg.norm(weights) + 1e-8)
        threshold = jax.random.normal(keys[1], ()) * 0.1
        return HyperplaneSplitParams(weights=weights, threshold=threshold)

    def compute_score(self, params: HyperplaneSplitParams, x: Array) -> Array:
        """Compute split score: w @ x - threshold.
        
        Positive → go right, Negative → go left.
        """
        # x: (..., num_features), weights: (num_features,)
        projection = jnp.sum(x * params.weights, axis=-1)
        return projection - params.threshold

