"""Boosting aggregation.

Combines tree predictions additively: f(x) = Σ weight_t * tree_t(x)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def boosting_aggregate(
    tree_predictions: Array,
    weights: Array | None = None,
) -> Array:
    """Aggregate tree predictions using boosting (weighted sum).

    Args:
        tree_predictions: Predictions from each tree,
            shape (num_trees, batch) or (num_trees,).
        weights: Optional weights for each tree, shape (num_trees,).
            If None, uniform weights (1.0) are used.

    Returns:
        Aggregated predictions, shape (batch,) or scalar.
    """
    if weights is None:
        return jnp.sum(tree_predictions, axis=0)
    else:
        return jnp.einsum("t,t...->...", weights, tree_predictions)
