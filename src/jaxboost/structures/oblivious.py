"""Oblivious tree structure.

An oblivious tree uses the same split at each depth level.
This makes it highly efficient for GPU computation.

Key property: For depth D, there are exactly 2^D leaves.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxboost.core.protocols import RoutingFn, SplitFn


class ObliviousTreeParams(NamedTuple):
    """Parameters for an oblivious tree.

    Attributes:
        split_params: Split parameters, one per depth level.
        leaf_values: Prediction values at leaves, shape (2^depth,).
    """

    split_params: list[Any]  # length = depth
    leaf_values: Array  # (2^depth,)


class ObliviousTree:
    """Oblivious (symmetric) tree structure.

    All nodes at the same depth share the same split.
    This enables efficient vectorized computation on GPU.
    """

    def init_params(
        self,
        key: Array,
        depth: int,
        num_features: int,
        split_fn: SplitFn[Any],
        init_leaf_scale: float = 0.01,
    ) -> ObliviousTreeParams:
        """Initialize tree parameters.

        Args:
            key: JAX PRNG key.
            depth: Tree depth (number of split levels).
            num_features: Number of input features.
            split_fn: Split function to use.
            init_leaf_scale: Scale for leaf value initialization.

        Returns:
            Initialized tree parameters.
        """
        num_leaves = 2**depth
        keys = jax.random.split(key, depth + 1)

        split_params = [
            split_fn.init_params(keys[d], num_features) for d in range(depth)
        ]
        leaf_values = jax.random.normal(keys[depth], (num_leaves,)) * init_leaf_scale

        return ObliviousTreeParams(split_params=split_params, leaf_values=leaf_values)

    def forward(
        self,
        params: ObliviousTreeParams,
        x: Array,
        split_fn: SplitFn[Any],
        routing_fn: RoutingFn,
    ) -> Array:
        """Forward pass through the tree.

        Args:
            params: Tree parameters.
            x: Input features, shape (batch, num_features) or (num_features,).
            split_fn: Split function.
            routing_fn: Routing function.

        Returns:
            Predictions, shape (batch,) or scalar.
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x[None, :]

        depth = len(params.split_params)
        p_rights = []
        for d in range(depth):
            score = split_fn.compute_score(params.split_params[d], x)
            p_rights.append(routing_fn(score))
        p_rights = jnp.stack(p_rights, axis=0)  # (depth, batch)

        leaf_probs = self._compute_leaf_probs(p_rights, depth)
        output = jnp.sum(leaf_probs * params.leaf_values, axis=-1)

        return output[0] if single_sample else output

    def _compute_leaf_probs(self, p_rights: Array, depth: int) -> Array:
        """Compute probability of reaching each leaf (fully vectorized).

        Uses bit manipulation + broadcasting - no Python loops!
        
        Each leaf index's binary representation encodes the path:
        bit d = 0 means go left, bit d = 1 means go right at depth d.
        """
        num_leaves = 2**depth

        # Create binary mask: which way to go at each depth for each leaf
        leaf_indices = jnp.arange(num_leaves)
        depth_indices = jnp.arange(depth)

        # bit_mask[leaf, depth] = 1 if leaf goes right at depth, else 0
        bit_mask = (leaf_indices[:, None] >> depth_indices) & 1  # (num_leaves, depth)

        # p_rights: (depth, batch) -> (batch, depth)
        p_rights_T = p_rights.T  # (batch, depth)

        # Broadcast: (batch, 1, depth) vs (1, num_leaves, depth)
        p_rights_expanded = p_rights_T[:, None, :]  # (batch, 1, depth)
        bit_mask_expanded = bit_mask[None, :, :]  # (1, num_leaves, depth)

        # routing_prob = p_right if go_right else (1 - p_right)
        routing_probs = (
            bit_mask_expanded * p_rights_expanded
            + (1 - bit_mask_expanded) * (1 - p_rights_expanded)
        )  # (batch, num_leaves, depth)

        # Product over depth dimension
        leaf_probs = jnp.prod(routing_probs, axis=-1)  # (batch, num_leaves)

        return leaf_probs
