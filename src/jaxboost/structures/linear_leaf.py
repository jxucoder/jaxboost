"""Linear Leaf Tree: Trees with linear models in leaves for extrapolation.

Traditional trees output constants in leaves, limiting them to interpolation.
Linear Leaf Trees use linear functions in each leaf, enabling:
1. Extrapolation beyond training data range
2. Better gradient capture within each region
3. Smooth predictions with soft routing

Mathematical formulation:
- Traditional: y = Σ_l p(l|x) · c_l
- Linear Leaf: y = Σ_l p(l|x) · (w_l @ x + b_l)

Where p(l|x) is the soft routing probability to leaf l.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxboost.core.protocols import RoutingFn, SplitFn


class LinearLeafParams(NamedTuple):
    """Parameters for linear leaf tree.

    Attributes:
        split_params: Split parameters, one per depth level.
        leaf_weights: Linear weights for each leaf, shape (2^depth, num_features).
        leaf_biases: Bias terms for each leaf, shape (2^depth,).
    """

    split_params: list[Any]  # length = depth
    leaf_weights: Array  # (2^depth, num_features)
    leaf_biases: Array  # (2^depth,)


class LinearLeafTree:
    """Oblivious tree with linear functions in leaves.

    Enables extrapolation beyond training data range while maintaining
    the efficient structure of oblivious trees.

    Example:
        >>> tree = LinearLeafTree()
        >>> split_fn = HyperplaneSplit()
        >>> params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)
        >>> 
        >>> # Forward pass with soft routing
        >>> routing_fn = lambda s: soft_routing(s, temperature=1.0)
        >>> predictions = tree.forward(params, x, split_fn, routing_fn)
        >>> 
        >>> # Works for extrapolation!
        >>> x_extrapolate = x * 2  # Beyond training range
        >>> pred_extrapolate = tree.forward(params, x_extrapolate, split_fn, routing_fn)
    """

    def __init__(self, l2_leaf_reg: float = 0.0) -> None:
        """Initialize LinearLeafTree.

        Args:
            l2_leaf_reg: L2 regularization strength for leaf weights.
                         Helps prevent overfitting in leaf linear models.
        """
        self.l2_leaf_reg = l2_leaf_reg

    def init_params(
        self,
        key: Array,
        depth: int,
        num_features: int,
        split_fn: SplitFn[Any],
        init_leaf_scale: float = 0.01,
        init_weight_scale: float = 0.001,
    ) -> LinearLeafParams:
        """Initialize tree parameters.

        Args:
            key: JAX PRNG key.
            depth: Tree depth (number of split levels).
            num_features: Number of input features.
            split_fn: Split function to use.
            init_leaf_scale: Scale for leaf bias initialization.
            init_weight_scale: Scale for leaf weight initialization.
                              Smaller than bias scale to start near constant leaves.

        Returns:
            Initialized tree parameters.
        """
        num_leaves = 2**depth
        keys = jax.random.split(key, depth + 2)

        split_params = [
            split_fn.init_params(keys[d], num_features) for d in range(depth)
        ]

        # Initialize weights small so initially behaves like constant leaf
        leaf_weights = (
            jax.random.normal(keys[depth], (num_leaves, num_features)) 
            * init_weight_scale
        )
        leaf_biases = (
            jax.random.normal(keys[depth + 1], (num_leaves,)) 
            * init_leaf_scale
        )

        return LinearLeafParams(
            split_params=split_params,
            leaf_weights=leaf_weights,
            leaf_biases=leaf_biases,
        )

    def forward(
        self,
        params: LinearLeafParams,
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

        # Compute routing probabilities
        p_rights = []
        for d in range(depth):
            score = split_fn.compute_score(params.split_params[d], x)
            p_rights.append(routing_fn(score))
        p_rights = jnp.stack(p_rights, axis=0)  # (depth, batch)

        # Get leaf probabilities
        leaf_probs = self._compute_leaf_probs(p_rights, depth)  # (batch, num_leaves)

        # Compute linear predictions for each leaf
        # leaf_weights: (num_leaves, num_features)
        # x: (batch, num_features)
        # Result: (batch, num_leaves)
        leaf_linear = jnp.einsum('lf,bf->bl', params.leaf_weights, x)
        leaf_predictions = leaf_linear + params.leaf_biases  # (batch, num_leaves)

        # Weighted sum over leaves
        output = jnp.sum(leaf_probs * leaf_predictions, axis=-1)  # (batch,)

        return output[0] if single_sample else output

    def forward_with_leaf_outputs(
        self,
        params: LinearLeafParams,
        x: Array,
        split_fn: SplitFn[Any],
        routing_fn: RoutingFn,
    ) -> tuple[Array, Array, Array]:
        """Forward pass returning intermediate values for analysis.

        Args:
            params: Tree parameters.
            x: Input features, shape (batch, num_features).
            split_fn: Split function.
            routing_fn: Routing function.

        Returns:
            Tuple of:
            - predictions: Final predictions, shape (batch,)
            - leaf_probs: Routing probabilities, shape (batch, num_leaves)
            - leaf_predictions: Per-leaf predictions, shape (batch, num_leaves)
        """
        single_sample = x.ndim == 1
        if single_sample:
            x = x[None, :]

        depth = len(params.split_params)

        p_rights = []
        for d in range(depth):
            score = split_fn.compute_score(params.split_params[d], x)
            p_rights.append(routing_fn(score))
        p_rights = jnp.stack(p_rights, axis=0)

        leaf_probs = self._compute_leaf_probs(p_rights, depth)
        leaf_linear = jnp.einsum('lf,bf->bl', params.leaf_weights, x)
        leaf_predictions = leaf_linear + params.leaf_biases
        output = jnp.sum(leaf_probs * leaf_predictions, axis=-1)

        if single_sample:
            return output[0], leaf_probs[0], leaf_predictions[0]
        return output, leaf_probs, leaf_predictions

    def _compute_leaf_probs(self, p_rights: Array, depth: int) -> Array:
        """Compute probability of reaching each leaf (fully vectorized).

        Uses bit manipulation + broadcasting - no Python loops!

        Each leaf index's binary representation encodes the path:
        bit d = 0 means go left, bit d = 1 means go right at depth d.
        """
        num_leaves = 2**depth

        leaf_indices = jnp.arange(num_leaves)
        depth_indices = jnp.arange(depth)

        bit_mask = (leaf_indices[:, None] >> depth_indices) & 1  # (num_leaves, depth)

        p_rights_T = p_rights.T  # (batch, depth)
        p_rights_expanded = p_rights_T[:, None, :]  # (batch, 1, depth)
        bit_mask_expanded = bit_mask[None, :, :]  # (1, num_leaves, depth)

        routing_probs = (
            bit_mask_expanded * p_rights_expanded
            + (1 - bit_mask_expanded) * (1 - p_rights_expanded)
        )

        leaf_probs = jnp.prod(routing_probs, axis=-1)  # (batch, num_leaves)

        return leaf_probs

    def compute_leaf_reg_loss(self, params: LinearLeafParams) -> Array:
        """Compute L2 regularization loss on leaf weights.

        Args:
            params: Tree parameters.

        Returns:
            Regularization loss (scalar).
        """
        return self.l2_leaf_reg * jnp.sum(params.leaf_weights ** 2)

    def get_effective_leaf_slopes(
        self,
        params: LinearLeafParams,
        feature_idx: int,
    ) -> Array:
        """Get the slope for a specific feature across all leaves.

        Useful for understanding how each leaf extrapolates.

        Args:
            params: Tree parameters.
            feature_idx: Index of feature to analyze.

        Returns:
            Slopes for the feature in each leaf, shape (num_leaves,).
        """
        return params.leaf_weights[:, feature_idx]


class LinearLeafEnsemble:
    """Ensemble of Linear Leaf Trees (like a mini gradient boosting).

    Combines multiple linear leaf trees for better performance.
    """

    def __init__(
        self,
        n_trees: int = 10,
        depth: int = 4,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 0.01,
    ) -> None:
        """Initialize ensemble.

        Args:
            n_trees: Number of trees in the ensemble.
            depth: Depth of each tree.
            learning_rate: Shrinkage factor for each tree's contribution.
            l2_leaf_reg: L2 regularization for leaf weights.
        """
        self.n_trees = n_trees
        self.depth = depth
        self.learning_rate = learning_rate
        self.tree = LinearLeafTree(l2_leaf_reg=l2_leaf_reg)

    def init_params(
        self,
        key: Array,
        num_features: int,
        split_fn: SplitFn[Any],
    ) -> list[LinearLeafParams]:
        """Initialize parameters for all trees.

        Args:
            key: JAX PRNG key.
            num_features: Number of input features.
            split_fn: Split function to use.

        Returns:
            List of tree parameters.
        """
        keys = jax.random.split(key, self.n_trees)
        return [
            self.tree.init_params(keys[i], self.depth, num_features, split_fn)
            for i in range(self.n_trees)
        ]

    def forward(
        self,
        params_list: list[LinearLeafParams],
        x: Array,
        split_fn: SplitFn[Any],
        routing_fn: RoutingFn,
    ) -> Array:
        """Forward pass through all trees.

        Args:
            params_list: List of tree parameters.
            x: Input features.
            split_fn: Split function.
            routing_fn: Routing function.

        Returns:
            Ensemble predictions.
        """
        predictions = jnp.zeros(x.shape[0] if x.ndim > 1 else 1)

        for params in params_list:
            tree_pred = self.tree.forward(params, x, split_fn, routing_fn)
            predictions = predictions + self.learning_rate * tree_pred

        return predictions

