"""Sparse hyperplane split function.

Learns sparse split directions via L1 regularization or learnable gates.
Only a few features participate in each split → interpretable interactions.

Example:
    "Split on: 0.6*age + 0.4*income > 0.3" (only 2 features, not all 100)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class SparseHyperplaneSplitParams(NamedTuple):
    """Parameters for sparse hyperplane split.

    Attributes:
        weights: Raw weight values, shape (num_features,).
        gates: Learnable sparsity gates (pre-sigmoid), shape (num_features,).
        threshold: Split threshold, scalar.
    """

    weights: Array  # (num_features,)
    gates: Array  # (num_features,) - sigmoid(gates) determines sparsity
    threshold: Array  # scalar


class SparseHyperplaneSplit:
    """Sparse hyperplane split with learnable feature selection.

    Uses soft gates to select which features participate in each split.
    The effective weight is: w_eff = w * sigmoid(gate * temperature)

    At low temperature: gates → hard 0/1 selection
    At high temperature: gates → soft selection (differentiable)

    Example:
        >>> split_fn = SparseHyperplaneSplit(target_sparsity=0.2)
        >>> params = split_fn.init_params(key, num_features=100)
        >>> score = split_fn.compute_score(params, x)
        >>> 
        >>> # Add sparsity regularization to loss:
        >>> reg = split_fn.sparsity_regularization(params, weight=0.1)
    """

    def __init__(
        self,
        target_sparsity: float = 0.2,
        gate_temperature: float = 1.0,
    ) -> None:
        """Initialize sparse hyperplane split.

        Args:
            target_sparsity: Target fraction of features to use (0.2 = 20%).
            gate_temperature: Temperature for gate sigmoid. Lower = sharper.
        """
        self.target_sparsity = target_sparsity
        self.gate_temperature = gate_temperature

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> SparseHyperplaneSplitParams:
        """Initialize split parameters.

        Gates are initialized to achieve target_sparsity on average.
        """
        keys = jax.random.split(key, 3)

        # Initialize weights small
        weights = jax.random.normal(keys[0], (num_features,)) * init_scale

        # Initialize gates so sigmoid(gate) ≈ target_sparsity on average
        # sigmoid(x) = target_sparsity → x = log(target_sparsity / (1 - target_sparsity))
        gate_bias = jnp.log(self.target_sparsity / (1 - self.target_sparsity + 1e-8))
        gates = jax.random.normal(keys[1], (num_features,)) * 0.5 + gate_bias

        threshold = jax.random.normal(keys[2], ()) * 0.1

        return SparseHyperplaneSplitParams(
            weights=weights, gates=gates, threshold=threshold
        )

    def compute_score(self, params: SparseHyperplaneSplitParams, x: Array) -> Array:
        """Compute split score with sparse weights.

        Args:
            params: Split parameters.
            x: Input features, shape (..., num_features).

        Returns:
            Split scores, shape (...).
        """
        # Soft gates: values in (0, 1)
        gate_probs = jax.nn.sigmoid(params.gates * self.gate_temperature)

        # Effective weights = raw weights * gate probability
        effective_weights = params.weights * gate_probs

        # Normalize for stability
        norm = jnp.linalg.norm(effective_weights) + 1e-8
        effective_weights = effective_weights / norm

        # Linear projection
        projection = jnp.sum(x * effective_weights, axis=-1)

        return projection - params.threshold

    def get_active_features(
        self, params: SparseHyperplaneSplitParams, threshold: float = 0.5
    ) -> Array:
        """Get indices of active features (for interpretability).

        Args:
            params: Split parameters.
            threshold: Gate probability threshold for "active".

        Returns:
            Boolean mask of active features, shape (num_features,).
        """
        gate_probs = jax.nn.sigmoid(params.gates * self.gate_temperature)
        return gate_probs > threshold

    def get_feature_importance(self, params: SparseHyperplaneSplitParams) -> Array:
        """Get importance score for each feature.

        Returns:
            Importance scores (|weight| * gate_prob), shape (num_features,).
        """
        gate_probs = jax.nn.sigmoid(params.gates * self.gate_temperature)
        return jnp.abs(params.weights) * gate_probs

    def sparsity_regularization(
        self,
        params: SparseHyperplaneSplitParams,
        weight: float = 0.1,
    ) -> Array:
        """L1-style regularization to encourage sparsity.

        Penalizes gate probabilities to encourage them toward 0.

        Args:
            params: Split parameters.
            weight: Regularization strength.

        Returns:
            Sparsity penalty (scalar).
        """
        gate_probs = jax.nn.sigmoid(params.gates * self.gate_temperature)
        # L1 on gate probabilities encourages sparsity
        l1_gates = jnp.mean(gate_probs)
        return weight * l1_gates

    def l2_regularization(
        self,
        params: SparseHyperplaneSplitParams,
        weight: float = 0.01,
    ) -> Array:
        """L2 regularization on weights.

        Args:
            params: Split parameters.
            weight: Regularization strength.

        Returns:
            L2 penalty (scalar).
        """
        return weight * jnp.sum(params.weights**2)

    def entropy_regularization(
        self,
        params: SparseHyperplaneSplitParams,
        weight: float = 0.01,
    ) -> Array:
        """Entropy regularization to push gates toward 0 or 1.

        Encourages hard selection (low entropy = more binary gates).

        Args:
            params: Split parameters.
            weight: Regularization strength.

        Returns:
            Negative entropy penalty (scalar).
        """
        gate_probs = jax.nn.sigmoid(params.gates * self.gate_temperature)
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        # We want to minimize this to push toward 0 or 1
        eps = 1e-8
        entropy = -(
            gate_probs * jnp.log(gate_probs + eps)
            + (1 - gate_probs) * jnp.log(1 - gate_probs + eps)
        )
        return weight * jnp.mean(entropy)


class TopKHyperplaneSplitParams(NamedTuple):
    """Parameters for top-k hyperplane split.

    Attributes:
        weights: Weight values, shape (num_features,).
        threshold: Split threshold, scalar.
    """

    weights: Array  # (num_features,)
    threshold: Array  # scalar


class TopKHyperplaneSplit:
    """Hyperplane split using only top-k features by weight magnitude.

    Hard sparsity constraint: exactly k features per split.
    Uses straight-through estimator for gradients.

    Example:
        >>> split_fn = TopKHyperplaneSplit(k=5)  # Use only 5 features
        >>> params = split_fn.init_params(key, num_features=100)
        >>> score = split_fn.compute_score(params, x)
    """

    def __init__(self, k: int = 5) -> None:
        """Initialize top-k hyperplane split.

        Args:
            k: Number of features to use per split.
        """
        self.k = k

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> TopKHyperplaneSplitParams:
        """Initialize split parameters."""
        keys = jax.random.split(key, 2)
        weights = jax.random.normal(keys[0], (num_features,)) * init_scale
        threshold = jax.random.normal(keys[1], ()) * 0.1
        return TopKHyperplaneSplitParams(weights=weights, threshold=threshold)

    def compute_score(self, params: TopKHyperplaneSplitParams, x: Array) -> Array:
        """Compute split score using only top-k features.

        Uses straight-through estimator: forward uses hard top-k mask,
        backward uses soft gradients through all weights.

        Args:
            params: Split parameters.
            x: Input features, shape (..., num_features).

        Returns:
            Split scores, shape (...).
        """
        # Get top-k mask based on weight magnitude
        abs_weights = jnp.abs(params.weights)
        k = min(self.k, len(params.weights))

        # Top-k indices
        top_k_vals, _ = jax.lax.top_k(abs_weights, k)
        threshold_val = top_k_vals[-1]

        # Hard mask (non-differentiable)
        hard_mask = (abs_weights >= threshold_val).astype(jnp.float32)

        # Straight-through estimator: use hard mask forward, soft gradient backward
        # soft_mask passes gradients, hard_mask is used in forward
        soft_mask = jax.nn.sigmoid((abs_weights - threshold_val) * 10.0)
        mask = hard_mask + soft_mask - jax.lax.stop_gradient(soft_mask)

        # Apply mask to weights
        sparse_weights = params.weights * mask
        norm = jnp.linalg.norm(sparse_weights) + 1e-8
        sparse_weights = sparse_weights / norm

        # Linear projection
        projection = jnp.sum(x * sparse_weights, axis=-1)

        return projection - params.threshold

    def get_active_features(self, params: TopKHyperplaneSplitParams) -> Array:
        """Get mask of active (top-k) features.

        Returns:
            Boolean mask, shape (num_features,).
        """
        abs_weights = jnp.abs(params.weights)
        k = min(self.k, len(params.weights))
        top_k_vals, _ = jax.lax.top_k(abs_weights, k)
        threshold_val = top_k_vals[-1]
        return abs_weights >= threshold_val

    def get_feature_importance(self, params: TopKHyperplaneSplitParams) -> Array:
        """Get importance scores (masked absolute weights).

        Returns:
            Importance scores, shape (num_features,).
        """
        mask = self.get_active_features(params)
        return jnp.abs(params.weights) * mask

    def l2_regularization(
        self,
        params: TopKHyperplaneSplitParams,
        weight: float = 0.01,
    ) -> Array:
        """L2 regularization on weights."""
        return weight * jnp.sum(params.weights**2)

