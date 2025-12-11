"""Interaction Discovery Split.

Learns WHICH features interact, not just interaction weights.
Outputs interpretable interaction structure: "features (0,3,7) interact"

Key insight: Don't enumerate all possible interactions (exponential).
Instead, learn feature embeddings where dot product = interaction strength.

Architecture:
1. Embed each feature into a low-dim space
2. Interaction strength = similarity in embedding space
3. Select top-k interactions (differentiable)
4. Compute explicit interaction features: x_i * x_j [* x_k]
5. Route through tree based on interaction features

This is novel because:
- Learns interaction STRUCTURE, not just weights
- Interpretable: outputs which feature subsets matter
- Efficient: O(d * embed_dim) not O(d^k)
- Higher-order: supports 2-way, 3-way, ... interactions
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class InteractionDiscoveryParams(NamedTuple):
    """Parameters for interaction discovery split.

    Attributes:
        feature_embeddings: Learned embeddings, shape (num_features, embed_dim).
        interaction_bias: Bias for each candidate interaction, shape (num_candidates,).
        output_weights: Weights to combine interaction features, shape (num_interactions,).
        threshold: Split threshold, scalar.
    """

    feature_embeddings: Array  # (num_features, embed_dim)
    interaction_bias: Array  # (num_candidates,)
    output_weights: Array  # (num_interactions,)
    threshold: Array  # scalar


class InteractionDiscoverySplit:
    """Split function that learns interaction structure.

    Instead of using all features or fixed interactions, this learns
    WHICH feature combinations matter for the split decision.

    The key insight: feature embeddings define interaction structure.
    Features with similar embeddings will interact strongly.

    Example:
        >>> split_fn = InteractionDiscoverySplit(
        ...     max_interactions=10,
        ...     max_order=3,  # Up to 3-way interactions
        ... )
        >>> params = split_fn.init_params(key, num_features=50)
        >>> score = split_fn.compute_score(params, x)
        >>>
        >>> # Get interpretable structure
        >>> interactions = split_fn.get_top_interactions(params, k=5)
        >>> # Returns: [(0, 3), (1, 7, 12), (4, 5), ...]
    """

    def __init__(
        self,
        max_interactions: int = 16,
        max_order: int = 2,
        embed_dim: int = 8,
        temperature: float = 1.0,
        num_candidate_samples: int = 64,
    ) -> None:
        """Initialize interaction discovery split.

        Args:
            max_interactions: Number of interactions to use in split.
            max_order: Maximum interaction order (2=pairwise, 3=triplets, etc).
            embed_dim: Dimension of feature embeddings.
            temperature: Softmax temperature for interaction selection.
            num_candidate_samples: Number of candidate interactions to consider.
        """
        self.max_interactions = max_interactions
        self.max_order = max_order
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.num_candidate_samples = num_candidate_samples

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> InteractionDiscoveryParams:
        """Initialize parameters.

        Feature embeddings are initialized so that initially,
        nearby features (by index) have similar embeddings.
        """
        keys = jax.random.split(key, 4)

        # Feature embeddings with structure-aware initialization
        # Add positional component so nearby features start similar
        random_embed = jax.random.normal(keys[0], (num_features, self.embed_dim)) * init_scale
        positional = jnp.sin(
            jnp.arange(num_features)[:, None] * jnp.arange(self.embed_dim)[None, :] / num_features
        ) * 0.1
        feature_embeddings = random_embed + positional

        # Bias for candidate interactions (learned)
        interaction_bias = jax.random.normal(keys[1], (self.num_candidate_samples,)) * 0.01

        # Output weights for combining interactions
        output_weights = jax.random.normal(keys[2], (self.max_interactions,)) * init_scale

        threshold = jnp.zeros(())

        return InteractionDiscoveryParams(
            feature_embeddings=feature_embeddings,
            interaction_bias=interaction_bias,
            output_weights=output_weights,
            threshold=threshold,
        )

    def _generate_candidate_interactions(
        self,
        key: Array,
        num_features: int,
        embeddings: Array,
    ) -> tuple[Array, Array]:
        """Generate candidate interactions based on embedding similarity.

        Returns indices of candidate feature combinations and their scores.
        Uses embedding dot products to find likely interactions.

        Args:
            key: PRNG key.
            num_features: Number of features.
            embeddings: Feature embeddings, shape (num_features, embed_dim).

        Returns:
            Tuple of:
                - candidate_indices: shape (num_candidates, max_order), padded with -1
                - candidate_scores: shape (num_candidates,)
        """
        # Compute pairwise similarity
        similarity = embeddings @ embeddings.T  # (num_features, num_features)

        # Zero out diagonal (no self-interactions)
        similarity = similarity - jnp.eye(num_features) * 1e9

        # For pairwise: take top similarities
        if self.max_order == 2:
            # Flatten upper triangle
            triu_indices = jnp.triu_indices(num_features, k=1)
            pair_scores = similarity[triu_indices]

            # Get top candidates
            num_pairs = len(pair_scores)
            k = min(self.num_candidate_samples, num_pairs)
            top_scores, top_indices = jax.lax.top_k(pair_scores, k)

            # Convert flat indices back to (i, j) pairs
            i_indices = triu_indices[0][top_indices]
            j_indices = triu_indices[1][top_indices]

            # Stack into candidate array, pad to max_order
            candidate_indices = jnp.stack([i_indices, j_indices], axis=1)
            # Pad with -1 for unused order dimensions
            padding = jnp.full((k, self.max_order - 2), -1)
            candidate_indices = jnp.concatenate([candidate_indices, padding], axis=1)

            return candidate_indices, top_scores

        else:
            # For higher-order: sample based on pairwise scores
            # Start with strong pairs, extend to triplets/etc
            keys = jax.random.split(key, 3)

            # Get top pairs as seeds
            triu_indices = jnp.triu_indices(num_features, k=1)
            pair_scores = similarity[triu_indices]
            num_seeds = min(self.num_candidate_samples // 2, len(pair_scores))
            _, top_pair_idx = jax.lax.top_k(pair_scores, num_seeds)

            i_indices = triu_indices[0][top_pair_idx]
            j_indices = triu_indices[1][top_pair_idx]

            # For each pair, find best extending features
            candidates = []
            scores = []

            # Pairwise candidates
            for idx in range(num_seeds):
                i, j = i_indices[idx], j_indices[idx]
                candidates.append(
                    jnp.array([i, j] + [-1] * (self.max_order - 2))
                )
                scores.append(similarity[i, j])

                # Extend to triplet if max_order >= 3
                if self.max_order >= 3:
                    # Score for adding feature k: avg similarity to i and j
                    extend_scores = (similarity[i, :] + similarity[j, :]) / 2
                    extend_scores = extend_scores.at[i].set(-1e9)
                    extend_scores = extend_scores.at[j].set(-1e9)

                    # Best extension
                    k = jnp.argmax(extend_scores)
                    triplet_score = similarity[i, j] + extend_scores[k]

                    candidates.append(
                        jnp.array([i, j, k] + [-1] * (self.max_order - 3))
                    )
                    scores.append(triplet_score)

            # Stack and truncate to num_candidates
            candidate_indices = jnp.stack(candidates[: self.num_candidate_samples])
            candidate_scores = jnp.array(scores[: self.num_candidate_samples])

            return candidate_indices, candidate_scores

    def _compute_interaction_features(
        self,
        x: Array,
        indices: Array,
        weights: Array,
    ) -> Array:
        """Compute weighted interaction features.

        Args:
            x: Input, shape (batch, num_features).
            indices: Interaction indices, shape (num_interactions, max_order).
            weights: Interaction weights, shape (num_interactions,).

        Returns:
            Interaction features, shape (batch, num_interactions).
        """
        batch_size = x.shape[0]
        num_interactions = indices.shape[0]

        def compute_single_interaction(idx_row):
            """Compute one interaction: product of selected features."""
            # idx_row: (max_order,), values are feature indices or -1
            result = jnp.ones(batch_size)
            for order in range(self.max_order):
                feat_idx = idx_row[order]
                # Use where to handle -1 (padding)
                feat_val = jnp.where(
                    feat_idx >= 0,
                    x[:, jnp.clip(feat_idx, 0, x.shape[1] - 1)],
                    jnp.ones(batch_size),
                )
                result = result * feat_val
            return result

        # Vectorize over interactions
        interactions = jax.vmap(compute_single_interaction)(indices)  # (num_interactions, batch)
        interactions = interactions.T  # (batch, num_interactions)

        # Weight by learned importance
        weighted = interactions * weights

        return weighted

    def compute_score(
        self,
        params: InteractionDiscoveryParams,
        x: Array,
        key: Array | None = None,
    ) -> Array:
        """Compute split score based on learned interactions.

        Args:
            params: Split parameters.
            x: Input features, shape (..., num_features).
            key: Optional PRNG key for candidate sampling.

        Returns:
            Split scores, shape (...).
        """
        # Handle batched input
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        batch_size, num_features = x_flat.shape

        # Use fixed key if not provided (deterministic at inference)
        if key is None:
            key = jax.random.PRNGKey(0)

        # Generate candidates based on learned embeddings
        candidate_indices, candidate_scores = self._generate_candidate_interactions(
            key, num_features, params.feature_embeddings
        )

        # Add learned bias to candidate scores
        candidate_scores = candidate_scores + params.interaction_bias[: len(candidate_scores)]

        # Soft top-k selection
        selection_logits = candidate_scores / self.temperature
        selection_probs = jax.nn.softmax(selection_logits)

        # Take top-k interactions
        k = min(self.max_interactions, len(candidate_scores))
        top_probs, top_idx = jax.lax.top_k(selection_probs, k)
        selected_indices = candidate_indices[top_idx]

        # Normalize selection weights
        selection_weights = top_probs / (jnp.sum(top_probs) + 1e-8)

        # Combine with output weights
        combined_weights = selection_weights * params.output_weights[:k]

        # Compute interaction features
        interaction_features = self._compute_interaction_features(
            x_flat, selected_indices, combined_weights
        )

        # Aggregate to single score
        score = jnp.sum(interaction_features, axis=-1)
        score = score - params.threshold

        return score.reshape(original_shape) if original_shape else score

    def get_top_interactions(
        self,
        params: InteractionDiscoveryParams,
        num_features: int,
        k: int = 10,
    ) -> list[tuple[int, ...]]:
        """Get the top-k learned interactions for interpretability.

        Args:
            params: Learned parameters.
            num_features: Number of features in the data.
            k: Number of top interactions to return.

        Returns:
            List of feature index tuples representing interactions.
            E.g., [(0, 3), (1, 7, 12), (4, 5)] means:
                - Features 0 and 3 interact
                - Features 1, 7, and 12 have a 3-way interaction
                - Features 4 and 5 interact
        """
        key = jax.random.PRNGKey(0)
        candidate_indices, candidate_scores = self._generate_candidate_interactions(
            key, num_features, params.feature_embeddings
        )

        # Add bias
        candidate_scores = candidate_scores + params.interaction_bias[: len(candidate_scores)]

        # Get top-k
        k = min(k, len(candidate_scores))
        _, top_idx = jax.lax.top_k(candidate_scores, k)

        # Convert to Python tuples
        interactions = []
        for idx in top_idx:
            indices = candidate_indices[idx]
            # Filter out padding (-1)
            valid_indices = tuple(int(i) for i in indices if i >= 0)
            interactions.append(valid_indices)

        return interactions

    def get_interaction_strength(
        self,
        params: InteractionDiscoveryParams,
    ) -> Array:
        """Get pairwise interaction strength matrix for visualization.

        Returns:
            Interaction strength matrix, shape (num_features, num_features).
            Higher values = features interact more strongly.
        """
        # Interaction strength = embedding similarity
        similarity = params.feature_embeddings @ params.feature_embeddings.T
        # Zero diagonal
        similarity = similarity - jnp.diag(jnp.diag(similarity))
        return similarity

    def l2_regularization(
        self,
        params: InteractionDiscoveryParams,
        weight: float = 0.01,
    ) -> Array:
        """L2 regularization on embeddings and weights."""
        l2 = (
            jnp.sum(params.feature_embeddings**2)
            + jnp.sum(params.output_weights**2)
            + jnp.sum(params.interaction_bias**2)
        )
        return weight * l2

    def sparsity_regularization(
        self,
        params: InteractionDiscoveryParams,
        weight: float = 0.1,
    ) -> Array:
        """Encourage sparse interaction structure.

        Pushes embeddings to be dissimilar (fewer interactions).
        """
        similarity = params.feature_embeddings @ params.feature_embeddings.T
        # Penalize high off-diagonal similarity
        off_diag = similarity - jnp.diag(jnp.diag(similarity))
        return weight * jnp.mean(jnp.abs(off_diag))

    def diversity_regularization(
        self,
        params: InteractionDiscoveryParams,
        weight: float = 0.1,
    ) -> Array:
        """Encourage diverse interactions (not all using same features).

        Maximizes entropy of feature usage across interactions.
        """
        # Compute how much each feature is used
        similarity = jnp.abs(params.feature_embeddings @ params.feature_embeddings.T)
        feature_usage = jnp.sum(similarity, axis=1)
        feature_usage = feature_usage / (jnp.sum(feature_usage) + 1e-8)

        # Entropy (higher = more diverse)
        entropy = -jnp.sum(feature_usage * jnp.log(feature_usage + 1e-8))

        # Return negative (we minimize loss, so minimize -entropy = maximize entropy)
        return -weight * entropy


class FactorizedInteractionParams(NamedTuple):
    """Parameters for factorized interaction model.

    More efficient version using tensor decomposition.
    """

    U: Array  # (num_features, rank) - feature embeddings
    V: Array  # (num_features, rank) - interaction embeddings
    threshold: Array  # scalar


class FactorizedInteractionSplit:
    """Efficient factorized interaction split.

    Uses low-rank factorization to represent all pairwise interactions:
        interaction(i,j) = <U[i], V[j]> + <U[j], V[i]>

    This is O(d * rank) instead of O(d²) for explicit pairwise.

    The split score is:
        score(x) = Σᵢⱼ interaction(i,j) * xᵢ * xⱼ
                 = x.T @ (U @ V.T + V @ U.T) @ x

    Which can be computed as:
        score(x) = ||U.T @ x||² + ||V.T @ x||² - ||diag||
                 = (U.T @ x) · (V.T @ x) * 2

    Very efficient: O(d * rank) per sample.
    """

    def __init__(
        self,
        rank: int = 8,
        include_linear: bool = True,
    ) -> None:
        """Initialize factorized interaction split.

        Args:
            rank: Rank of factorization (higher = more expressive).
            include_linear: Whether to include linear (non-interaction) term.
        """
        self.rank = rank
        self.include_linear = include_linear

    def init_params(
        self,
        key: Array,
        num_features: int,
        init_scale: float = 0.1,
    ) -> FactorizedInteractionParams:
        """Initialize factorization matrices."""
        keys = jax.random.split(key, 3)

        # Initialize with small values
        U = jax.random.normal(keys[0], (num_features, self.rank)) * init_scale
        V = jax.random.normal(keys[1], (num_features, self.rank)) * init_scale
        threshold = jnp.zeros(())

        return FactorizedInteractionParams(U=U, V=V, threshold=threshold)

    def compute_score(
        self,
        params: FactorizedInteractionParams,
        x: Array,
    ) -> Array:
        """Compute split score with factorized interactions.

        Efficiently computes: Σᵢⱼ <Uᵢ,Vⱼ> xᵢxⱼ

        Args:
            params: Split parameters.
            x: Input, shape (..., num_features).

        Returns:
            Split scores, shape (...).
        """
        # Project features: (batch, rank)
        u_proj = x @ params.U  # (batch, rank)
        v_proj = x @ params.V  # (batch, rank)

        # Interaction score: sum of element-wise products
        # This computes Σᵢⱼ <Uᵢ,Vⱼ> xᵢxⱼ efficiently
        interaction_score = jnp.sum(u_proj * v_proj, axis=-1)

        return interaction_score - params.threshold

    def get_interaction_matrix(
        self,
        params: FactorizedInteractionParams,
    ) -> Array:
        """Reconstruct full interaction matrix for interpretability.

        Returns:
            Interaction weights, shape (num_features, num_features).
        """
        return params.U @ params.V.T

    def get_top_interactions(
        self,
        params: FactorizedInteractionParams,
        k: int = 10,
    ) -> list[tuple[int, int, float]]:
        """Get top-k strongest interactions.

        Returns:
            List of (feature_i, feature_j, strength) tuples.
        """
        interaction_matrix = self.get_interaction_matrix(params)
        num_features = interaction_matrix.shape[0]

        # Get upper triangle (avoid duplicates)
        triu_idx = jnp.triu_indices(num_features, k=1)
        values = interaction_matrix[triu_idx]

        # Get top-k by absolute value
        abs_values = jnp.abs(values)
        k = min(k, len(values))
        _, top_idx = jax.lax.top_k(abs_values, k)

        # Extract (i, j, strength) tuples
        interactions = []
        for idx in top_idx:
            i = int(triu_idx[0][idx])
            j = int(triu_idx[1][idx])
            strength = float(values[idx])
            interactions.append((i, j, strength))

        return interactions

    def l2_regularization(
        self,
        params: FactorizedInteractionParams,
        weight: float = 0.01,
    ) -> Array:
        """L2 regularization on factorization matrices."""
        return weight * (jnp.sum(params.U**2) + jnp.sum(params.V**2))

