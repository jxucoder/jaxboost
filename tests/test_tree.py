"""Tests for oblivious tree and split functions."""

import jax
import jax.numpy as jnp
import pytest

from jaxboost import (
    AxisAlignedSplit,
    HyperplaneSplit,
    ObliviousTree,
    soft_routing,
)


class TestAxisAlignedSplit:
    def test_init_params(self):
        """Test axis-aligned split initialization."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        params = split_fn.init_params(key, num_features=5)

        assert params.feature_logits.shape == (5,)
        assert params.threshold.shape == ()

    def test_compute_score(self):
        """Test score computation."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        params = split_fn.init_params(key, num_features=3)

        x = jnp.array([1.0, 2.0, 3.0])
        score = split_fn.compute_score(params, x)

        assert score.shape == ()

    def test_compute_score_batch(self):
        """Test score computation with batch."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        params = split_fn.init_params(key, num_features=3)

        X = jnp.ones((10, 3))
        scores = split_fn.compute_score(params, X)

        assert scores.shape == (10,)


class TestHyperplaneSplit:
    def test_init_params(self):
        """Test hyperplane split initialization."""
        key = jax.random.PRNGKey(0)
        split_fn = HyperplaneSplit()
        params = split_fn.init_params(key, num_features=5)

        assert params.weights.shape == (5,)
        assert params.threshold.shape == ()
        # Weights should be normalized
        assert jnp.isclose(jnp.linalg.norm(params.weights), 1.0, atol=0.01)

    def test_compute_score(self):
        """Test hyperplane score computation."""
        key = jax.random.PRNGKey(0)
        split_fn = HyperplaneSplit()
        params = split_fn.init_params(key, num_features=3)

        x = jnp.array([1.0, 2.0, 3.0])
        score = split_fn.compute_score(params, x)

        assert score.shape == ()

    def test_compute_score_batch(self):
        """Test hyperplane score with batch."""
        key = jax.random.PRNGKey(0)
        split_fn = HyperplaneSplit()
        params = split_fn.init_params(key, num_features=3)

        X = jnp.ones((10, 3))
        scores = split_fn.compute_score(params, X)

        assert scores.shape == (10,)

    def test_with_tree(self):
        """Test hyperplane split works with oblivious tree."""
        key = jax.random.PRNGKey(0)
        split_fn = HyperplaneSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=3, num_features=5, split_fn=split_fn)
        X = jnp.ones((10, 5))

        output = tree.forward(params, X, split_fn, soft_routing)

        assert output.shape == (10,)

    def test_gradients_flow(self):
        """Test gradients flow through hyperplane tree."""
        key = jax.random.PRNGKey(0)
        split_fn = HyperplaneSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=2, num_features=4, split_fn=split_fn)
        X = jnp.ones((5, 4))
        y = jnp.zeros(5)

        def loss_fn(params):
            preds = tree.forward(params, X, split_fn, soft_routing)
            return jnp.mean((preds - y) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Check gradients exist for weights
        assert grads.split_params[0].weights is not None
        assert not jnp.allclose(grads.split_params[0].weights, 0)


class TestObliviousTree:
    def test_init_params(self):
        """Test tree parameter initialization."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=3, num_features=5, split_fn=split_fn)

        assert len(params.split_params) == 3
        assert params.leaf_values.shape == (8,)  # 2^3

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=2, num_features=4, split_fn=split_fn)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        output = tree.forward(params, x, split_fn, soft_routing)

        assert output.shape == ()  # scalar

    def test_forward_batch(self):
        """Test forward pass with batch."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=2, num_features=4, split_fn=split_fn)
        X = jnp.ones((10, 4))

        output = tree.forward(params, X, split_fn, soft_routing)

        assert output.shape == (10,)

    def test_gradients_flow(self):
        """Test that gradients flow through the tree."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=2, num_features=4, split_fn=split_fn)
        X = jnp.ones((5, 4))
        y = jnp.zeros(5)

        def loss_fn(params):
            preds = tree.forward(params, X, split_fn, soft_routing)
            return jnp.mean((preds - y) ** 2)

        grads = jax.grad(loss_fn)(params)

        # Check gradients exist and are not all zeros
        assert grads.leaf_values is not None
        assert not jnp.allclose(grads.leaf_values, 0)

    def test_leaf_probs_sum_to_one(self):
        """Test that leaf probabilities sum to 1."""
        key = jax.random.PRNGKey(0)
        split_fn = AxisAlignedSplit()
        tree = ObliviousTree()

        params = tree.init_params(key, depth=3, num_features=4, split_fn=split_fn)
        X = jax.random.normal(key, (100, 4))

        # Access internal method to check leaf probs
        p_rights = []
        for d in range(3):
            score = split_fn.compute_score(params.split_params[d], X)
            p_rights.append(soft_routing(score))
        p_rights = jnp.stack(p_rights, axis=0)

        leaf_probs = tree._compute_leaf_probs(p_rights, 3)

        # Each row should sum to 1
        sums = jnp.sum(leaf_probs, axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)
