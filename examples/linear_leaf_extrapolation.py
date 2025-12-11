"""Demonstration: Linear Leaf Trees can extrapolate beyond training data.

This example shows the key difference between:
- ObliviousTree: Constant leaves → cannot extrapolate
- LinearLeafTree: Linear leaves → can extrapolate

Run: python examples/linear_leaf_extrapolation.py
"""

import jax
import jax.numpy as jnp
import optax

from jaxboost.structures import ObliviousTree, LinearLeafTree
from jaxboost.splits import HyperplaneSplit
from jaxboost.routing import soft_routing


def generate_data(key, n_samples=200, noise_std=0.1):
    """Generate simple linear data: y = 2*x + 1 + noise"""
    x = jax.random.uniform(key, (n_samples, 1), minval=0, maxval=5)
    y = 2 * x[:, 0] + 1 + jax.random.normal(key, (n_samples,)) * noise_std
    return x, y


def train_tree(tree, params, x, y, split_fn, routing_fn, n_steps=500, lr=0.1):
    """Train tree with gradient descent."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params):
        pred = tree.forward(params, x, split_fn, routing_fn)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}: loss = {loss:.4f}")

    return params


def main():
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)

    # Generate training data (x in [0, 5])
    x_train, y_train = generate_data(keys[0])
    print(f"Training data: x in [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"True function: y = 2*x + 1\n")

    # Test points for extrapolation (x in [5, 10])
    x_test = jnp.linspace(5, 10, 50)[:, None]
    y_test_true = 2 * x_test[:, 0] + 1

    # Setup
    split_fn = HyperplaneSplit()
    routing_fn = lambda s: soft_routing(s, temperature=1.0)
    depth = 4
    num_features = 1

    # ============================================
    # Train ObliviousTree (constant leaves)
    # ============================================
    print("=" * 50)
    print("Training ObliviousTree (constant leaves)...")
    print("=" * 50)

    oblivious_tree = ObliviousTree()
    oblivious_params = oblivious_tree.init_params(
        keys[1], depth, num_features, split_fn
    )
    oblivious_params = train_tree(
        oblivious_tree, oblivious_params, x_train, y_train, split_fn, routing_fn
    )

    # ============================================
    # Train LinearLeafTree (linear leaves)
    # ============================================
    print("\n" + "=" * 50)
    print("Training LinearLeafTree (linear leaves)...")
    print("=" * 50)

    linear_tree = LinearLeafTree()
    linear_params = linear_tree.init_params(
        keys[2], depth, num_features, split_fn
    )
    linear_params = train_tree(
        linear_tree, linear_params, x_train, y_train, split_fn, routing_fn
    )

    # ============================================
    # Compare extrapolation
    # ============================================
    print("\n" + "=" * 50)
    print("Extrapolation Results (x in [5, 10], outside training range)")
    print("=" * 50)

    # Predictions on extrapolation region
    pred_oblivious = oblivious_tree.forward(
        oblivious_params, x_test, split_fn, routing_fn
    )
    pred_linear = linear_tree.forward(
        linear_params, x_test, split_fn, routing_fn
    )

    # Calculate errors
    mse_oblivious = jnp.mean((pred_oblivious - y_test_true) ** 2)
    mse_linear = jnp.mean((pred_linear - y_test_true) ** 2)

    print(f"\nTrue values at x=5:  y = {2*5+1:.1f}")
    print(f"True values at x=10: y = {2*10+1:.1f}")
    print()

    print("ObliviousTree predictions:")
    print(f"  x=5:  pred = {pred_oblivious[0]:.2f} (true: 11.0)")
    print(f"  x=10: pred = {pred_oblivious[-1]:.2f} (true: 21.0)")
    print(f"  MSE on extrapolation: {mse_oblivious:.4f}")
    print()

    print("LinearLeafTree predictions:")
    print(f"  x=5:  pred = {pred_linear[0]:.2f} (true: 11.0)")
    print(f"  x=10: pred = {pred_linear[-1]:.2f} (true: 21.0)")
    print(f"  MSE on extrapolation: {mse_linear:.4f}")
    print()

    improvement = (mse_oblivious - mse_linear) / mse_oblivious * 100
    print(f"LinearLeafTree improvement: {improvement:.1f}% lower MSE on extrapolation")

    # ============================================
    # Analyze leaf slopes (shows extrapolation direction)
    # ============================================
    print("\n" + "=" * 50)
    print("Leaf Analysis")
    print("=" * 50)

    slopes = linear_tree.get_effective_leaf_slopes(linear_params, feature_idx=0)
    print(f"\nLinearLeafTree leaf slopes for feature 0:")
    print(f"  Mean slope: {slopes.mean():.3f} (true slope: 2.0)")
    print(f"  Min slope:  {slopes.min():.3f}")
    print(f"  Max slope:  {slopes.max():.3f}")

    print("\n✓ LinearLeafTree can extrapolate because leaves have learned slopes!")
    print("✗ ObliviousTree cannot extrapolate - it only outputs constant values.\n")


if __name__ == "__main__":
    main()

