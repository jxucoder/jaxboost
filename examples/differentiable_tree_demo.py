"""
Differentiable Tree Demo
========================

This demo shows how soft oblivious trees are fully differentiable:
1. Soft feature selection (softmax over features)
2. Soft routing (sigmoid for left/right decisions)
3. Gradients flow through the entire tree
4. End-to-end training with gradient descent
"""

import jax
import jax.numpy as jnp
import optax

from jaxboost import (
    AxisAlignedSplit,
    AxisAlignedSplitParams,
    ObliviousTree,
    ObliviousTreeParams,
    mse_loss,
    soft_routing,
)


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")


def demo_soft_feature_selection():
    """Demo 1: Soft feature selection via softmax."""
    print_header("Demo 1: Soft Feature Selection")

    # Create split parameters
    key = jax.random.PRNGKey(0)
    split_fn = AxisAlignedSplit()
    params = split_fn.init_params(key, num_features=4)

    print("Feature logits (learnable parameters):")
    print(f"  {params.feature_logits}")

    # Softmax converts logits to probabilities
    feature_probs = jax.nn.softmax(params.feature_logits)
    print(f"\nFeature probabilities (after softmax):")
    print(f"  {feature_probs}")
    print(f"  Sum = {feature_probs.sum():.4f}")

    # Sample input
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    print(f"\nInput features: {x}")

    # Soft feature selection = weighted sum
    selected_value = jnp.sum(x * feature_probs)
    print(f"Selected value (weighted sum): {selected_value:.4f}")

    # Compare to hard selection
    hard_idx = jnp.argmax(params.feature_logits)
    print(f"\nIf hard selection (argmax): would pick feature {hard_idx}, value = {x[hard_idx]}")
    print("→ Soft selection is differentiable, hard selection is not!")


def demo_soft_routing():
    """Demo 2: Soft routing via sigmoid."""
    print_header("Demo 2: Soft Routing (Sigmoid)")

    # Different temperatures
    scores = jnp.linspace(-3, 3, 7)

    print("Split scores → Routing probabilities (P(go right))")
    print("-" * 50)

    for temp in [0.5, 1.0, 2.0, 5.0]:
        probs = soft_routing(scores, temperature=temp)
        print(f"\nTemperature = {temp}:")
        for s, p in zip(scores, probs):
            bar = "█" * int(p * 20)
            print(f"  score={s:+.1f} → P(right)={p:.3f} {bar}")

    print("\n→ Lower temperature = sharper (more like hard decision)")
    print("→ Higher temperature = softer (more uncertainty)")


def demo_gradient_flow():
    """Demo 3: Gradients flow through the tree."""
    print_header("Demo 3: Gradient Flow")

    key = jax.random.PRNGKey(42)
    split_fn = AxisAlignedSplit()
    tree = ObliviousTree()

    # Initialize a depth-2 tree (4 leaves)
    params = tree.init_params(key, depth=2, num_features=3, split_fn=split_fn)

    print("Tree structure: depth=2, features=3, leaves=4")
    print(f"Leaf values: {params.leaf_values}")

    # Single sample
    x = jnp.array([0.5, -0.5, 1.0])
    y_target = 1.0

    print(f"\nInput: {x}")
    print(f"Target: {y_target}")

    # Forward pass
    y_pred = tree.forward(params, x, split_fn, soft_routing)
    print(f"Prediction: {y_pred:.4f}")

    # Compute gradients
    def loss_fn(params):
        pred = tree.forward(params, x, split_fn, soft_routing)
        return (pred - y_target) ** 2

    loss, grads = jax.value_and_grad(loss_fn)(params)

    print(f"\nLoss: {loss:.4f}")
    print(f"\nGradients exist for all parameters:")
    print(f"  - Leaf values gradient: {grads.leaf_values}")

    for i, split_grad in enumerate(grads.split_params):
        print(f"  - Split {i} feature_logits gradient: {split_grad.feature_logits}")
        print(f"  - Split {i} threshold gradient: {split_grad.threshold:.4f}")


def demo_learning():
    """Demo 4: End-to-end learning."""
    print_header("Demo 4: End-to-End Learning")

    key = jax.random.PRNGKey(0)

    # Simple XOR-like problem
    X = jnp.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ], dtype=jnp.float32)

    # XOR: output 1 if signs differ
    y = jnp.array([0., 1., 1., 0.])

    print("XOR-like problem:")
    print("  x1  x2  →  y")
    for xi, yi in zip(X, y):
        print(f"  {xi[0]:+.0f}  {xi[1]:+.0f}  →  {yi:.0f}")

    # Initialize tree
    split_fn = AxisAlignedSplit()
    tree = ObliviousTree()
    params = tree.init_params(key, depth=2, num_features=2, split_fn=split_fn)

    # Optimizer
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params)

    def loss_fn(params):
        preds = tree.forward(params, X, split_fn, soft_routing)
        return mse_loss(preds, y)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("\nTraining...")
    print("-" * 40)

    for epoch in range(201):
        params, opt_state, loss = train_step(params, opt_state)
        if epoch % 50 == 0:
            preds = tree.forward(params, X, split_fn, soft_routing)
            print(f"Epoch {epoch:3d}: loss={loss:.4f}, preds={preds.round(2)}")

    print("\nFinal predictions vs targets:")
    final_preds = tree.forward(params, X, split_fn, soft_routing)
    for xi, yi, pi in zip(X, y, final_preds):
        status = "ok" if abs(yi - pi) < 0.3 else "miss"
        print(f"  {xi} → pred={pi:.2f}, target={yi:.0f} {status}")

    # Show learned feature weights
    print("\nLearned feature importance at each split:")
    for i, sp in enumerate(params.split_params):
        probs = jax.nn.softmax(sp.feature_logits)
        print(f"  Split {i}: feature 0 = {probs[0]:.2%}, feature 1 = {probs[1]:.2%}")


def demo_jit_compatible():
    """Demo 5: JIT compilation."""
    print_header("Demo 5: JIT Compilation")

    key = jax.random.PRNGKey(0)
    split_fn = AxisAlignedSplit()
    tree = ObliviousTree()

    params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)
    X = jax.random.normal(key, (1000, 10))

    # JIT compile the forward pass
    @jax.jit
    def predict(params, X):
        return tree.forward(params, X, split_fn, soft_routing)

    # Warm up
    _ = predict(params, X)

    # Time it
    import time
    start = time.time()
    for _ in range(100):
        _ = predict(params, X).block_until_ready()
    elapsed = time.time() - start

    print(f"JIT-compiled forward pass:")
    print(f"  - 1000 samples × 10 features × depth-4 tree")
    print(f"  - 100 iterations: {elapsed*1000:.2f} ms")
    print(f"  - Per iteration: {elapsed*10:.3f} ms")
    print(f"\nFully compatible with jax.jit.")


def main():
    print("\n" + "=" * 60)
    print(" JAXBOOST: Differentiable Tree Demo")
    print("=" * 60)

    demo_soft_feature_selection()
    demo_soft_routing()
    demo_gradient_flow()
    demo_learning()
    demo_jit_compatible()

    print_header("Summary")
    print("""
Key takeaways:

1. SOFT FEATURE SELECTION
   - Instead of picking one feature (hard), we use softmax to
     get a weighted combination of all features
   - This is differentiable!

2. SOFT ROUTING
   - Instead of if-else (hard), we use sigmoid to get
     probability of going left/right
   - This is differentiable!

3. GRADIENT FLOW
   - Gradients flow through the entire tree structure
   - We can optimize: leaf values, thresholds, feature selection

4. END-TO-END LEARNING
   - Train with standard optimizers (Adam, SGD)
   - No greedy split finding needed
   - Global optimization of tree parameters

5. JAX BENEFITS
   - JIT compilation for speed
   - vmap for batching
   - Automatic differentiation
""")


if __name__ == "__main__":
    main()

