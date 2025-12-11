# jaxboost

Differentiable gradient boosting in JAX.

⚠️ **This is a research/learning project.** The main purpose is to explore novel ideas in gradient boosting using JAX's differentiable programming capabilities. Not intended to replace production libraries like XGBoost/LightGBM. Issues and ideas welcome!

## Features

### Core
- **Soft Differentiable Trees**: Sigmoid routing enables end-to-end gradient descent training
- **Oblivious Tree Structure**: Same split at each depth level, efficient for GPU
- **GPU Acceleration**: Fully vectorized JAX operations
- **End-to-End Training**: Train via optax optimizers, not greedy tree building

### Split Functions
| Split Type | Description | Use Case |
|------------|-------------|----------|
| `HyperplaneSplit` | Linear combinations of features | Default, captures interactions |
| `AxisAlignedSplit` | Single feature splits | Interpretable, like traditional trees |
| `SparseHyperplaneSplit` | L1-regularized hyperplanes | Feature selection |
| `TopKHyperplaneSplit` | Hard top-k feature selection | Strict sparsity |
| `AttentionSplit` | Input-dependent feature weighting | Complex interactions (large data) |
| `InteractionDiscoverySplit` | Automatic pairwise interactions | Interaction detection |

### Tree Structures
| Structure | Description | Key Feature |
|-----------|-------------|-------------|
| `ObliviousTree` | Constant leaf values | Standard, fast |
| `LinearLeafTree` | Linear models in leaves | **Extrapolation beyond training range** |

### Advanced Modules
| Module | Description |
|--------|-------------|
| `ODEBoosting` | Continuous-time boosting via Neural ODEs |
| `IBTree` | Information Bottleneck regularization for principled complexity control |

## Installation

```bash
pip install jaxboost
```

Or from source:

```bash
git clone https://github.com/jxu/jaxboost.git
cd jaxboost
pip install -e .
```

## Quick Start

### High-Level API

```python
from jaxboost import GBMTrainer, TrainerConfig

# Regression
trainer = GBMTrainer(task="regression")
model = trainer.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification
trainer = GBMTrainer(task="classification")
model = trainer.fit(X_train, y_train)
probabilities = model.predict(X_test)
classes = model.predict_class(X_test)
```

### Configuration

```python
config = TrainerConfig(
    n_trees=20,          # Number of trees
    depth=4,             # Tree depth
    learning_rate=0.01,  # Optimizer learning rate
    epochs=500,          # Training epochs
    patience=50,         # Early stopping patience
    verbose=True,
)
trainer = GBMTrainer(task="regression", config=config)
```

### Low-Level API

```python
import jax
from jaxboost import (
    ObliviousTree, 
    HyperplaneSplit, 
    soft_routing,
    boosting_aggregate,
)

# Initialize components
key = jax.random.PRNGKey(0)
tree = ObliviousTree()
split_fn = HyperplaneSplit()
routing_fn = lambda s: soft_routing(s, temperature=1.0)

# Initialize and use tree
params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)
predictions = tree.forward(params, X, split_fn, routing_fn)
```

### Linear Leaf Trees (Extrapolation)

```python
from jaxboost.structures import LinearLeafTree

# Trees that can extrapolate beyond training data
tree = LinearLeafTree(l2_leaf_reg=0.01)
params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)

# Works for x outside training range!
predictions = tree.forward(params, x_extrapolate, split_fn, routing_fn)
```

### Information Bottleneck Trees

```python
from jaxboost.ib import IBTree

# Principled regularization via Information Bottleneck
tree = IBTree(depth=4, beta=0.1)  # beta controls complexity
params = tree.init_params(key, num_features=10)

# Prediction with uncertainty
mean, variance = tree.predict_with_uncertainty(params, X)
```

### ODE Boosting

```python
from jaxboost.aggregation import ODEBoosting

# Continuous-time boosting (requires: pip install diffrax)
ode_boost = ODEBoosting(depth=4, t_span=(0.0, 1.0))
params = ode_boost.init_params(key, num_features=10)
predictions = ode_boost.forward(params, X)
```

## Examples

```bash
# Quick start
python examples/quickstart.py

# Extrapolation demo
python examples/linear_leaf_extrapolation.py

# Benchmark different split functions
python examples/benchmark_splits.py

# Differentiable tree demo
python examples/differentiable_tree_demo.py
```

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- optax >= 0.1.7
- diffrax (optional, for ODE boosting)

## Project Structure

```
src/jaxboost/
├── training/       # High-level GBMTrainer API
├── structures/     # Tree structures (ObliviousTree, LinearLeafTree)
├── splits/         # Split functions (Hyperplane, Sparse, Attention, etc.)
├── routing/        # Soft routing functions
├── aggregation/    # Boosting aggregation, ODE boosting
├── losses/         # Loss functions
├── ib/             # Information Bottleneck trees
└── core/           # Protocols and base types
```

## License

MIT
