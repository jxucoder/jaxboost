# Research: Differentiable Gradient Boosting

!!! abstract "Archived Features"
    This document describes research work that was explored in JAXBoost but has been archived to focus the library on its core value proposition: **automatic objective functions for XGBoost/LightGBM**.
    
    The code for these features has been archived. If you're interested in any of these directions, please [open an issue](https://github.com/jxu/jaxboost/issues).

---

## Soft Decision Trees

End-to-end differentiable tree ensembles using sigmoid routing, trainable via gradient descent (unlike greedy XGBoost/LightGBM).

### ObliviousTree (CatBoost-style)

All nodes at the same depth share the same split, enabling efficient vectorized computation:

```python
from jaxboost import ObliviousTree, HyperplaneSplit, soft_routing

tree = ObliviousTree()
split_fn = HyperplaneSplit()
params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)
predictions = tree.forward(params, X, split_fn, lambda s: soft_routing(s, temperature=1.0))
```

### LinearLeafTree

Trees with linear models at leaves for extrapolation beyond training range:

```python
from jaxboost import LinearLeafTree

tree = LinearLeafTree(depth=4)
# Leaf predictions: w · x + b instead of constant values
```

### GBMTrainer (High-level API)

```python
from jaxboost import GBMTrainer, TrainerConfig

config = TrainerConfig(n_trees=20, depth=4, learning_rate=0.01, epochs=500)
trainer = GBMTrainer(task="regression", config=config)
model = trainer.fit(X_train, y_train)
predictions = model.predict(X_test)
```

---

## Split Functions

Various decision boundary shapes for tree nodes:

| Split | Description |
|-------|-------------|
| `HyperplaneSplit` | Linear combination of features: \( \mathbf{w} \cdot \mathbf{x} \leq t \) |
| `AxisAlignedSplit` | Single feature threshold (like traditional trees): \( x_j \leq t \) |
| `SparseHyperplaneSplit` | Learned feature selection via soft L0 gates |
| `TopKHyperplaneSplit` | Hard top-k feature selection |
| `AttentionSplit` | Input-dependent feature weighting via attention |
| `InteractionDiscoverySplit` | Automatic feature interaction discovery |

---

## Information Bottleneck Trees (IB-Trees)

Principled regularization for tree models using the Information Bottleneck framework:

$$
\mathcal{L} = -I(Y; Z) + \beta \cdot I(X; Z) \approx \text{Prediction Loss} + \beta \cdot \text{KL}[p(z|x) \| p(z)]
$$

!!! info "Key Insight"
    Trees are information channels. Each tree compresses input \( X \) into leaf assignment \( Z \), then predicts \( Y \) from \( Z \).

### Results on Small Datasets

| Dataset | n | XGBoost | IB-Tree | Improvement |
|---------|---|---------|---------|-------------|
| Diabetes | 442 | 3430 MSE | 2619 MSE | +23.6% |
| Breast Cancer | 569 | 95.6% | 99.1% | +3.7% |

IB-Trees showed **18% improvement** over XGBoost on small medical datasets through principled regularization.

---

## Mixture of Experts (MOE)

Differentiable MOE with soft tree experts:

```python
from jaxboost.ensemble import MOEEnsemble

moe = MOEEnsemble(num_experts=4, trees_per_expert=10, gating="tree")
params = moe.fit(X_train, y_train)
predictions = moe.predict(params, X_test)
```

### Gating Options

| Gating | Description |
|--------|-------------|
| `LinearGating` | Softmax over learned logits |
| `MLPGating` | Neural network gating |
| `TreeGating` | Tree-based expert routing |

### EM-MOE with Traditional GBDT Experts

```python
from jaxboost.ensemble import EMMOE, create_xgboost_expert

experts = [create_xgboost_expert(n_estimators=100) for _ in range(4)]
moe = EMMOE(experts, num_experts=4, em_iterations=10)
moe.fit(X_train, y_train)
mean, std = moe.predict_with_uncertainty(X_test)
```

---

## Neural ODE Boosting

Model boosting as solving an ODE:

$$
\frac{df(\mathbf{x}, t)}{dt} = \text{tree}(\mathbf{x}; \theta)
$$

Instead of discrete boosting rounds, solve an ODE where the tree output is the velocity field. Traditional boosting with learning rate \( \eta \) is equivalent to Euler discretization with step size \( \eta \).

!!! success "Benefits"
    - Implicit regularization through ODE dynamics
    - Adaptive step sizes
    - Controllable smoothness

---

## Prior-Fitted Networks (PFN)

Empirical Bayes approach to in-context learning:

1. Analyze observed data to discover structure
2. Learn a prior that matches the data distribution  
3. Train a transformer to do in-context learning on this prior

```python
from jaxboost.pfn import EmpiricalPFNTrainer

trainer = EmpiricalPFNTrainer()
pfn = trainer.fit(X_train, y_train)  # Learn prior from your data
y_pred = pfn.predict(X_context, y_context, X_test)  # Fast in-context prediction
```

---

## Archived Examples

The following example scripts demonstrated these features:

| Example | Description |
|---------|-------------|
| `quickstart.py` | Basic GBMTrainer usage |
| `differentiable_tree_demo.py` | Soft tree training |
| `linear_leaf_extrapolation.py` | LinearLeafTree for extrapolation |
| `benchmark_splits.py` | Comparing split functions |
| `moe_demo.py` | Basic MOE ensemble |
| `hybrid_moe_demo.py` | MOE with XGBoost experts |
| `benchmark_moe.py` | MOE benchmarks |
| `benchmark_em_moe.py` | EM-MOE benchmarks |
| `empirical_pfn_demo.py` | PFN demonstration |
| `empirical_prior_pfn.py` | Empirical prior learning |
| `benchmark_extended.py` | Soft trees vs XGBoost |

---

## References

??? note "Differentiable Trees"
    - NODE: Neural Oblivious Decision Ensembles (Popov et al., 2019)
    - Soft Decision Trees (Frosst & Hinton, 2017)
    - Deep Neural Decision Forests (Kontschieder et al., 2015)

??? note "Attention for Tabular"
    - TabNet (Arik & Pfister, 2019)
    - FT-Transformer (Gorishniy et al., 2021)

??? note "Information Bottleneck"
    - The Information Bottleneck Method (Tishby et al., 2000)
    - Deep Learning and the Information Bottleneck Principle (Tishby & Zaslavsky, 2015)

??? note "Neural ODEs"
    - Neural Ordinary Differential Equations (Chen et al., 2018)

---

## Why These Were Removed

!!! question "Why focus on `@auto_objective`?"
    The research features were removed to focus JAXBoost on its clearest value proposition:
    
    **`@auto_objective`**: Write a loss function, get XGBoost/LightGBM gradients and Hessians automatically via JAX autodiff.
    
    This is immediately useful, requires no migration, and works with production-ready GBDT libraries.

The research features, while interesting, had limited practical applicability:

- Soft trees are slower than XGBoost for large datasets
- Most users don't need differentiable trees
- Research-stage APIs create maintenance burden

If there's interest in reviving any of these features, please [open an issue on GitHub](https://github.com/jxu/jaxboost/issues).
