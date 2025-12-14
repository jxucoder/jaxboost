# jaxboost

Differentiable gradient boosting in JAX.

Two things:

1. **Soft trees** - End-to-end differentiable tree ensembles using sigmoid routing, trainable via gradient descent (unlike greedy XGBoost/LightGBM)

2. **XGBoost/LightGBM objectives** - Define custom losses with `@auto_objective` decorator, get gradients/Hessians via JAX autodiff.

## Install

```bash
pip install jaxboost
```

## XGBoost/LightGBM Objectives

Use built-in objectives or define your own - gradients and Hessians computed automatically:

```python
import xgboost as xgb
from jaxboost.objective import focal_loss, huber, quantile, auto_objective

# Built-in objectives
model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)
model = xgb.train(params, dtrain, obj=huber.xgb_objective)
model = xgb.train(params, dtrain, obj=quantile(0.9).xgb_objective)

# Custom objective - just write the loss, autodiff handles the rest
@auto_objective
def my_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
```

Available: `focal_loss`, `huber`, `quantile`, `tweedie`, `softmax_cross_entropy`, `cox_partial_likelihood`, `multi_task_regression`, and more. See `jaxboost.objective`.

## Soft Trees

```python
from jaxboost import GBMTrainer

trainer = GBMTrainer(task="regression")  # or "classification"
model = trainer.fit(X_train, y_train)
predictions = model.predict(X_test)
```

With config:

```python
from jaxboost import GBMTrainer, TrainerConfig

config = TrainerConfig(
    n_trees=20,
    depth=4,
    learning_rate=0.01,
    epochs=500,
    patience=50,
)
trainer = GBMTrainer(task="regression", config=config)
```

## Split Functions

Control how each tree node splits the data:

| Split | What it does |
|-------|--------------|
| `HyperplaneSplit` | Linear combination of features (default) |
| `AxisAlignedSplit` | Single feature threshold, like traditional trees |
| `SparseHyperplaneSplit` | Learned feature selection via soft gates |
| `TopKHyperplaneSplit` | Hard top-k feature selection |
| `AttentionSplit` | Input-dependent feature weighting |

## Tree Structures

| Structure | What it does |
|-----------|--------------|
| `ObliviousTree` | Same split at each depth (like CatBoost), constant leaf values |
| `LinearLeafTree` | Linear models at leaves, can extrapolate beyond training range |

## Mixture of Experts

Differentiable MOE with soft tree experts:

```python
from jaxboost.ensemble import MOEEnsemble

moe = MOEEnsemble(num_experts=4, trees_per_expert=10, gating="tree")
params = moe.fit(X_train, y_train)
predictions = moe.predict(params, X_test)
```

EM-MOE with XGBoost/LightGBM/CatBoost experts:

```python
from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert

experts = [create_xgboost_expert(n_estimators=100) for _ in range(4)]
config = EMConfig(num_experts=4, em_iterations=10, expert_init_strategy="cluster")

moe = EMMOE(experts, config=config)
moe.fit(X_train, y_train)
mean, std = moe.predict_with_uncertainty(X_test)
```

## Low-Level API

```python
import jax
from jaxboost import ObliviousTree, HyperplaneSplit, soft_routing

key = jax.random.PRNGKey(0)
tree = ObliviousTree()
split_fn = HyperplaneSplit()

params = tree.init_params(key, depth=4, num_features=10, split_fn=split_fn)
predictions = tree.forward(params, X, split_fn, lambda s: soft_routing(s, temperature=1.0))
```

## Examples

```bash
python examples/quickstart.py
python examples/linear_leaf_extrapolation.py
python examples/benchmark_splits.py
python examples/moe_demo.py
python examples/hybrid_moe_demo.py
```

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- optax >= 0.1.7

## License

MIT
