# Ensemble & Mixture of Experts

## Differentiable MOE

End-to-end trainable MOE with soft decision tree experts.

::: jaxboost.ensemble.MOEEnsemble
    options:
      members:
        - __init__
        - fit
        - predict
        - get_expert_weights

::: jaxboost.ensemble.MOEParams

## Gating Networks

::: jaxboost.ensemble.LinearGating

::: jaxboost.ensemble.MLPGating

::: jaxboost.ensemble.TreeGating

## EM-MOE

External GBDT experts (XGBoost, LightGBM, CatBoost) trained via EM algorithm.

::: jaxboost.ensemble.hybrid_moe.EMMOE
    options:
      members:
        - __init__
        - fit
        - predict
        - predict_with_uncertainty
        - get_expert_weights
        - get_responsibilities

::: jaxboost.ensemble.hybrid_moe.EMConfig

### Variants

::: jaxboost.ensemble.hybrid_moe.HardEMMOE

::: jaxboost.ensemble.hybrid_moe.SparseEMMOE

### Expert Factories

::: jaxboost.ensemble.hybrid_moe.create_xgboost_expert

::: jaxboost.ensemble.hybrid_moe.create_lightgbm_expert

::: jaxboost.ensemble.hybrid_moe.create_catboost_expert

## Utilities

::: jaxboost.ensemble.sparse_top_k

::: jaxboost.ensemble.load_balance_loss

## Examples

```python
from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert

experts = [
    create_xgboost_expert(task="regression", n_estimators=100, max_depth=4)
    for _ in range(4)
]

config = EMConfig(
    num_experts=4,
    em_iterations=10,
    expert_init_strategy="cluster",
)

moe = EMMOE(experts, config=config)
moe.fit(X_train, y_train)

mean, std = moe.predict_with_uncertainty(X_test)
```

```python
from jaxboost.ensemble import MOEEnsemble

moe = MOEEnsemble(
    num_experts=4,
    trees_per_expert=10,
    tree_depth=4,
    gating="tree",
    task="regression",
)

params = moe.fit(X_train, y_train, epochs=300)
predictions = moe.predict(params, X_test)
```
