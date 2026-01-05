<p align="center">
  <img src="assets/logo.png" alt="jaxboost" width="300">
</p>

**JAX autodiff for XGBoost/LightGBM objectives**

Write a loss function, get gradients and Hessians automatically. No manual derivation needed.

## Features

- **Automatic Gradients** - JAX computes gradients for any loss function
- **Automatic Hessians** - Second derivatives computed automatically
- **Built-in Objectives** - Focal loss, Huber, quantile, survival, and more
- **Works Everywhere** - XGBoost, LightGBM, CatBoost compatible

## Installation

```bash
pip install jaxboost
```

## Quick Example

```python
import xgboost as xgb
from jaxboost import auto_objective, focal_loss

# Use built-in objectives
model = xgb.train(params, dtrain, obj=focal_loss.xgb_objective)

# Or create your own
@auto_objective
def my_loss(y_pred, y_true):
    return (y_pred - y_true) ** 2

model = xgb.train(params, dtrain, obj=my_loss.xgb_objective)
```

## Why jaxboost?

| Traditional Approach | jaxboost |
|---------------------|----------|
| Derive gradients by hand | Write loss, get gradients free |
| Derive Hessians by hand | Write loss, get Hessians free |
| Error-prone math | JAX autodiff is correct by construction |
| One loss = hours of work | One loss = 5 lines of code |

## Next Steps

- [Quick Start Guide](getting-started/quickstart.md) - Get started with jaxboost
- [API Reference](api/index.md) - Detailed API documentation
- [Research Notes](research.md) - Archived research work on differentiable trees
