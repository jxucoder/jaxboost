# Quick Start

This guide will help you get started with jaxboost.

## Installation

```bash
pip install jaxboost
```

## Basic Usage

jaxboost lets you write custom loss functions and automatically generates the gradients and Hessians needed by XGBoost, LightGBM, or CatBoost.

### Using Built-in Objectives

```python
import xgboost as xgb
from jaxboost import focal_loss, huber, quantile

# Load your data
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Use focal loss for imbalanced classification
params = {"max_depth": 4, "eta": 0.1}
model = xgb.train(params, dtrain, num_boost_round=100, obj=focal_loss.xgb_objective)

# Use Huber loss for robust regression
model = xgb.train(params, dtrain, num_boost_round=100, obj=huber.xgb_objective)

# Use quantile loss for median regression
model = xgb.train(params, dtrain, num_boost_round=100, obj=quantile(0.5).xgb_objective)
```

### Custom Objectives

Create your own objective function with the `@auto_objective` decorator:

```python
import jax.numpy as jnp
from jaxboost import auto_objective

@auto_objective
def asymmetric_mse(y_pred, y_true, alpha=0.7):
    """Penalize under-predictions more than over-predictions."""
    error = y_true - y_pred
    return jnp.where(error > 0, alpha * error**2, (1 - alpha) * error**2)

# Use with XGBoost
model = xgb.train(params, dtrain, num_boost_round=100, obj=asymmetric_mse.xgb_objective)

# Pass custom parameters
model = xgb.train(
    params, dtrain, num_boost_round=100,
    obj=asymmetric_mse.get_xgb_objective(alpha=0.9)
)
```

### LightGBM

```python
import lightgbm as lgb
from jaxboost import huber

train_data = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 4, "learning_rate": 0.1}

model = lgb.train(params, train_data, num_boost_round=100, fobj=huber.lgb_objective)
```

## Multi-class Classification

For multi-class problems, use `@multiclass_objective`:

```python
import jax
import jax.numpy as jnp
from jaxboost import multiclass_objective

@multiclass_objective(num_classes=3)
def custom_multiclass(logits, label):
    probs = jax.nn.softmax(logits)
    return -jnp.log(probs[label] + 1e-7)

params = {"num_class": 3, "objective": "multi:softmax", "max_depth": 4}
model = xgb.train(params, dtrain, num_boost_round=100, obj=custom_multiclass.xgb_objective)
```

## Multi-task Learning

Handle multiple targets with optional missing labels:

```python
from jaxboost import MaskedMultiTaskObjective

# 3 regression tasks
objective = MaskedMultiTaskObjective(n_tasks=3)

# Create mask for missing labels (1 = valid, 0 = missing)
mask = np.ones_like(y_train)
mask[some_indices] = 0  # Mark missing labels

model = xgb.train(
    params, dtrain, num_boost_round=100,
    obj=objective.get_xgb_objective(mask=mask)
)
```

## Survival Analysis

Built-in objectives for survival models:

```python
from jaxboost import cox_partial_likelihood, aft

# Cox proportional hazards
model = xgb.train(params, dtrain, obj=cox_partial_likelihood.xgb_objective)

# Accelerated failure time
model = xgb.train(params, dtrain, obj=aft.xgb_objective)
```

## Available Objectives

See the [API Reference](../api/losses.md) for a complete list of built-in objectives:

- **Regression**: `mse`, `huber`, `quantile`, `tweedie`, `asymmetric`, `log_cosh`
- **Binary Classification**: `focal_loss`, `binary_crossentropy`, `hinge_loss`
- **Multi-class**: `softmax_cross_entropy`, `focal_multiclass`, `label_smoothing`
- **Survival**: `cox_partial_likelihood`, `aft`, `weibull_aft`
- **Multi-task**: `multi_task_regression`, `multi_task_classification`
- **Uncertainty**: `gaussian_nll`, `laplace_nll`
