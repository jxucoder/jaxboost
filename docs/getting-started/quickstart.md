# Quick Start

This guide will help you get started with JAXBoost in minutes.

---

## Installation

```bash
pip install jaxboost
```

!!! tip "Using uv?"
    ```bash
    uv add jaxboost
    ```

---

## Basic Usage

JAXBoost lets you write custom loss functions and automatically generates the gradients and Hessians needed by XGBoost and LightGBM.

### Using Built-in Objectives

=== "XGBoost"

    ```python
    import xgboost as xgb
    from jaxboost import focal_loss, huber, quantile

    # Load your data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {"max_depth": 4, "eta": 0.1}

    # Focal loss for imbalanced classification
    model = xgb.train(params, dtrain, num_boost_round=100, obj=focal_loss.xgb_objective)

    # Huber loss for robust regression
    model = xgb.train(params, dtrain, num_boost_round=100, obj=huber.xgb_objective)

    # Quantile loss for median regression
    model = xgb.train(params, dtrain, num_boost_round=100, obj=quantile(0.5).xgb_objective)
    ```

=== "LightGBM"

    ```python
    import lightgbm as lgb
    from jaxboost import huber

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {"max_depth": 4, "learning_rate": 0.1}

    model = lgb.train(params, train_data, num_boost_round=100, fobj=huber.lgb_objective)
    ```


---

## Custom Objectives

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
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"max_depth": 4, "eta": 0.1}
model = xgb.train(params, dtrain, num_boost_round=100, obj=asymmetric_mse.xgb_objective)

# Use with LightGBM
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 4, "learning_rate": 0.1}
model = lgb.train(params, train_data, num_boost_round=100, fobj=asymmetric_mse.lgb_objective)
```

!!! info "How it works"
    The `@auto_objective` decorator wraps your loss function and uses JAX's `grad` and `hessian` to automatically compute derivatives. Your function should:
    
    1. Take `y_pred` and `y_true` as the first two arguments
    2. Return a scalar loss value per sample
    3. Use JAX-compatible operations (e.g., `jax.numpy` instead of `numpy`)

### Passing Custom Parameters

```python
# Default parameters
model = xgb.train(params, dtrain, obj=asymmetric_mse.xgb_objective)

# Custom parameters
model = xgb.train(
    params, dtrain, num_boost_round=100,
    obj=asymmetric_mse.get_xgb_objective(alpha=0.9)
)
```

---

## Multi-class Classification

For multi-class problems, use `@multiclass_objective`:

```python
import jax
import jax.numpy as jnp
from jaxboost import multiclass_objective

@multiclass_objective(num_classes=3)
def custom_multiclass(logits, label):
    """Custom multi-class loss."""
    probs = jax.nn.softmax(logits)
    return -jnp.log(probs[label] + 1e-7)

params = {"num_class": 3, "max_depth": 4}
model = xgb.train(params, dtrain, num_boost_round=100, obj=custom_multiclass.xgb_objective)
```

!!! warning "XGBoost Configuration"
    For multi-class objectives, make sure to set `num_class` in params to match your decorator.

---

## Multi-task Learning

Handle multiple targets with optional missing labels:

```python
import numpy as np
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

---

## Survival Analysis

Built-in objectives for time-to-event modeling:

```python
from jaxboost import aft, weibull_aft

# Accelerated failure time model
model = xgb.train(params, dtrain, obj=aft.xgb_objective)

# Weibull AFT model
model = xgb.train(params, dtrain, obj=weibull_aft.xgb_objective)
```

!!! tip "Data Format"
    For survival analysis, ensure your labels encode both the event time and censoring indicator according to the expected format.

---

## Uncertainty Estimation

Predict both values and uncertainty using multi-output objectives:

```python
from jaxboost import gaussian_nll

# Predicts mean and log-variance
model = xgb.train(params, dtrain, obj=gaussian_nll.xgb_objective)

# Get predictions (shape: [n_samples, 2])
preds = model.predict(dtest)
mean = preds[:, 0]
variance = np.exp(preds[:, 1])  # log-variance → variance
```

---

## Available Objectives

| Category | Objectives |
|----------|------------|
| **Regression** | `mse`, `huber`, `quantile`, `tweedie`, `asymmetric`, `log_cosh`, `pseudo_huber`, `mae_smooth`, `poisson`, `gamma` |
| **Binary Classification** | `focal_loss`, `binary_crossentropy`, `weighted_binary_crossentropy`, `hinge_loss` |
| **Multi-class** | `softmax_cross_entropy`, `focal_multiclass`, `label_smoothing`, `class_balanced` |
| **Survival** | `aft`, `weibull_aft` |
| **Multi-task** | `multi_task_regression`, `multi_task_classification`, `multi_task_huber`, `multi_task_quantile` |
| **Uncertainty** | `gaussian_nll`, `laplace_nll` |

See the [API Reference](../api/losses.md) for detailed documentation of each objective.

---

## Next Steps

- **[API Reference](../api/index.md)** — Full documentation
- **[Research Notes](../research.md)** — Archived research on differentiable trees
