<p align="center">
  <img src="assets/logo.png" alt="jaxboost" width="300">
</p>

[![Tests](https://github.com/jxucoder/jaxboost/actions/workflows/tests.yml/badge.svg)](https://github.com/jxucoder/jaxboost/actions/workflows/tests.yml)
[![Lint](https://github.com/jxucoder/jaxboost/actions/workflows/lint.yml/badge.svg)](https://github.com/jxucoder/jaxboost/actions/workflows/lint.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**JAX autodiff for XGBoost/LightGBM objectives.**

Write a loss function, get gradients and Hessians automatically. No manual derivation needed.

Works with **XGBoost** and **LightGBM**.

## Install

```bash
pip install jaxboost
```

## Quick Start

### XGBoost

```python
import xgboost as xgb
import jax.numpy as jnp
from jaxboost import auto_objective, focal_loss, huber, quantile

# Prepare your data
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"max_depth": 4, "eta": 0.1}

# Built-in objectives - just use them
model = xgb.train(params, dtrain, num_boost_round=100, obj=focal_loss.xgb_objective)
model = xgb.train(params, dtrain, num_boost_round=100, obj=huber.xgb_objective)
model = xgb.train(params, dtrain, num_boost_round=100, obj=quantile(0.9).xgb_objective)

# Custom objective - write the loss, autodiff handles the rest
@auto_objective
def asymmetric_mse(y_pred, y_true, alpha=0.7):
    error = y_true - y_pred
    return jnp.where(error > 0, alpha * error**2, (1 - alpha) * error**2)

model = xgb.train(params, dtrain, num_boost_round=100, obj=asymmetric_mse.xgb_objective)
```

### LightGBM

```python
import lightgbm as lgb
from jaxboost import huber

train_data = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 4, "learning_rate": 0.1}

model = lgb.train(params, train_data, num_boost_round=100, fobj=huber.lgb_objective)
```


## Available Objectives

### Regression
| Objective | Description |
|-----------|-------------|
| `mse` | Mean squared error |
| `huber` | Huber loss (robust to outliers) |
| `pseudo_huber` | Smooth approximation of Huber loss |
| `log_cosh` | Log-cosh loss |
| `mae_smooth` | Smooth approximation of MAE |
| `quantile(q)` | Quantile regression |
| `asymmetric(alpha)` | Asymmetric squared error |
| `tweedie(p)` | Tweedie deviance |

### Binary Classification
| Objective | Description |
|-----------|-------------|
| `focal_loss` | Focal loss for imbalanced data |
| `binary_crossentropy` | Standard log loss |
| `weighted_binary_crossentropy` | Weighted binary cross-entropy |
| `hinge_loss` | SVM-style hinge loss |

### Multi-class Classification
| Objective | Description |
|-----------|-------------|
| `softmax_cross_entropy` | Standard multi-class |
| `focal_multiclass` | Focal loss for multi-class |
| `label_smoothing(eps)` | Label smoothing regularization |
| `class_balanced` | Class-balanced loss |

### Survival Analysis
| Objective | Description |
|-----------|-------------|
| `aft` | Accelerated failure time (log-normal) |
| `weibull_aft` | Weibull AFT model |

### Multi-task Learning
| Objective | Description |
|-----------|-------------|
| `multi_task_regression` | Multiple regression targets |
| `multi_task_classification` | Multiple classification targets |
| `multi_task_huber` | Multi-task Huber loss |
| `multi_task_quantile` | Multi-task quantile loss |
| `MaskedMultiTaskObjective` | Handle missing labels |

### Uncertainty Estimation
| Objective | Description |
|-----------|-------------|
| `gaussian_nll` | Predict mean + variance |
| `laplace_nll` | Predict median + scale |

## Custom Objectives

The `@auto_objective` decorator turns any loss function into an XGBoost/LightGBM objective:

```python
import xgboost as xgb
import lightgbm as lgb
import jax.numpy as jnp
from jaxboost import auto_objective

@auto_objective
def my_custom_loss(y_pred, y_true, **kwargs):
    # Write your loss here - JAX computes grad/hess automatically
    return (y_pred - y_true) ** 2

# Use with XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"max_depth": 4, "eta": 0.1}
model = xgb.train(params, dtrain, num_boost_round=100, obj=my_custom_loss.xgb_objective)

# Use with LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
params = {"max_depth": 4, "learning_rate": 0.1}
model = lgb.train(params, train_data, num_boost_round=100, fobj=my_custom_loss.lgb_objective)

# Pass parameters
model = xgb.train(
    params, dtrain, num_boost_round=100,
    obj=my_custom_loss.get_xgb_objective(alpha=0.5)
)
```

## Multi-class Example

```python
import xgboost as xgb
import jax
import jax.numpy as jnp
from jaxboost import multiclass_objective

@multiclass_objective(num_classes=3)
def custom_multiclass(logits, label):
    # logits: (num_classes,), label: scalar
    probs = jax.nn.softmax(logits)
    return -jnp.log(probs[label] + 1e-7)

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(
    {"num_class": 3, "max_depth": 4, "eta": 0.1},
    dtrain,
    num_boost_round=100,
    obj=custom_multiclass.xgb_objective
)
```

## Why jaxboost?

| Traditional Approach | jaxboost |
|---------------------|----------|
| Derive gradients by hand | Write loss, get gradients free |
| Derive Hessians by hand | Write loss, get Hessians free |
| Error-prone math | JAX autodiff is correct by construction |
| One loss = hours of work | One loss = 5 lines of code |

## Requirements

- Python >= 3.10
- JAX >= 0.4.20

## Documentation

Full documentation available at: https://jxucoder.github.io/jaxboost/

## License

Apache 2.0
