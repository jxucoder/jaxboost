# API Reference

Welcome to the JAXBoost API documentation.

!!! abstract "What is JAXBoost?"
    JAXBoost provides automatic objective functions for XGBoost and LightGBM using JAX automatic differentiation. Write a loss function, get gradients and Hessians automatically.

## Quick Example

=== "XGBoost"

    ```python
    import xgboost as xgb
    from jaxboost import auto_objective, focal_loss

    # Load your data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {"max_depth": 4, "eta": 0.1}

    # Use built-in objective
    model = xgb.train(params, dtrain, num_boost_round=100, obj=focal_loss.xgb_objective)

    # Create custom objective
    @auto_objective
    def my_loss(y_pred, y_true):
        return (y_pred - y_true) ** 2

    model = xgb.train(params, dtrain, num_boost_round=100, obj=my_loss.xgb_objective)
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

## Core API

| Class/Decorator | Description |
|-----------------|-------------|
| [`@auto_objective`](losses.md#jaxboost.objective.AutoObjective) | Decorator for scalar loss functions |
| [`AutoObjective`](losses.md#jaxboost.objective.AutoObjective) | Base class for custom objectives |
| [`MultiClassObjective`](losses.md#jaxboost.objective.MultiClassObjective) | Multi-class classification objectives |
| [`MultiOutputObjective`](losses.md#jaxboost.objective.MultiOutputObjective) | Multi-output objectives (uncertainty) |
| [`MaskedMultiTaskObjective`](losses.md#jaxboost.objective.MaskedMultiTaskObjective) | Multi-task with missing labels |

---

## Built-in Objectives

### Binary Classification

| Objective | Description |
|-----------|-------------|
| [`focal_loss`](losses.md#jaxboost.objective.focal_loss) | Focal loss for imbalanced data |
| [`binary_crossentropy`](losses.md#jaxboost.objective.binary_crossentropy) | Standard binary cross-entropy |
| [`weighted_binary_crossentropy`](losses.md#jaxboost.objective.weighted_binary_crossentropy) | Weighted binary cross-entropy |
| [`hinge_loss`](losses.md#jaxboost.objective.hinge_loss) | SVM-style hinge loss |

### Regression

| Objective | Description |
|-----------|-------------|
| [`mse`](losses.md#jaxboost.objective.mse) | Mean squared error |
| [`huber`](losses.md#jaxboost.objective.huber) | Huber loss (robust to outliers) |
| [`pseudo_huber`](losses.md#jaxboost.objective.pseudo_huber) | Smooth approximation of Huber |
| [`log_cosh`](losses.md#jaxboost.objective.log_cosh) | Log-cosh loss |
| [`mae_smooth`](losses.md#jaxboost.objective.mae_smooth) | Smooth approximation of MAE |
| [`quantile`](losses.md#jaxboost.objective.quantile) | Quantile regression |
| [`asymmetric`](losses.md#jaxboost.objective.asymmetric) | Asymmetric loss |
| [`tweedie`](losses.md#jaxboost.objective.tweedie) | Tweedie deviance |
| [`poisson`](losses.md#jaxboost.objective.poisson) | Poisson deviance |
| [`gamma`](losses.md#jaxboost.objective.gamma) | Gamma deviance |

### Multi-class Classification

| Objective | Description |
|-----------|-------------|
| [`softmax_cross_entropy`](losses.md#jaxboost.objective.softmax_cross_entropy) | Standard softmax cross-entropy |
| [`focal_multiclass`](losses.md#jaxboost.objective.focal_multiclass) | Focal loss for multi-class |
| [`label_smoothing`](losses.md#jaxboost.objective.label_smoothing) | Label smoothing regularization |
| [`class_balanced`](losses.md#jaxboost.objective.class_balanced) | Class-balanced loss |

### Survival Analysis

| Objective | Description |
|-----------|-------------|
| [`aft`](losses.md#jaxboost.objective.aft) | Accelerated failure time |
| [`weibull_aft`](losses.md#jaxboost.objective.weibull_aft) | Weibull AFT model |

### Multi-task Learning

| Objective | Description |
|-----------|-------------|
| [`multi_task_regression`](losses.md#jaxboost.objective.multi_task_regression) | Multi-task MSE |
| [`multi_task_classification`](losses.md#jaxboost.objective.multi_task_classification) | Multi-task classification |
| [`multi_task_huber`](losses.md#jaxboost.objective.multi_task_huber) | Multi-task Huber loss |
| [`multi_task_quantile`](losses.md#jaxboost.objective.multi_task_quantile) | Multi-task quantile loss |

### Uncertainty Estimation

| Objective | Description |
|-----------|-------------|
| [`gaussian_nll`](losses.md#jaxboost.objective.gaussian_nll) | Gaussian negative log-likelihood |
| [`laplace_nll`](losses.md#jaxboost.objective.laplace_nll) | Laplace negative log-likelihood |

---

## Module Structure

```
jaxboost/
└── objective/           # Automatic objective functions
    ├── auto.py          # @auto_objective decorator
    ├── binary.py        # Binary classification
    ├── regression.py    # Regression objectives
    ├── multiclass.py    # Multi-class classification
    ├── multi_output.py  # Multi-output (uncertainty)
    ├── multi_task.py    # Multi-task learning
    └── survival.py      # Survival analysis
```
