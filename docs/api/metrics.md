# Metrics

Evaluation metrics for XGBoost/LightGBM that work with custom objectives.

!!! warning "Important: Disable Default Metrics"
    When using custom objectives, XGBoost's default evaluation metrics may not be meaningful.
    Always set `'disable_default_eval_metric': 1` in params for XGBoost, or `'metric': 'None'` for LightGBM.

---

## Quick Start

```python
from jaxboost.objective import ordinal_logit
from jaxboost.metric import qwk_metric

# Create ordinal objective
ordinal = ordinal_logit(n_classes=6)
ordinal.init_thresholds_from_data(y_train)

# Train with custom metric
model = xgb.train(
    {'disable_default_eval_metric': 1, 'max_depth': 4},
    dtrain,
    obj=ordinal.xgb_objective,
    custom_metric=ordinal.qwk_metric.xgb_metric,  # Built-in metric
    evals=[(dtest, 'test')]
)
```

---

## Base Classes

### Metric

::: jaxboost.metric.Metric

### make_metric

::: jaxboost.metric.make_metric

---

## Ordinal Metrics

For ordered categorical outcomes (ratings, grades, severity levels).

### qwk_metric

::: jaxboost.metric.qwk_metric

### ordinal_mae_metric

::: jaxboost.metric.ordinal_mae_metric

### ordinal_accuracy_metric

::: jaxboost.metric.ordinal_accuracy_metric

### adjacent_accuracy_metric

::: jaxboost.metric.adjacent_accuracy_metric

---

## Classification Metrics

For binary classification problems.

### auc_metric

::: jaxboost.metric.auc_metric

### log_loss_metric

::: jaxboost.metric.log_loss_metric

### accuracy_metric

::: jaxboost.metric.accuracy_metric

### f1_metric

::: jaxboost.metric.f1_metric

### precision_metric

::: jaxboost.metric.precision_metric

### recall_metric

::: jaxboost.metric.recall_metric

---

## Regression Metrics

For continuous target prediction.

### mse_metric

::: jaxboost.metric.mse_metric

### rmse_metric

::: jaxboost.metric.rmse_metric

### mae_metric

::: jaxboost.metric.mae_metric

### r2_metric

::: jaxboost.metric.r2_metric

---

## Bounded Regression Metrics

For proportion/rate prediction in [0, 1].

### bounded_mse_metric

::: jaxboost.metric.bounded_mse_metric

### out_of_bounds_metric

::: jaxboost.metric.out_of_bounds_metric

---

## XGBoost vs LightGBM Interface

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Metric param | `custom_metric=` | `feval=` |
| Metric method | `.xgb_metric` | `.lgb_metric` |
| Return value | `(name, value)` | `(name, value, is_higher_better)` |
| Disable default | `'disable_default_eval_metric': 1` | `'metric': 'None'` |

### XGBoost Example

```python
model = xgb.train(
    {'disable_default_eval_metric': 1},
    dtrain,
    obj=objective.xgb_objective,
    custom_metric=metric.xgb_metric,
    evals=[(dtest, 'test')]
)
```

### LightGBM Example

```python
model = lgb.train(
    {'metric': 'None'},
    train_data,
    fobj=objective.lgb_objective,
    feval=metric.lgb_metric,
    valid_sets=[valid_data]
)
```
