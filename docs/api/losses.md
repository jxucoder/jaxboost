# Objectives

Automatic objective functions for XGBoost and LightGBM using JAX autodiff.

!!! tip "How it works"
    JAXBoost uses JAX automatic differentiation to compute gradients and Hessians from your loss function. You write the loss, JAX computes the derivatives—no manual math required.

---

## Core Classes

These are the building blocks for creating custom objectives.

### AutoObjective

The primary class for scalar loss functions (binary classification, regression).

::: jaxboost.objective.AutoObjective

---

### MultiClassObjective

For multi-class classification problems where predictions are logit vectors.

::: jaxboost.objective.MultiClassObjective

---

### MultiOutputObjective

For multi-output predictions like uncertainty estimation (mean + variance).

::: jaxboost.objective.MultiOutputObjective

---

### MaskedMultiTaskObjective

For multi-task learning with potentially missing labels.

::: jaxboost.objective.MaskedMultiTaskObjective

---

## Binary Classification

Objectives for binary classification tasks (labels in {0, 1}).

::: jaxboost.objective.focal_loss

::: jaxboost.objective.binary_crossentropy

::: jaxboost.objective.weighted_binary_crossentropy

::: jaxboost.objective.hinge_loss

---

## Regression

Objectives for continuous target prediction.

### Standard

::: jaxboost.objective.mse

::: jaxboost.objective.huber

::: jaxboost.objective.pseudo_huber

::: jaxboost.objective.log_cosh

::: jaxboost.objective.mae_smooth

### Quantile & Asymmetric

::: jaxboost.objective.quantile

::: jaxboost.objective.asymmetric

### Distribution-Based

::: jaxboost.objective.tweedie

::: jaxboost.objective.poisson

::: jaxboost.objective.gamma

---

## Multi-class Classification

Objectives for classification with more than two classes.

::: jaxboost.objective.softmax_cross_entropy

::: jaxboost.objective.focal_multiclass

::: jaxboost.objective.label_smoothing

::: jaxboost.objective.class_balanced

---

## Survival Analysis

Objectives for time-to-event modeling.

::: jaxboost.objective.aft

::: jaxboost.objective.weibull_aft

---

## Multi-task Learning

Objectives for predicting multiple targets simultaneously.

::: jaxboost.objective.multi_task_regression

::: jaxboost.objective.multi_task_classification

::: jaxboost.objective.multi_task_huber

::: jaxboost.objective.multi_task_quantile

---

## Uncertainty Estimation

Multi-output objectives that predict both value and uncertainty.

::: jaxboost.objective.gaussian_nll

::: jaxboost.objective.laplace_nll
