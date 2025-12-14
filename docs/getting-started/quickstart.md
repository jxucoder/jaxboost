# Quick Start

This guide will help you get started with JAXBoost.

## Installation

```bash
pip install jaxboost
```

## Basic Usage

### Regression

```python
import jax.numpy as jnp
from jaxboost import GBMTrainer

# Generate sample data
X_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y_train = jnp.array([1.0, 2.0, 3.0])

# Create and fit trainer
trainer = GBMTrainer(task="regression")
model = trainer.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_train)
```

### Binary Classification

```python
from jaxboost import GBMTrainer

trainer = GBMTrainer(task="binary_classification")
model = trainer.fit(X_train, y_train)

# Get probability predictions
probabilities = model.predict_proba(X_train)
```

## Configuration

Use `TrainerConfig` for advanced configuration:

```python
from jaxboost import GBMTrainer, TrainerConfig

config = TrainerConfig(
    n_trees=100,
    max_depth=4,
    learning_rate=0.1,
)

trainer = GBMTrainer(config=config, task="regression")
```

## Mixture of Experts

For heterogeneous data with distinct patterns:

```python
from jaxboost.ensemble import EMMOE, EMConfig, create_xgboost_expert

experts = [
    create_xgboost_expert(task="regression", n_estimators=100)
    for _ in range(4)
]

config = EMConfig(num_experts=4, em_iterations=10)
moe = EMMOE(experts, config=config)
moe.fit(X_train, y_train)

predictions = moe.predict(X_test)
```

See [Ensemble API](../api/ensemble.md) for more details.

