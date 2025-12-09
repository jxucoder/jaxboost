# jaxboost

Differentiable gradient boosting in JAX.

⚠️ **This is a personal learning project, very much a work in progress.** There is no intention to replace production boosting libraries like XGBoost, LightGBM, or CatBoost. The main purpose is to learn JAX while rethinking gradient boosting from first principles. No guarantee of reliability or correctness for now. Issues welcome!

## What it is

A gradient boosting implementation using soft (differentiable) oblivious trees. The entire model is trained end-to-end with gradient descent via optax, rather than the traditional greedy tree-building approach.

Key characteristics:
- Soft routing with sigmoid functions (trees are differentiable)
- Oblivious tree structure (same split at each level)
- Hyperplane splits (linear combinations of features)
- Runs on GPU via JAX

## Installation

```bash
pip install jaxboost
```

Or from source:

```bash
git clone https://github.com/jxu/jaxboost.git
cd jaxboost
pip install -e .
```

## Usage

```python
from jaxboost import GBMTrainer, TrainerConfig

# Regression
trainer = GBMTrainer(task="regression")
model = trainer.fit(X_train, y_train)
predictions = model.predict(X_test)

# Classification
trainer = GBMTrainer(task="classification")
model = trainer.fit(X_train, y_train)
probabilities = model.predict(X_test)
classes = model.predict_class(X_test)
```

### Configuration

```python
config = TrainerConfig(
    n_trees=20,          # Number of trees
    depth=4,             # Tree depth
    learning_rate=0.01,  # Optimizer learning rate
    epochs=500,          # Training epochs
    patience=50,         # Early stopping patience
    verbose=True,        # Print progress
)
trainer = GBMTrainer(task="regression", config=config)
```

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- optax >= 0.1.7

## License

MIT
