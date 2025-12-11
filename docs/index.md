# JAXBoost

**Next-generation differentiable gradient boosting with JAX**

## Features

- 🌳 **Soft Oblivious Trees** - Differentiable tree structures with sigmoid routing
- ✂️ **Hyperplane Splits** - Capture feature interactions beyond axis-aligned splits
- 🚀 **GPU-Efficient** - Vectorized computation leveraging JAX's JIT compilation
- 🔄 **End-to-End Training** - Gradient-based optimization via optax

## Installation

```bash
pip install jaxboost
```

## Quick Example

```python
from jaxboost import GBMTrainer

# Create trainer
trainer = GBMTrainer(task="regression")

# Fit model
model = trainer.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Next Steps

- [Quick Start Guide](getting-started/quickstart.md) - Get started with JAXBoost
- [API Reference](api/index.md) - Detailed API documentation

