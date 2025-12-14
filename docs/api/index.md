# API Reference

Welcome to the JAXBoost API reference documentation.

## Overview

JAXBoost provides both high-level and low-level APIs:

### High-Level API (Recommended)

- [`GBMTrainer`](training.md#jaxboost.training.GBMTrainer) - Main training interface
- [`TrainerConfig`](training.md#jaxboost.training.TrainerConfig) - Configuration options

### Ensemble & Mixture of Experts

- [`MOEEnsemble`](ensemble.md#jaxboost.ensemble.MOEEnsemble) - Differentiable MOE with soft tree experts
- [`EMMOE`](ensemble.md#jaxboost.ensemble.hybrid_moe.EMMOE) - EM-trained MOE with XGBoost/LightGBM/CatBoost experts
- [Gating Networks](ensemble.md) - Linear, MLP, and Tree gating

### Low-Level Components

- [Splits](splits.md) - Split functions (axis-aligned, hyperplane)
- [Structures](structures.md) - Tree structures (oblivious trees)
- [Routing](routing.md) - Soft routing functions
- [Aggregation](aggregation.md) - Boosting aggregation
- [Losses](losses.md) - Loss functions

## Module Structure

```
jaxboost/
├── training/      # High-level training API
├── ensemble/      # MOE architectures (differentiable & hybrid)
├── splits/        # Split mechanisms
├── structures/    # Tree structures
├── routing/       # Soft routing
├── aggregation/   # Ensemble aggregation
└── losses/        # Loss functions
```

