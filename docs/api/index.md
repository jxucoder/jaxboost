# API Reference

Welcome to the JAXBoost API reference documentation.

## Overview

JAXBoost provides both high-level and low-level APIs:

### High-Level API (Recommended)

- [`GBMTrainer`](training.md#jaxboost.training.GBMTrainer) - Main training interface
- [`TrainerConfig`](training.md#jaxboost.training.TrainerConfig) - Configuration options

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
├── splits/        # Split mechanisms
├── structures/    # Tree structures
├── routing/       # Soft routing
├── aggregation/   # Ensemble aggregation
└── losses/        # Loss functions
```

