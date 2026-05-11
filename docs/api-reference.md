# API Reference

## Core

### `CellSearchSpace(config)`
The search space defining valid architectures.

### `Genotype(normal, reduce)`
A discoverable architecture — compact, serializable.

## Algorithms

### `DARTSSearch(space, config)`
Differentiable Architecture Search.

### `EvolutionSearch(space, config)`
Regularized Evolution search.

### `RandomSearch(space, n_samples)`
Random search baseline.