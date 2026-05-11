# Contributing to NeuralForge

First off, thanks for taking the time to contribute! 🎉

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you're expected to uphold it.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/NeuralForge.git`
3. Install in dev mode: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install  # Optional but recommended
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=neuralforge  # With coverage
```

## Code Style

- We use [Black](https://github.com/psf/black) for formatting
- [isort](https://github.com/PyCQA/isort) for import sorting
- [flake8](https://github.com/PyCQA/flake8) for linting
- Type hints are required for all public APIs

Run formatting:
```bash
black neuralforge tests
isort neuralforge tests
```

## Pull Request Process

1. Update docs if needed
2. Add tests for new functionality
3. Ensure CI passes
4. Update the README if your change affects the API

## Adding New Search Algorithms

1. Create a new file in `neuralforge/algorithms/`
2. Inherit from `SearchAlgorithm` base class
3. Implement the `search()` method
4. Add tests in `tests/`
5. Register in `neuralforge/algorithms/__init__.py`

## Adding New Operations

1. Add the primitive name to `PRIMITIVES` in `neuralforge/core/operations.py`
2. Add the factory function to `OPS` dict
3. Add a visualization color/symbol in `visualization.py`
4. Test that forward pass works

## Questions?

Open an issue with the `question` label. We're friendly!
