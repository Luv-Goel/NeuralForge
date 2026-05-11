# Contributing

We love contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/Luv-Goel/NeuralForge.git
cd NeuralForge
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# Full test suite
pytest tests/ -v --cov=neuralforge

# Quick run (stop on first failure)
pytest tests/ -x -q

# Single test file
pytest tests/test_operations.py -v
```

## Code Style

We use [Black](https://github.com/psf/black) with line length 88 and [isort](https://pycqa.github.io/isort/)
with the Black profile. Pre-commit handles formatting automatically.

```bash
# Format all code
black neuralforge/ tests/
isort neuralforge/ tests/

# Check formatting
black --check neuralforge/ tests/
```

## Linting

```bash
flake8 neuralforge/ --count --select=E9,F63,F7,F82 --show-source --statistics
mypy neuralforge/ --ignore-missing-imports
```

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes with clear commit messages.
3. Ensure all tests pass (`pytest tests/`).
4. Run the formatter (`black neuralforge/ tests/`).
5. Submit a PR with a clear description of changes.
6. Wait for CI checks to pass.
7. Request review when ready.

## Architecture

```
neuralforge/
├── algorithms/       # Search algorithms (DARTS, Evolution, Random)
├── core/             # Core types (operations, genotypes, search space)
├── utils/            # Profiling, visualization, augmentations
├── contrib/          # Third-party integrations (Lightning)
├── surrogate/        # Surrogate predictor for efficiency
└── distributed/      # Distributed population management
```

## Adding a New Algorithm

1. Create `neuralforge/algorithms/your_algorithm.py`.
2. Subclass `SearchAlgorithm` from `neuralforge.algorithms.base`.
3. Implement the `search()` method.
4. Add tests in `tests/`.
5. Register in `neuralforge/algorithms/__init__.py`.

For more details, see [CONTRIBUTING.md](https://github.com/Luv-Goel/NeuralForge/blob/main/CONTRIBUTING.md).
