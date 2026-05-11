# Installation

## Quick Install

```bash
pip install neuralforge
```

## From Source

```bash
git clone https://github.com/Luv-Goel/NeuralForge.git
cd NeuralForge
pip install -e ".[dev]"
```

## Dependencies

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| PyTorch | 1.13 | Deep learning framework |
| torchvision | 0.14 | Datasets & transforms |
| NumPy | 1.21 | Numerical computing |
| SciPy | 1.7 | Scientific computing |
| tqdm | 4.62 | Progress bars |
| PyYAML | 6.0 | Config parsing |
| Matplotlib | 3.5 | Visualization |
| NetworkX | 2.8 | Graph operations |

### Optional Dependencies

- **dev**: pytest, black, isort, flake8, mypy, pre-commit
- **lightning**: pytorch-lightning >= 1.8
- **surrogate**: scikit-learn >= 1.0, gpytorch >= 1.9

Install all optional dependencies:

```bash
pip install -e ".[all]"
```

## Verifying Installation

```python
import neuralforge
print(neuralforge.__version__)
```

Run the test suite:

```bash
pytest tests/ -v
```
