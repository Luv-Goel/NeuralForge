<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Luv-Goel/NeuralForge/main/assets/logo-dark.svg">
    <img alt="NeuralForge" src="https://raw.githubusercontent.com/Luv-Goel/NeuralForge/main/assets/logo-light.svg" width="600">
  </picture>
</p>

<h3 align="center">
  Neural Architecture Search &bull; Model Optimization &bull; AutoML
</h3>

<p align="center">
  <em>A differentiable &amp; evolutionary framework for discovering optimal neural architectures — from research to production.</em>
</p>

<p align="center">
  <a href="https://github.com/Luv-Goel/NeuralForge/actions">
    <img src="https://github.com/Luv-Goel/NeuralForge/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/neuralforge/">
    <img src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue" alt="Python">
  </a>
  <a href="https://github.com/Luv-Goel/NeuralForge/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT">
  </a>
  <a href="https://github.com/Luv-Goel/NeuralForge/stargazers">
    <img src="https://img.shields.io/github/stars/Luv-Goel/NeuralForge" alt="Stars">
  </a>
  <a href="https://github.com/Luv-Goel/NeuralForge/issues">
    <img src="https://img.shields.io/github/issues/Luv-Goel/NeuralForge" alt="Issues">
  </a>
  <a href="https://github.com/Luv-Goel/NeuralForge/pulls">
    <img src="https://img.shields.io/github/issues-pr/Luv-Goel/NeuralForge" alt="PRs">
  </a>
  <br>
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-algorithms">Algorithms</a> •
  <a href="#-documentation">Docs</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🔥 What is NeuralForge?

**NeuralForge** is a modular, research-grade framework for **Neural Architecture Search (NAS)** and **model optimization** built on PyTorch. It implements state-of-the-art NAS algorithms — from differentiable methods like DARTS to evolutionary approaches like Regularized Evolution — in a clean, well-tested, and extensible package.

Think of it as the missing toolbox for **automated architecture discovery**: instead of manually tweaking layer counts and kernel sizes, NeuralForge searches the space of possible architectures for you.

```
                     ┌─────────────────────────────────────┐
                     │         NeuralForge Pipeline         │
                     └─────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│   Search Space   │    │   Search Algorithm   │    │  Best Genotype   │
│  ┌────────────┐  │    │  ┌───────────────┐   │    │  ┌────────────┐  │
│  │ Normal Cell │  │    │  │ DARTS (grad)  │   │    │  │ Architecture│  │
│  │ Reduce Cell │──┼───▶│  │ Evolution     │───┼───▶│  │ Profile     │  │
│  │ OPS: 7-10   │  │    │  │ Random Search │   │    │  │ Visualize   │  │
│  └────────────┘  │    │  └───────────────┘   │    │  └────────────┘  │
└──────────────────┘    └──────────────────────┘    └──────────────────┘
```

### Why NeuralForge?

| Why not just use... | NeuralForge gives you |
|---------------------|----------------------|
| Manual architecture tuning | Automated search across 10⁹+ architectures |
 | Off-the-shelf models | Task-specific architectures optimized for YOUR data |
 | Single search method | 3 algorithms: differentiable, evolutionary, random |
 | Black-box AutoML | Full transparency: inspect, visualize, and export every architecture |

## ✨ Key Features

- **🧬 Differentiable NAS** — Full DARTS implementation with first & second-order optimization, architecture parameter learning via bi-level optimization
- **🌿 Evolutionary Search** — Regularized Evolution (AmoebaNet-style) with tournament selection, age regularization, mutation, and crossover
- **🎲 Baseline Methods** — Random search with statistical rigor (Li & Talwalkar, 2020)
- **🔮 Surrogate Predictors** — MLP and Gaussian Process surrogates to predict architecture performance without full training
- **📊 Model Profiling** — FLOPs, MACs, parameter counts, memory estimation — all built-in
- **🎨 Architecture Visualization** — Plot cells as DAGs, search trajectories, operation importance
- **🧪 Well Tested** — 30+ unit and integration tests, CI with multiple Python versions
- **🔌 Extensible** — Add new operations, search algorithms, or surrogate models with minimal boilerplate

## 📦 Installation

```bash
pip install neuralforge
```

Or from source for the latest:

```bash
git clone https://github.com/Luv-Goel/NeuralForge.git
cd NeuralForge
pip install -e ".[dev]"  # Includes dev dependencies
```

### Dependencies

- Python 3.9+
- PyTorch 1.13+
- torchvision (for CIFAR-10 experiments)
- numpy, scipy, matplotlib, networkx, tqdm

## 🚀 Quick Start

```python
from neuralforge.core.search_space import CellSearchSpace, SearchSpaceConfig
from neuralforge.core.genotypes import random_genotype, build_network_from_genotype
from neuralforge.utils.profiling import profile_model

# 1. Define the search space
config = SearchSpaceConfig(init_channels=16, layers=8, num_classes=10)
space = CellSearchSpace(config)
print(f"Search space: {space}")  # 10 operations × 14 edges per cell type

# 2. Generate a random architecture (or use a search algorithm)
genotype = random_genotype(space.config.primitive_set, nodes=4)
print(genotype.to_compact_string())

# 3. Build the discrete network
model = build_network_from_genotype(
    genotype, init_channels=16, num_classes=10, layers=8,
)

# 4. Profile it
stats = profile_model(model, (3, 32, 32))
print(f"Params: {stats['params_total_str']}, MACs: {stats['macs_str']}")
```

### Running DARTS Search

```python
from neuralforge.algorithms.darts import DARTSSearch
from neuralforge.algorithms.base import SearchConfig

# Configure search
search_cfg = SearchConfig(
    epochs=50, batch_size=64, init_channels=16,
    layers=8, learning_rate=0.025,
)

# Create searcher
searcher = DARTSSearch(space, search_cfg)

# Run search (requires CIFAR-10 via torchvision)
best_genotype = searcher.search()

# Export results
searcher.export_results("results.json")
print(best_genotype.to_compact_string())
```

### Evolutionary Search

```python
from neuralforge.algorithms.evolution import EvolutionSearch, EvolutionConfig

evo_cfg = EvolutionConfig(
    population_size=50, tournament_size=10,
    cycles=200, mutation_rate=0.3,
)

searcher = EvolutionSearch(space, evo_cfg)
best_genotype = searcher.search(evaluate_fn=my_evaluator)
```

## 🧠 Algorithms

### DARTS (Differentiable Architecture Search)

The flagship algorithm: continuously relax the discrete architecture decision by replacing operation choices with softmax-weighted combinations. Architecture parameters (α) are learned via **bi-level optimization**:

```
min_α  L_val(w*(α), α)
s.t.   w*(α) = argmin_w L_train(w, α)
```

- **First-order**: Approximate gradient, fast (1-2 GPU-days on CIFAR-10)
- **Second-order**: Unrolled gradient, more accurate (3-4 GPU-days)
- Supports auxiliary heads for improved gradient flow

### Regularized Evolution

Evolutionary search with **age regularization**: younger models are favored in tournament selection, preventing the population from stagnating. The algorithm:

1. Initialize population of random architectures
2. Evaluate each (partial training)
3. Tournament select parent → mutate → evaluate child
4. **Remove oldest individual** → add child
5. Repeat until budget exhausted

Key advantage: simpler than DARTS, no gradients needed, works with any search space.

### Random Search

The often-overlooked baseline. Li & Talwalkar (2020) showed that random search with the same compute budget matches many published NAS methods. We include it as a rigorous comparison point.

## 🎨 Visualization

```python
from neuralforge.utils.visualization import plot_architecture, render_cell_graph

# Text representation
print(render_cell_graph(genotype))

# Save architecture DAG plot
plot_architecture(genotype, filename="my_architecture.png")
```

Architecture plots show the directed acyclic graph with color-coded operations:

- 🔵 **Separable convolutions** (3×3, 5×5) — core feature extractors
- 🟢 **Skip connections** — gradient highways
- 🟡 **Dilated convolutions** — multi-scale context
- 🔴 **Pooling** — spatial reduction
- 🟣 **1×1 convolutions** — channel mixing

## 📊 Model Profiling

```python
from neuralforge.utils.profiling import profile_model

stats = profile_model(model, (3, 224, 224))
# {
#   'params_total_str': '5.2M',
#   'macs_str': '850.3M',
#   'flops_str': '1.7G',
#   'memory_mb': {'params_mb': 20.8, 'activations_mb': 45.3, 'total_mb': 66.1}
# }
```

## 🏗️ Project Structure

```
NeuralForge/
├── neuralforge/
│   ├── core/              # Search space, operations, genotypes
│   │   ├── operations.py  # 10+ primitive operations (conv, pool, skip, etc.)
│   │   ├── search_space.py # Cell-based DAG search space
│   │   └── genotypes.py   # Genotype encoding, mutation, crossover
│   ├── algorithms/        # Search algorithms
│   │   ├── darts.py       # Differentiable Architecture Search
│   │   ├── evolution.py   # Regularized Evolution (AmoebaNet)
│   │   └── random_search.py # Random search baseline
│   ├── surrogate/         # Performance predictors
│   │   └── predictor.py   # MLP & Gaussian Process surrogates
│   ├── utils/             # Profiling & visualization
│   ├── distributed/       # Distributed population management
│   └── contrib/           # PyTorch Lightning integration
├── tests/                 # 30+ tests
├── examples/              # End-to-end examples
└── docs/                  # Documentation
```

## 📚 Papers Implemented

| Paper | Venue | Module |
|-------|-------|--------|
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR 2019 | `algorithms/darts.py` |
| [Regularized Evolution for Image Classifier Arch. Search](https://arxiv.org/abs/1802.01548) | AAAI 2019 | `algorithms/evolution.py` |
| [Random Search and Reproducibility for NAS](https://arxiv.org/abs/1902.07638) | UAI 2020 | `algorithms/random_search.py` |
| [NAS-Bench-101](https://arxiv.org/abs/1902.09635) | CVPR 2019 | `core/search_space.py` |
| [Understanding and Simplifying DARTS](https://arxiv.org/abs/2101.08069) | NeurIPS 2021 | `algorithms/darts.py` |

## 🛣️ Roadmap

- [x] DARTS (first & second-order)
- [x] Regularized Evolution
- [x] Random Search baseline
- [x] Surrogate predictors
- [x] Model profiling (FLOPs, MACs, params)
- [x] Architecture visualization
- [ ] PC-DARTS & ProxylessNAS
- [ ] Weight-sharing (ENAS, One-Shot)
- [ ] Multi-objective search (latency-accuracy Pareto)
- [ ] ONNX export for discovered architectures
- [ ] Distributed search across multiple GPUs
- [ ] Integration with HuggingFace Hub

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

Areas we'd love help with:
- New search algorithms (ENAS, ProxylessNAS, SPOS)
- Transformer/NLP search spaces
- Benchmark integrations (NAS-Bench-201, TransNAS-Bench)
- Performance optimizations
- Documentation improvements

## 📄 License

This project is **MIT** licensed. See [LICENSE](LICENSE) for details.

## ⭐ Citation

If you use NeuralForge in your research, please cite:

```bibtex
@misc{goel2024neuralforge,
  author = {Goel, Luv},
  title = {NeuralForge: A Neural Architecture Search & Model Optimization Framework},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Luv-Goel/NeuralForge}
}
```

---

<p align="center">
  Built with 🔥 and PyTorch &nbsp;|&nbsp; 
  <a href="https://github.com/Luv-Goel/NeuralForge/issues">Report Bug</a> •
  <a href="https://github.com/Luv-Goel/NeuralForge/discussions">Discussions</a>
</p>
