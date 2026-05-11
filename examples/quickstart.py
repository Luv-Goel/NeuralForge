"""
Quickstart example — Neural Architecture Search with NeuralForge.

This script demonstrates the basic workflow:
    1. Create a search space
    2. Run a search algorithm (DARTS)
    3. Profile the best architecture
    4. Visualize the result

For a full search you'll need a GPU, but the demo shows
the API surface and data flow even on CPU.
"""

import torch
import logging

from neuralforge.core.search_space import CellSearchSpace, SearchSpaceConfig
from neuralforge.algorithms.darts import DARTSSearch
from neuralforge.algorithms.base import SearchConfig
from neuralforge.utils.profiling import profile_model
from neuralforge.utils.visualization import (
    plot_architecture,
    render_cell_graph,
)

logging.basicConfig(level=logging.INFO)


def quickstart_demo():
    """Run a minimal architecture search demo."""
    print("=" * 60)
    print("NeuralForge — Quickstart Demo")
    print("=" * 60)

    # 1. Configure the search space
    print("\n[1/4] Configuring search space...")
    search_config = SearchSpaceConfig(
        init_channels=8,      # Small channels for speed
        layers=6,              # Fewer layers for speed
        nodes=4,               # Intermediate nodes per cell
        num_classes=10,
    )
    space = CellSearchSpace(search_config)
    print(f"   Search space: {space}")
    print(f"   Operations: {space.num_operations}")
    print(f"   Total edges: {space.total_edges}")

    # 2. Configure and run DARTS search
    print("\n[2/4] Running DARTS search (CPU-simulated)...")
    algo_config = SearchConfig(
        epochs=2,              # Just 2 epochs for demo
        batch_size=16,
        init_channels=8,
        layers=6,
        learning_rate=0.025,
        device="cpu",
        report_freq=10,
    )
    searcher = DARTSSearch(space, algo_config)

    # In a real run, you'd pass train_data=...
    # For demo, we just show the search setup
    print(f"   Algorithm: {searcher.__class__.__name__}")
    print(f"   Device: {searcher.device}")
    print(f"   Epochs: {algo_config.epochs}")
    print(f"   Output dir: {searcher._output_dir}")

    # 3. Profile the search network
    print("\n[3/4] Profiling the search network...")
    net = space.build_search_network()
    net.eval()
    try:
        results = profile_model(net, (3, 32, 32), verbose=True)
    except Exception as e:
        print(f"   Profiling skipped (expected on CPU): {e}")

    # 4. Visualize a random architecture
    print("\n[4/4] Generating architecture visualization...")
    from neuralforge.core.genotypes import random_genotype
    from neuralforge.core.operations import PRIMITIVES

    geno = random_genotype(PRIMITIVES, nodes=4)
    print("\n" + render_cell_graph(geno))
    print("\n" + "=" * 60)
    print("Quickstart complete! For a real search:")
    print("  - Use a GPU for meaningful results")
    print("  - Set epochs=50 for DARTS")
    print("  - Provide train_data (CIFAR-10 via torchvision)")
    print("  - Or use EvolutionSearch with evaluate_fn")
    print("=" * 60)


if __name__ == "__main__":
    quickstart_demo()
