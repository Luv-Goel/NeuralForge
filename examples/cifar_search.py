"""
CIFAR-10 Neural Architecture Search — Full Example.

This is a complete search pipeline using DARTS on CIFAR-10.
It demonstrates:
    - Data loading with augmentations
    - Search with architecture parameter updates
    - Genotype extraction and logging
    - Result export

Requires: torchvision (for CIFAR-10), GPU recommended.

Usage:
    python examples/cifar_search.py --epochs 50 --gpu 0
"""

import argparse
import logging
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from neuralforge.core.search_space import CellSearchSpace, SearchSpaceConfig
from neuralforge.algorithms.darts import DARTSSearch
from neuralforge.algorithms.base import SearchConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser("DARTS CIFAR-10 Search")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--init_channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--unrolled", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./search_results")
    args = parser.parse_args()

    # Validate GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU (very slow!)")
        device = "cpu"
    else:
        device = f"cuda:{args.gpu}" if torch.cuda.device_count() > args.gpu else "cuda:0"

    # Search space
    space_config = SearchSpaceConfig(
        init_channels=args.init_channels,
        layers=args.layers,
        num_classes=10,
    )
    space = CellSearchSpace(space_config)
    logger.info(f"Search space: {space}")

    # Search config
    search_config = SearchConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        init_channels=args.init_channels,
        layers=args.layers,
        device=device,
        output_dir=args.output_dir,
        unrolled=args.unrolled,
        report_freq=50,
    )

    # Run search
    searcher = DARTSSearch(space, search_config)

    # Load CIFAR-10
    logger.info("Loading CIFAR-10...")
    train_data = searcher._load_cifar10()
    logger.info(f"Loaded {len(train_data)} training images")

    # Search!
    logger.info("Starting search...")
    best_genotype = searcher.search(train_data=train_data)

    # Results
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)
    print(best_genotype.to_compact_string())
    print()

    # Export
    searcher.export_results(os.path.join(args.output_dir, "results.json"))
    logger.info(f"Results exported to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
