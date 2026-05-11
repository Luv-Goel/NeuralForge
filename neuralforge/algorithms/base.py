"""
NeuralForge — Base Search Algorithm
====================================

Abstract base class for all architecture search algorithms.
Provides common infrastructure:
    - Training loop with configurable batch size, epochs, learning rate
    - Data loading with CIFAR-10 support
    - Model checkpointing
    - Metric logging
    - Genotype export

Subclasses implement `search()` which runs the core search logic.
"""

from __future__ import annotations
import abc
import copy
import json
import os
import time
import logging
from pathlib import Path
from typing import (
    Optional, Callable, Dict, Any, List, Tuple, Union,
)
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from neuralforge.core.search_space import (
    SearchSpaceConfig,
    CellSearchSpace,
    SearchSpaceCNN,
)
from neuralforge.core.genotypes import Genotype

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for architecture search.

    Attributes:
        epochs: Number of search epochs.
        batch_size: Training batch size.
        init_channels: Base channel count.
        layers: Number of cells in the network.
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        learning_rate_min: Minimum learning rate (cosine schedule).
        momentum: SGD momentum.
        weight_decay: L2 weight decay.
        grad_clip: Gradient clipping norm.
        train_portion: Fraction of training data for weight training.
        unrolled: Whether to use second-order (unrolled) gradients.
        arch_learning_rate: Learning rate for architecture params.
        arch_weight_decay: Weight decay for architecture params.
        report_freq: Logging frequency (in steps).
        save_freq: Checkpoint frequency (in epochs).
        output_dir: Directory for logs and checkpoints.
        device: Device to use ('cuda' or 'cpu').
        seed: Random seed.
        cutout: Cutout augmentation length (0 = disabled).
        drop_path_prob: Drop-path probability.
        grad_clip: Max gradient norm.
    """
    epochs: int = 50
    batch_size: int = 64
    init_channels: int = 16
    layers: int = 8
    num_classes: int = 10
    learning_rate: float = 0.025
    learning_rate_min: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 3e-4
    grad_clip: float = 5.0
    train_portion: float = 0.5
    unrolled: bool = False
    arch_learning_rate: float = 3e-4
    arch_weight_decay: float = 1e-3
    report_freq: int = 50
    save_freq: int = 10
    output_dir: str = "./search_output"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    cutout: int = 0
    drop_path_prob: float = 0.0
    eager: bool = True  # eager validation after each epoch


class SearchAlgorithm(abc.ABC):
    """Abstract base for NAS search algorithms.

    Subclasses must implement:
        - search() → runs the search and returns best Genotype
        
    Override:
        - _initialize_search() → custom init logic
        - _search_step() → one step/epoch of search
        - _evaluate_population() → evaluate found architectures
    """

    def __init__(
        self,
        search_space: CellSearchSpace,
        config: Optional[SearchConfig] = None,
    ):
        self.search_space = search_space
        self.config = config or SearchConfig()

        # Setup
        self.device = torch.device(self.config.device)
        self._setup_logging()
        self._set_seed()
        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._best_genotype: Optional[Genotype] = None
        self._history: List[Dict[str, Any]] = []
        self._start_time: float = 0.0

        logger.info(
            f"Initialized {self.__class__.__name__} on {self.device} "
            f"with {self.search_space}"
        )

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # Note: there's still nondeterminism from CuDNN
        torch.backends.cudnn.benchmark = True

    # ——— Abstract ———

    @abc.abstractmethod
    def search(
        self,
        train_data: Optional[Dataset] = None,
        **kwargs,
    ) -> Genotype:
        """Run architecture search.
        
        Args:
            train_data: Training dataset. If None, loads CIFAR-10.
            **kwargs: Algorithm-specific parameters.
        
        Returns:
            The best discovered genotype.
        """
        ...

    # —── Utilities ————

    def _make_dataloaders(
        self,
        dataset: Dataset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Split dataset into train and validation loaders."""
        n = len(dataset)
        n_train = int(n * self.config.train_portion)
        n_val = n - n_train
        train_data, val_data = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=2,
        )
        return train_loader, val_loader

    def _accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Compute top-1 accuracy."""
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0)

    def _save_checkpoint(
        self,
        state: Dict[str, Any],
        filename: str = "checkpoint.pt",
    ) -> None:
        path = self._output_dir / filename
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

    def _log_metrics(
        self,
        step: int,
        epoch: int,
        loss: float,
        accuracy: float,
        phase: str = "train",
    ) -> None:
        self._history.append({
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "phase": phase,
            "time": time.time() - self._start_time,
        })

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Search history as list of metric dicts."""
        return self._history.copy()

    @property
    def best_genotype(self) -> Optional[Genotype]:
        """Best discovered genotype so far."""
        return self._best_genotype

    def export_results(self, path: Optional[str] = None) -> str:
        """Export search results to JSON.
        
        Args:
            path: Output file path. If None, returns JSON string.
        
        Returns:
            JSON string of results.
        """
        results = {
            "algorithm": self.__class__.__name__,
            "config": self.config.__dict__,
            "best_genotype": (
                self._best_genotype.to_dict() if self._best_genotype else None
            ),
            "history": self._history,
            "search_space": str(self.search_space),
        }
        json_str = json.dumps(results, indent=2)
        if path:
            Path(path).write_text(json_str)
        return json_str
