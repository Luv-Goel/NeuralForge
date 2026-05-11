"""
NeuralForge — Search Space
==========================

Defines the cell-based search space used in DARTS and related NAS methods.

The search space consists of repeated cells (normal + reduction) that form
a CNN. Each cell is a directed acyclic graph (DAG) with:
    - 2 input nodes (outputs of previous two cells)
    - N intermediate nodes (default: 4)
    - 1 output node (depthwise concat of all intermediates)

Each intermediate node receives connections from 2 previous nodes,
each with a chosen operation from the PRIMITIVES set.

This module is the backbone of the entire search — clean abstraction
so that search algorithms (DARTS, Evolution, Random) can operate
on the same representation.

References:
    - DARTS: Differentiable Architecture Search (Liu et al., 2019)
    - SNAS: Stochastic Neural Architecture Search (Xie et al., 2019)
    - NASNet (Zoph et al., 2018)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Type

from neuralforge.core.operations import (
    PRIMITIVES,
    OPS,
    get_op,
    FactorizedReduce,
    ReLUConvBN,
    Stem,
    AuxiliaryHead,
)


@dataclass
class SearchSpaceConfig:
    """Configuration for the cell-based search space.

    Attributes:
        c_in: Number of input channels (3 for RGB images).
        init_channels: Number of channels after the stem conv.
        num_classes: Number of output classes.
        layers: Number of cells in the network.
        nodes: Number of intermediate nodes per cell.
        primitive_set: List of available operation names.
        auxiliary: Whether to use auxiliary head (DARTS-specific).
        auxiliary_weight: Loss weight for auxiliary head.
        drop_path_prob: Drop-path regularization probability.
        stem_multiplier: Channel multiplier for the stem.
    """
    c_in: int = 3
    init_channels: int = 16
    num_classes: int = 10
    layers: int = 8
    nodes: int = 4
    primitive_set: List[str] = field(default_factory=lambda: PRIMITIVES[:])
    auxiliary: bool = False
    auxiliary_weight: float = 0.4
    drop_path_prob: float = 0.2
    stem_multiplier: int = 3

    def __post_init__(self):
        if len(self.primitive_set) < 2:
            raise ValueError("Need at least 2 primitive operations")


class Cell(nn.Module):
    """A single cell in the search space — DAG of operations.

    During search (ARCHITECTURE mode), edges are mixed operations.
    During evaluation (DISCRETE mode), edges are discrete operations
    determined by a Genotype.
    """

    def __init__(
        self,
        steps: int,
        multiplier: int,
        c_prev_prev: int,
        c_prev: int,
        c_curr: int,
        reduction: bool,
        reduction_prev: bool,
        primitive_set: List[str],
    ):
        super().__init__()
        self.steps = steps
        self.multiplier = multiplier
        self.reduction = reduction

        # Handle reduction between cells
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0)

        self._ops = nn.ModuleList()
        self._primitive_set = primitive_set

        # Build the DAG edges
        for i in range(self.steps):
            for j in range(2 + i):
                # Each edge: (node_j -> node_i)
                stride = 2 if reduction and j < 2 else 1
                op = FactorizedReduce(c_curr, c_curr) if (
                    stride == 2 and j < 2 and reduction
                ) else ReLUConvBN(c_curr, c_curr, 1, 1, 0)
                # We'll handle the actual operations externally
                # This is a placeholder — real operations are set during search
                self._ops.append(nn.Identity())

    def forward(self, s0, s1, weights=None):
        """Forward pass.
        
        Args:
            s0: Output of cell at position -2
            s1: Output of cell at position -1
            weights: Architecture weights (used during search with DARTS).
                     If None, uses discrete operations (eval mode).
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(
                self._ops[offset + j](h) * weights[offset + j]
                if weights is not None
                else self._ops[offset + j](h)
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self.steps:], dim=1)


class SearchSpaceCNN(nn.Module):
    """Complete CNN composed of cells from the search space.

    This is the final architecture used during search. It contains:
        - A stem conv to process raw inputs
        - A sequence of normal and reduction cells
        - An auxiliary head (optional, for DARTS-style auxiliary loss)
        - A global average pooling + linear classifier
    """

    def __init__(self, config: SearchSpaceConfig):
        super().__init__()
        self._config = config
        c_curr = config.stem_multiplier * config.init_channels

        self.stem = Stem(config.c_in, c_curr)

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, config.init_channels

        self.cells = nn.ModuleList()
        self._auxiliary = config.auxiliary
        self._auxiliary_weight = config.auxiliary_weight

        reduction_prev = False
        for i in range(config.layers):
            # Reduction cells at 1/3 and 2/3 of the network depth
            if i in [config.layers // 3, 2 * config.layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(
                steps=config.nodes,
                multiplier=config.nodes,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                c_curr=c_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
                primitive_set=config.primitive_set,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, config.nodes * c_curr

            # Attach auxiliary head at 2/3 depth if enabled
            if config.auxiliary and i == 2 * config.layers // 3:
                self.auxiliary_head = AuxiliaryHead(
                    c_prev, config.num_classes,
                )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional auxiliary output."""
        s0 = s1 = self.stem(x)
        aux_logits = None

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if (
                self._auxiliary
                and hasattr(self, "auxiliary_head")
                and i == 2 * self._config.layers // 3
            ):
                aux_logits = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.training and aux_logits is not None:
            return logits, aux_logits
        return logits


class CellSearchSpace:
    """High-level search space interface.

    Provides methods to:
        - Build the search network
        - Parse genotypes into discrete networks
        - Compute search space size
    """

    def __init__(self, config: Optional[SearchSpaceConfig] = None):
        self.config = config or SearchSpaceConfig()

    @property
    def num_operations(self) -> int:
        """Number of primitive operations in this search space."""
        return len(self.config.primitive_set)

    @property
    def total_edges(self) -> int:
        """Total number of edges in the DAG (both normal and reduction cells)."""
        nodes = self.config.nodes
        edges_per_cell = sum(2 + i for i in range(nodes))
        # Two cell types × edges per cell × one op per edge (in discrete mode)
        return 2 * edges_per_cell

    def build_search_network(self) -> SearchSpaceCNN:
        """Build the search network for architecture search."""
        return SearchSpaceCNN(self.config)

    def build_discrete_network(
        self,
        genotype: Any,
        init_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> nn.Module:
        """Build a discrete network from a genotype.
        
        Args:
            genotype: A Genotype object defining the architecture.
            init_channels: Override init_channels.
            num_classes: Override num_classes.
        
        Returns:
            A nn.Module with the discrete architecture.
        """
        from neuralforge.core.genotypes import build_network_from_genotype
        return build_network_from_genotype(
            genotype,
            init_channels=init_channels or self.config.init_channels,
            num_classes=num_classes or self.config.num_classes,
            layers=self.config.layers,
        )

    def __repr__(self) -> str:
        return (
            f"CellSearchSpace("
            f"ops={self.num_operations}, "
            f"nodes={self.config.nodes}, "
            f"layers={self.config.layers}, "
            f"edges={self.total_edges})"
        )
