"""
NeuralForge — Genotypes
=======================

Genotype encoding/decoding for neural architectures.
A Genotype is a compact, serializable representation of a discovered
architecture. It can be saved to JSON, logged, or used to reconstruct
the full network.

The representation follows DARTS convention:
    - normal: list of (op_name, from_node) tuples for normal cells
    - normal_concat: which intermediate nodes to concatenate
    - reduce: same structure but for reduction cells
    - reduce_concat: which nodes to concatenate for reduction cells

We also provide utilities for mutating genotypes (used in
evolutionary search) and converting between representations.

References:
    - DARTS: Differentiable Architecture Search (ICLR 2019)
    - SNAS: Stochastic NAS (ICLR 2019)
    - AmoebaNet (Real et al., 2019)
"""

from __future__ import annotations
import copy
import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn

from neuralforge.core.operations import (
    OPS,
    PRIMITIVES,
    get_op,
    FactorizedReduce,
    ReLUConvBN,
    Stem,
    AuxiliaryHead,
)


@dataclass
class Genotype:
    """A discoverable neural architecture.

    Attributes:
        normal: List of (op_name, input_node) for the normal cell.
        normal_concat: Indices of nodes to concatenate for output.
        reduce: List of (op_name, input_node) for the reduction cell.
        reduce_concat: Indices of nodes to concatenate for output.
        accuracy: Validation accuracy (populated after evaluation).
        params: Parameter count (populated after profiling).
        flops: FLOP count (populated after profiling).
    """
    normal: List[Tuple[str, int]] = field(default_factory=list)
    normal_concat: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    reduce: List[Tuple[str, int]] = field(default_factory=list)
    reduce_concat: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    accuracy: float = 0.0
    params: float = 0.0
    flops: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "normal": [[op, idx] for op, idx in self.normal],
            "normal_concat": self.normal_concat,
            "reduce": [[op, idx] for op, idx in self.reduce],
            "reduce_concat": self.reduce_concat,
            "accuracy": self.accuracy,
            "params": self.params,
            "flops": self.flops,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Genotype":
        """Create from a dict (e.g., loaded from JSON)."""
        return cls(
            normal=[(op, idx) for op, idx in data["normal"]],
            normal_concat=data.get("normal_concat", [2, 3, 4, 5]),
            reduce=[(op, idx) for op, idx in data["reduce"]],
            reduce_concat=data.get("reduce_concat", [2, 3, 4, 5]),
            accuracy=data.get("accuracy", 0.0),
            params=data.get("params", 0.0),
            flops=data.get("flops", 0.0),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_compact_string(self) -> str:
        """Human-readable compact representation.
        
        Example output:
            Normal: sep_conv_3x3->0, skip_connect->0, sep_conv_5x5->1, ...
            Reduce: max_pool_3x3->0, dil_conv_3x3->0, sep_conv_3x3->1, ...
        """
        def _fmt(ops):
            return ", ".join(f"{op}->{idx}" for op, idx in ops)
        return (
            f"Normal: {_fmt(self.normal)}\n"
            f"Reduce: {_fmt(self.reduce)}\n"
            f"Acc: {self.accuracy:.4f}, Params: {self.params:.1f}M, "
            f"FLOPs: {self.flops:.1f}M"
        )

    def __repr__(self) -> str:
        return (
            f"Genotype(acc={self.accuracy:.4f}, "
            f"params={self.params:.1f}M, flops={self.flops:.1f}M)"
        )


# ——— Genotype mutation operators (for evolutionary search) ———

def mutate_genotype(
    genotype: Genotype,
    primitive_set: List[str],
    mutation_rate: float = 0.3,
) -> Genotype:
    """Mutate a genotype by randomly changing operations.
    
    Each edge's operation has `mutation_rate` chance of being replaced
    by a random different operation. The DAG structure (which nodes
    connect where) is preserved — only the operation type changes.
    
    This is the 'structural mutation' used in Regularized Evolution
    (AmoebaNet, Real et al., 2019).
    
    Args:
        genotype: The parent genotype.
        primitive_set: Available operations.
        mutation_rate: Per-edge mutation probability.
    
    Returns:
        A new Genotype with mutations applied.
    """
    def _mutate_edges(edges: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        new_edges = []
        for op_name, node_idx in edges:
            if random.random() < mutation_rate:
                # Pick a different operation
                candidates = [p for p in primitive_set if p != op_name]
                if candidates:
                    new_edges.append((random.choice(candidates), node_idx))
                else:
                    new_edges.append((op_name, node_idx))
            else:
                new_edges.append((op_name, node_idx))
        return new_edges

    return Genotype(
        normal=_mutate_edges(genotype.normal),
        normal_concat=genotype.normal_concat[:],
        reduce=_mutate_edges(genotype.reduce),
        reduce_concat=genotype.reduce_concat[:],
    )


def crossover_genotypes(
    parent_a: Genotype,
    parent_b: Genotype,
) -> Genotype:
    """One-point crossover between two genotypes.
    
    Each edge is independently inherited from one of the two parents
    with equal probability. This preserves the DAG structure while
    mixing operation choices.
    """
    def _crossover_edges(
        edges_a: List[Tuple[str, int]],
        edges_b: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        assert len(edges_a) == len(edges_b)
        return [
            random.choice([a, b])
            for a, b in zip(edges_a, edges_b)
        ]

    return Genotype(
        normal=_crossover_edges(parent_a.normal, parent_b.normal),
        normal_concat=parent_a.normal_concat[:],
        reduce=_crossover_edges(parent_a.reduce, parent_b.reduce),
        reduce_concat=parent_a.reduce_concat[:],
    )


def random_genotype(
    primitive_set: List[str],
    nodes: int = 4,
) -> Genotype:
    """Generate a random genotype for the given search space.
    
    Each intermediate node connects to exactly 2 previous nodes
    (uniformly sampled), each with a random operation.
    This produces valid architectures in the DARTS search space.
    """
    normal_edges = _sample_edges(primitive_set, nodes)
    reduce_edges = _sample_edges(primitive_set, nodes)
    return Genotype(
        normal=normal_edges,
        normal_concat=list(range(2, 2 + nodes)),
        reduce=reduce_edges,
        reduce_concat=list(range(2, 2 + nodes)),
    )


def _sample_edges(
    primitive_set: List[str],
    nodes: int,
) -> List[Tuple[str, int]]:
    """Sample random edges for a cell with `nodes` intermediate nodes."""
    edges = []
    for i in range(nodes):
        # Node i connects to 2 of its predecessors
        predecessors = list(range(2 + i))
        # Sample 2 distinct predecessors (or 1 if only 1 available)
        k = min(2, len(predecessors))
        chosen = random.sample(predecessors, k)
        for pred in chosen:
            op = random.choice(primitive_set)
            edges.append((op, pred))
    return edges


# ——— Discrete network construction ———

class DiscreteCell(nn.Module):
    """A cell with discrete operations determined by a genotype."""

    def __init__(
        self,
        cell_edges: List[Tuple[str, int]],
        concat_indices: List[int],
        c_prev_prev: int,
        c_prev: int,
        c_curr: int,
        reduction: bool,
        reduction_prev: bool,
    ):
        super().__init__()
        self._concat = concat_indices
        self.multiplier = len(concat_indices)

        # Preprocessing
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c_prev_prev, c_curr)
        else:
            self.preprocess0 = ReLUConvBN(c_prev_prev, c_curr, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c_prev, c_curr, 1, 1, 0)

        # Build discrete operations from genotype edges
        # DARTS format: flat list of (op, from_node) pairs ordered by target node
        self._node_ops = nn.ModuleList()
        self._node_inputs = []

        start_idx = 0
        for node_idx in range(len(concat_indices)):
            k = min(2, node_idx + 2)  # each intermediate node connects to k predecessors
            node_ops = nn.ModuleList()
            node_from = []
            for e_idx in range(k):
                global_idx = start_idx + e_idx
                if global_idx < len(cell_edges):
                    op_name, from_idx = cell_edges[global_idx]
                    node_from.append(from_idx)
                    stride = 2 if reduction and from_idx < 2 else 1
                    op = get_op(op_name, c_curr, c_curr, stride, affine=True)
                    node_ops.append(op)
                else:
                    node_from.append(from_idx if global_idx < len(cell_edges) else 0)
                    node_ops.append(nn.Identity())
            self._node_ops.append(node_ops)
            self._node_inputs.append(node_from)
            start_idx += k

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]

        for node_ops, from_nodes in zip(self._node_ops, self._node_inputs):
            s = sum(op(states[f]) for op, f in zip(node_ops, from_nodes))
            states.append(s)

        return torch.cat([states[i] for i in self._concat_indices], dim=1)


class DiscreteNetwork(nn.Module):
    """A full CNN built from a discrete Genotype."""

    def __init__(
        self,
        genotype: Genotype,
        init_channels: int = 16,
        num_classes: int = 10,
        layers: int = 8,
        auxiliary: bool = False,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self._genotype = genotype
        self._layers = layers
        self._drop_path_prob = drop_path_prob
        self._auxiliary = auxiliary

        c_curr = 3 * init_channels  # stem_multiplier
        self.stem = Stem(3, c_curr)

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, init_channels
        self.cells = nn.ModuleList()

        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            cell_edges = genotype.reduce if reduction else genotype.normal
            concat_idx = genotype.reduce_concat if reduction else genotype.normal_concat

            cell = DiscreteCell(
                cell_edges=cell_edges,
                concat_indices=concat_idx,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                c_curr=c_curr,
                reduction=reduction,
                reduction_prev=reduction_prev,
            )
            reduction_prev = reduction
            self.cells.append(cell)
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr

            if auxiliary and i == 2 * layers // 3:
                self.auxiliary_head = AuxiliaryHead(c_prev, num_classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        aux_logits = None

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if (
                self._auxiliary
                and hasattr(self, "auxiliary_head")
                and i == 2 * self._layers // 3
            ):
                aux_logits = self.auxiliary_head(s1)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.training and aux_logits is not None:
            return logits, aux_logits
        return logits


def decode_genotype(genotype: Genotype) -> str:
    """Human-readable summary of a genotype."""
    return genotype.to_compact_string()


def build_network_from_genotype(
    genotype: Genotype,
    init_channels: int = 16,
    num_classes: int = 10,
    layers: int = 8,
    auxiliary: bool = False,
    drop_path_prob: float = 0.0,
) -> DiscreteNetwork:
    """Build a full CNN from a Genotype.
    
    This is the main entry point for constructing a discrete
    architecture after search completes.
    """
    return DiscreteNetwork(
        genotype=genotype,
        init_channels=init_channels,
        num_classes=num_classes,
        layers=layers,
        auxiliary=auxiliary,
        drop_path_prob=drop_path_prob,
    )
