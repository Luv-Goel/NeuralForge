"""
Tests for the DARTS search algorithm.
Uses mocked forward passes to verify structure without GPU.
"""

import pytest
import torch
import torch.nn as nn

from neuralforge.core.search_space import CellSearchSpace
from neuralforge.core.genotypes import Genotype
from neuralforge.algorithms.darts import (
    DARTSSearch,
    DARTSNetwork,
    DARTSCell,
    MixedOp,
)
from neuralforge.core.operations import PRIMITIVES


class TestMixedOp:
    def test_mixed_op_creation(self):
        """MixedOp should create ops for each primitive."""
        op = MixedOp(16, 16, 1, PRIMITIVES[:4])
        assert len(op._ops) == 4

    def test_mixed_op_forward(self):
        """MixedOp forward should produce weighted sum."""
        op = MixedOp(16, 16, 1, PRIMITIVES[:4])
        x = torch.randn(1, 16, 8, 8)
        weights = torch.softmax(torch.randn(4), dim=-1)
        out = op(x, weights)
        assert out.shape == (1, 16, 8, 8)


class TestDARTSCell:
    def test_darts_cell_creation(self):
        """DARTSCell should build correctly."""
        cell = DARTSCell(
            nodes=4,
            c_prev_prev=16,
            c_prev=16,
            c_curr=16,
            reduction=False,
            reduction_prev=False,
            primitive_set=PRIMITIVES,
        )
        # 4 nodes, each with 2+i edges = 2+3+4+5 = 14 edges
        assert len(cell._ops) == 14

    def test_darts_cell_reduction(self):
        """Reduction cell should set stride=2 on first two inputs."""
        cell = DARTSCell(
            nodes=4,
            c_prev_prev=16,
            c_prev=32,
            c_curr=32,
            reduction=True,
            reduction_prev=True,
            primitive_set=PRIMITIVES,
        )
        # reduction_prev=True means FactorizedReduce for preprocess0
        assert cell.preprocess0 is not None

    def test_darts_cell_forward(self):
        """DARTSCell forward should produce correct output shape."""
        cell = DARTSCell(
            nodes=4,
            c_prev_prev=16,
            c_prev=16,
            c_curr=16,
            reduction=False,
            reduction_prev=False,
            primitive_set=PRIMITIVES,
        )
        s0 = torch.randn(1, 16, 16, 16)
        s1 = torch.randn(1, 16, 16, 16)
        n_edges = 14
        weights = torch.softmax(torch.randn(n_edges, len(PRIMITIVES)), dim=-1)
        out = cell(s0, s1, weights)
        # Output channels = nodes * c_curr = 4 * 16 = 64
        assert out.shape == (1, 64, 16, 16)


class TestDARTSNetwork:
    def test_network_creation(self):
        """DARTSNetwork should initialize with correct structure."""
        net = DARTSNetwork(
            c_in=3,
            init_channels=16,
            num_classes=10,
            layers=8,
            nodes=4,
            primitive_set=PRIMITIVES,
        )
        assert len(net.cells) == 8
        assert net.alphas_normal.shape[-1] == len(PRIMITIVES)
        assert net.alphas_reduce.shape[-1] == len(PRIMITIVES)

    def test_network_arch_parameters(self):
        """arch_parameters should return both alphas."""
        net = DARTSNetwork(
            c_in=3, init_channels=16, num_classes=10, layers=4, nodes=4,
        )
        params = net.arch_parameters()
        assert len(params) == 2
        assert params[0] is net.alphas_normal
        assert params[1] is net.alphas_reduce

    def test_network_genotype_shape(self):
        """genotype() should return a valid Genotype with correct edge counts."""
        net = DARTSNetwork(
            c_in=3, init_channels=16, num_classes=10, layers=4, nodes=4,
        )
        geno = net.genotype()
        assert isinstance(geno, Genotype)
        # 4 nodes -> 8 edges total (2 per node)
        assert len(geno.normal) == 8
        assert len(geno.reduce) == 8

    def test_network_forward_cpu(self):
        """Forward pass should work on CPU without cuda errors."""
        net = DARTSNetwork(
            c_in=3, init_channels=16, num_classes=10, layers=4, nodes=4,
        )
        net.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, 10)


class TestDARTSSearch:
    def test_search_algorithm_creation(self):
        """DARTSSearch should initialize without error."""
        space = CellSearchSpace()
        algo = DARTSSearch(space)
        assert algo is not None
        assert algo._unrolled is False

    def test_search_default_config(self):
        """Search should apply DARTS-specific defaults."""
        space = CellSearchSpace()
        algo = DARTSSearch(space)
        assert algo.config.epochs == 50
        assert algo.config.arch_learning_rate == 3e-4
