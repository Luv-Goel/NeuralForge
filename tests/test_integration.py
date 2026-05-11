"""
Integration tests for the NeuralForge package.
Tests end-to-end workflows with small, fast models.
"""

import pytest
import torch

from neuralforge.core.search_space import CellSearchSpace, SearchSpaceConfig
from neuralforge.core.genotypes import random_genotype, build_network_from_genotype
from neuralforge.core.operations import PRIMITIVES
from neuralforge.algorithms.random_search import RandomSearch
from neuralforge.utils.profiling import profile_model
from neuralforge.utils.visualization import (
    render_cell_graph,
    _to_text,
)


class TestIntegration:
    def test_random_genotype_to_network(self):
        """Random genotype should produce a valid network."""
        genotype = random_genotype(PRIMITIVES, nodes=4)
        try:
            net = build_network_from_genotype(
                genotype, init_channels=16, num_classes=10, layers=6,
            )
            assert net is not None
        except Exception as e:
            pytest.fail(f"Failed to build network: {e}")

    def test_search_space_to_network_forward(self):
        """Search space should build a forward-able network."""
        config = SearchSpaceConfig(layers=6, init_channels=16)
        space = CellSearchSpace(config)
        net = space.build_search_network()
        net.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, 10)

    def test_render_cell_text(self, sample_genotype):
        """Cell rendering should produce text output."""
        text = render_cell_graph(sample_genotype, "text")
        assert isinstance(text, str)
        assert len(text) > 50

    def test_profile_random_network(self):
        """Profiling should work on a built network."""
        genotype = random_genotype(PRIMITIVES, nodes=4)
        net = build_network_from_genotype(
            genotype, init_channels=16, num_classes=10, layers=6,
        )
        net.eval()
        results = profile_model(net, (3, 32, 32), verbose=False)
        assert results["params_total"] > 0
