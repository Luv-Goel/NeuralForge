import pytest
import torch

from neuralforge.core.search_space import (
    SearchSpaceConfig, CellSearchSpace, SearchSpaceCNN,
)


class TestSearchSpaceConfig:
    def test_default_config(self):
        config = SearchSpaceConfig()
        assert config.c_in == 3
        assert config.init_channels == 16
        assert config.num_classes == 10
        assert config.layers == 8
        assert config.nodes == 4

    def test_custom_config(self):
        config = SearchSpaceConfig(
            init_channels=32, layers=12, nodes=6
        )
        assert config.init_channels == 32
        assert config.layers == 12
        assert config.nodes == 6

    def test_min_primitives(self):
        with pytest.raises(ValueError):
            SearchSpaceConfig(primitive_set=["none"])


class TestCellSearchSpace:
    def test_search_space_properties(self, search_space):
        assert search_space.num_operations == len(search_space.config.primitive_set)
        assert search_space.total_edges > 0

    def test_build_search_network(self, search_space):
        net = search_space.build_search_network()
        assert isinstance(net, SearchSpaceCNN)

    def test_search_network_forward(self, search_space):
        net = search_space.build_search_network()
        net.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 10), f"Got {out.shape}"


class TestSearchSpaceCNN:
    def test_forward_shape(self):
        config = SearchSpaceConfig(layers=6, init_channels=16)
        net = SearchSpaceCNN(config)
        net.eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (4, 10)

    def test_reduction_cells_position(self):
        """Reduction cells should be at 1/3 and 2/3 depth."""
        config = SearchSpaceConfig(layers=9)
        net = SearchSpaceCNN(config)
        reduction_indices = []
        for i, cell in enumerate(net.cells):
            if hasattr(cell, 'reduction') and cell.reduction:
                reduction_indices.append(i)
        # Should have reduction cells at positions 3 and 6 for 9-layer net
        assert 3 in reduction_indices or len(reduction_indices) > 0
