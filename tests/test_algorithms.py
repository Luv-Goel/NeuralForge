import pytest
import torch
from neuralforge.algorithms.darts import DARTSNetwork, MixedOp
from neuralforge.core.search_space import CellSearchSpace

def test_darts_network_forward():
    model = DARTSNetwork(c_in=3, init_channels=16, num_classes=10, layers=2, nodes=2)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)

def test_mixed_op():
    op = MixedOp(16, 16, 1, ["skip_connect", "none"])
    x = torch.randn(2, 16, 8, 8)
    weights = torch.tensor([0.5, 0.5])
    out = op(x, weights)
    assert out.shape == (2, 16, 8, 8)
