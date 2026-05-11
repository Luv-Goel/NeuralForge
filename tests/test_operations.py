"""
Tests for core operations.
"""

import pytest
import torch
import torch.nn as nn

from neuralforge.core.operations import (
    OPS, PRIMITIVES, get_op, ReLUConvBN, SepConv,
    DilConv, FactorizedReduce, Stem, AuxiliaryHead,
    Zero, count_parameters,
)


class TestOperations:
    def test_all_primitives_defined(self):
        """All primitives should have corresponding ops."""
        for p in PRIMITIVES:
            assert p in OPS, f"Missing op: {p}"

    def test_op_construction(self):
        """All ops should construct correctly."""
        for op_name in PRIMITIVES:
            op = get_op(op_name, 16, 16)
            assert isinstance(op, nn.Module), f"{op_name} not a Module"

    def test_op_forward(self):
        """All ops should forward without error."""
        x = torch.randn(2, 16, 8, 8)
        for op_name in PRIMITIVES:
            op = get_op(op_name, 16, 16)
            try:
                out = op(x)
                assert out.shape[-2:] == x.shape[-2:], f"{op_name} changed spatial dim"
                if op_name != "none":
                    assert out.shape[1] == 16, f"{op_name} changed channels"
            except Exception as e:
                pytest.fail(f"{op_name} failed: {e}")

    def test_factorized_reduce(self):
        """FactorizedReduce should halve spatial dims."""
        fr = FactorizedReduce(16, 32)
        x = torch.randn(2, 16, 16, 16)
        out = fr(x)
        assert out.shape == (2, 32, 8, 8), f"Got {out.shape}"

    def test_stem_conv(self):
        """Stem should process input correctly."""
        stem = Stem(3, 32)
        x = torch.randn(2, 3, 32, 32)
        out = stem(x)
        assert out.shape == (2, 32, 32, 32)

    def test_skip_connect_identity(self):
        """Skip connection should be identity when same shape."""
        op = get_op("skip_connect", 16, 16)
        x = torch.randn(2, 16, 8, 8)
        out = op(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x), "Skip connect should be identity"

    def test_sep_conv_forward(self):
        """SepConv should work with different kernel sizes."""
        for ks in [3, 5]:
            for stride in [1, 2]:
                padding = ks // 2
                conv = SepConv(16, 32, ks, stride, padding)
                h = 16 // stride
                x = torch.randn(2, 16, 16, 16)
                out = conv(x)
                assert out.shape == (2, 32, h, h), f"SepConv {ks}s{stride}: {out.shape}"

    def test_dil_conv_forward(self):
        """Dilated conv should work."""
        dil = DilConv(16, 32, 3, 1, 2, 2)
        x = torch.randn(2, 16, 16, 16)
        out = dil(x)
        assert out.shape == (2, 32, 16, 16)

    def test_auxiliary_head(self):
        """Auxiliary head should produce correct logits."""
        aux = AuxiliaryHead(64, 10)
        x = torch.randn(2, 64, 16, 16)
        out = aux(x)
        assert out.shape == (2, 10), f"Got {out.shape}"

    def test_zero_op(self):
        """Zero op should zero out."""
        z = Zero(stride=2)
        x = torch.randn(2, 16, 16, 16)
        out = z(x)
        assert out.shape == (2, 16, 8, 8)
        assert out.abs().sum() == 0.0, "Zero op should produce zeros"

    def test_count_parameters(self):
        """Parameter counting should work."""
        op = get_op("sep_conv_3x3", 16, 32)
        assert count_parameters(op) > 0
