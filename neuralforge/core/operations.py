"""
NeuralForge — Core Operations
=============================

Primitive operations for neural architecture search cells.
Each operation is a nn.Module with a consistent interface.

Heavily inspired by the DARTS paper (Liu et al., 2019) and
the pycls implementation from Facebook Research, but with
our own tweaks and additions.

References:
    - DARTS: Differentiable Architecture Search (ICLR 2019)
    - NAS-Bench-101 (Google Brain, 2019)
    - AmoebaNet: Regularized Evolution (ICML 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# ——— The primitive operations available during search ———
# Each is (name, kernel_size) — the actual nn.Module is built by OPS dict
# Factorized reduction uses stride=2 variants automatically

PRIMITIVES = [
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]

# DARTS-CNN search space; for CIFAR-10, this is the standard set.
# We also support the "NAS-Bench-201" space via separate config.

NAS_BENCH_201_PRIMITIVES = [
    "none",
    "skip_connect",
    "conv_1x1",
    "conv_3x3",
    "avg_pool_3x3",
]


class ReLUConvBN(nn.Module):
    """(ReLU -> Conv -> BN) block, workhorse of NAS cells."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        affine: bool = True,
    ):
        super().__init__()
        padding = padding or (kernel_size // 2)
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c_in, c_out, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SepConv(nn.Module):
    """Depthwise-separable convolution — factorizes a standard conv
    into a depthwise conv + pointwise conv. Fewer params, similar
    representational power. Used heavily in DARTS/PNASNet."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        affine: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c_in, c_in, kernel_size, stride, padding,
                groups=c_in, bias=False,
            ),
            nn.Conv2d(c_in, c_in, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_in, affine=affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c_in, c_in, kernel_size, 1, padding,
                groups=c_in, bias=False,
            ),
            nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DilConv(nn.Module):
    """Dilated convolution — captures larger receptive field
    without increasing kernel size. Good for dense prediction tasks
    but also useful in NAS cells for multi-scale feature extraction."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 2,
        affine: bool = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c_in, c_in, kernel_size, stride, padding,
                dilation=dilation, groups=c_in, bias=False,
            ),
            nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedReduce(nn.Module):
    """Reduces spatial resolution by 2x while increasing channels.
    Uses two parallel conv paths (left half, right half of channels)
    to avoid information loss from strided convs on low-res feature maps."""

    def __init__(self, c_in: int, c_out: int, affine: bool = True):
        super().__init__()
        assert c_out % 2 == 0, f"c_out {c_out} must be even"
        self.conv_1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1))
        return x


class Stem(nn.Module):
    """Initial stem convolution that processes raw input images
    before the first cell. Standard 3x3 conv -> BN."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxiliaryHead(nn.Module):
    """Auxiliary classification head for intermediate supervision
    (used in DARTS to add a loss at the 2/3 point of the network).
    Helps gradient flow in deep architectures during search."""

    def __init__(self, c_in: int, num_classes: int, stride: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0),
            nn.Conv2d(c_in, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class Zero(nn.Module):
    """The 'none' operation — zeros out the input.
    Used in discrete architecture search to represent missing edges."""

    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        return x.mul(0.0)


# ——— Operation registry ———
# Maps primitive names to factory functions

OPS = {
    "none": lambda c_in, c_out, stride, affine: (
        Zero(stride) if c_in == c_out else nn.Sequential()
    ),
    "avg_pool_3x3": lambda c_in, c_out, stride, affine: nn.AvgPool2d(
        3, stride=stride, padding=1, count_include_pad=False,
    ),
    "max_pool_3x3": lambda c_in, c_out, stride, affine: nn.MaxPool2d(
        3, stride=stride, padding=1,
    ),
    "skip_connect": lambda c_in, c_out, stride, affine: (
        nn.Identity()
        if stride == 1 and c_in == c_out
        else FactorizedReduce(c_in, c_out, affine=affine)
        if stride == 2
        else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, 0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine),
        )
    ),
    "sep_conv_3x3": lambda c_in, c_out, stride, affine: SepConv(
        c_in, c_out, 3, stride, 1, affine=affine,
    ),
    "sep_conv_5x5": lambda c_in, c_out, stride, affine: SepConv(
        c_in, c_out, 5, stride, 2, affine=affine,
    ),
    "dil_conv_3x3": lambda c_in, c_out, stride, affine: DilConv(
        c_in, c_out, 3, stride, 2, 2, affine=affine,
    ),
    "dil_conv_5x5": lambda c_in, c_out, stride, affine: DilConv(
        c_in, c_out, 5, stride, 4, 2, affine=affine,
    ),
    "conv_1x1": lambda c_in, c_out, stride, affine: ReLUConvBN(
        c_in, c_out, 1, stride, 0, affine=affine,
    ),
    "conv_3x3": lambda c_in, c_out, stride, affine: ReLUConvBN(
        c_in, c_out, 3, stride, 1, affine=affine,
    ),
}


def get_op(
    op_name: str,
    c_in: int,
    c_out: int,
    stride: int = 1,
    affine: bool = True,
) -> nn.Module:
    """Construct an operation module by name.
    
    Args:
        op_name: One of the keys in OPS, or any PRIMITIVE name.
        c_in: Input channels.
        c_out: Output channels.
        stride: Convolution stride.
        affine: Whether BatchNorm affine params are trainable.
    
    Returns:
        An nn.Module representing the operation.
    """
    if op_name not in OPS:
        raise KeyError(
            f"Unknown operation '{op_name}'. Available: {list(OPS.keys())}"
        )
    return OPS[op_name](c_in, c_out, stride, affine)


def count_parameters(op: nn.Module) -> int:
    """Count trainable parameters in an operation module.
    Useful for architecture profiling."""
    return sum(p.numel() for p in op.parameters() if p.requires_grad)
