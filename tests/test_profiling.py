import pytest
import torch
import torch.nn as nn

from neuralforge.utils.profiling import (
    count_parameters, count_macs, count_flops,
    estimate_memory, profile_model,
)


class TestProfiling:
    def test_count_parameters(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        counts = count_parameters(model)
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]

    def test_count_macs_small_model(self):
        model = nn.Linear(10, 5)
        macs = count_macs(model, (1, 10))
        assert macs == 50, f"Expected 50 MACs, got {macs}"

    def test_conv_macs(self):
        model = nn.Conv2d(3, 16, 3, padding=1)
        macs = count_macs(model, (1, 3, 32, 32))
        expected = 1 * 32 * 32 * 3 * 3 * 3 * 16
        assert macs == expected, f"Expected {expected}, got {macs}"

    def test_profile_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        results = profile_model(model, (3, 32, 32), verbose=False)
        assert results["params_total"] > 0
        assert results["macs"] > 0
        assert results["flops"] > 0
        assert "params_total_str" in results
        assert "macs_str" in results
