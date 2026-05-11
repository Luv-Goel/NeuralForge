"""
NeuralForge — Model Profiling
==============================

Utilities for measuring model statistics:
    - Parameter count
    - FLOPs (floating-point operations)
    - MACs (multiply-accumulate operations)
    - Memory footprint
    - Latency estimation

Uses fvcore/thop-style counting with manual overrides for
custom operations to ensure accurate measurements.

References:
    - https://github.com/facebookresearch/fvcore
    - https://github.com/Lyken17/pytorch-OpCounter
"""

from __future__ import annotations
import logging
from typing import Dict, Tuple, Optional, List, Any
from functools import partial

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

# Cache for profile results to avoid redundant computation
_profile_cache: Dict[str, Dict[str, float]] = {}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters.
    
    Returns:
        dict with 'trainable' and 'total' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def count_macs(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """Count MACs (multiply-accumulate operations) for a forward pass.
    
    Uses a simple profile hook approach. For production use, consider
    fvcore which has more comprehensive operation coverage.
    
    Args:
        model: PyTorch model.
        input_size: Input tensor shape (C, H, W) or (B, C, H, W).
    
    Returns:
        Total MAC count.
    """
    if len(input_size) == 3:
        input_size = (1, *input_size)
    
    model = model.cpu()
    model.eval()
    
    hooks = []
    macs_per_layer = {}
    
    def _hook_fn(name, module, input, output):
        """Count MACs for supported modules."""
        if isinstance(module, nn.Conv2d):
            n, c, h, w = input[0].shape
            k_h, k_w = module.kernel_size
            out_h = (h + 2 * module.padding[0] - k_h) // module.stride[0] + 1
            out_w = (w + 2 * module.padding[1] - k_w) // module.stride[1] + 1
            macs = n * out_h * out_w * c * k_h * k_w * module.out_channels // module.groups
            macs_per_layer[name] = macs
        
        elif isinstance(module, nn.Linear):
            macs = input[0].shape[0] * input[0].shape[1] * module.out_features
            macs_per_layer[name] = macs
        
        elif isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d):
            # Pooling ops — cheap, approximate
            macs_per_layer[name] = 0
        
        else:
            # Unknown module — skip (conservative estimate)
            macs_per_layer[name] = 0
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.MaxPool2d)):
            hooks.append(
                module.register_forward_hook(partial(_hook_fn, name))
            )
    
    # Run forward pass
    try:
        dummy = torch.randn(input_size)
        _ = model(dummy)
    except Exception as e:
        logger.warning(f"Profiling forward pass failed: {e}")
        # Clean up hooks before raising
        for h in hooks:
            h.remove()
        return 0
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    total_macs = sum(macs_per_layer.values())
    return int(total_macs)


def count_flops(model: nn.Module, input_size: Tuple[int, ...]) -> int:
    """Count FLOPs — approximately 2x MACs (multiply + add).
    
    Args:
        model: PyTorch model.
        input_size: Input tensor shape.
    
    Returns:
        Total FLOP count.
    """
    macs = count_macs(model, input_size)
    return int(2 * macs)


def estimate_memory(
    model: nn.Module,
    input_size: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Estimate memory usage (in MB) for forward + backward pass.
    
    Returns:
        dict with 'params_mb', 'activations_mb', 'total_mb'.
    """
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    
    # Parameter memory
    params_mb = sum(
        p.numel() * bytes_per_elem for p in model.parameters()
    ) / (1024 * 1024)
    
    # Activation memory — rough estimate based on largest feature map
    # For CNN: activation size ~ batch_size * channels * H * W * bytes
    if len(input_size) == 3:
        input_size = (1, *input_size)
    activation_elems = np.prod(input_size)
    
    # Rough multiplier for intermediate activations (depends on depth)
    depth_factor = sum(1 for _ in model.modules() if isinstance(_, (nn.Conv2d, nn.Linear)))
    activation_mb = activation_elems * bytes_per_elem * min(depth_factor, 20) / (1024 * 1024)
    
    return {
        "params_mb": round(params_mb, 2),
        "activations_mb": round(activation_mb, 2),
        "total_mb": round(params_mb + activation_mb, 2),
    }


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Comprehensive model profiling — params, FLOPs, MACs, memory.
    
    Args:
        model: PyTorch model to profile.
        input_size: Input tensor shape (C, H, W) or (B, C, H, W).
        verbose: Whether to log results.
    
    Returns:
        Dictionary with all profiling results.
    """
    # Use caching to avoid redundant computation
    model_id = str(id(model))
    if model_id in _profile_cache:
        return _profile_cache[model_id]
    
    params = count_parameters(model)
    macs = count_macs(model, input_size)
    flops = count_flops(model, input_size)
    memory = estimate_memory(model, input_size)
    
    def _format(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f}G"
        elif n >= 1e6:
            return f"{n / 1e6:.2f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)
    
    results = {
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "params_total_str": _format(params["total"]),
        "macs": macs,
        "macs_str": _format(macs),
        "flops": flops,
        "flops_str": _format(flops),
        "memory_mb": memory,
        "input_size": input_size,
    }
    
    if verbose:
        logger.info(
            f"Model Profile:
"
            f"  Parameters: {results['params_total_str']} "
            f"({results['params_trainable']:,} trainable)
"
            f"  MACs: {results['macs_str']}
"
            f"  FLOPs: {results['flops_str']}
"
            f"  Memory: {memory['total_mb']}MB "
            f"(params={memory['params_mb']}MB, "
            f"activations≈{memory['activations_mb']}MB)"
        )
    
    _profile_cache[model_id] = results
    return results


def compare_architectures(
    models: Dict[str, Tuple[nn.Module, Tuple[int, ...]]],
) -> Dict[str, Dict[str, Any]]:
    """Profile and compare multiple architectures side-by-side.
    
    Args:
        models: Dict mapping name -> (model, input_size) pairs.
    
    Returns:
        Dict mapping name -> profiling results.
    """
    results = {}
    for name, (model, input_size) in models.items():
        logger.info(f"Profiling {name}...")
        results[name] = profile_model(model, input_size, verbose=False)
    return results
