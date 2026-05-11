"""
NeuralForge — Visualization Utilities
======================================

Tools for visualizing architectures, search trajectories,
operation importance, and more.

Uses matplotlib and networkx for rendering. All plots
can be saved to file or displayed inline.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from neuralforge.core.genotypes import Genotype

logger = logging.getLogger(__name__)

# Color map for operations
OP_COLORS = {
    "none": "#cccccc",
    "skip_connect": "#4ecdc4",
    "sep_conv_3x3": "#45b7d1",
    "sep_conv_5x5": "#96ceb4",
    "dil_conv_3x3": "#ffeaa7",
    "dil_conv_5x5": "#dfe6e9",
    "max_pool_3x3": "#ff7675",
    "avg_pool_3x3": "#74b9ff",
    "conv_1x1": "#a29bfe",
    "conv_3x3": "#fd79a8",
}

OP_SYMBOLS = {
    "none": "✗",
    "skip_connect": "→",
    "sep_conv_3x3": "3✧",
    "sep_conv_5x5": "5✧",
    "dil_conv_3x3": "3◈",
    "dil_conv_5x5": "5◈",
    "max_pool_3x3": "▽",
    "avg_pool_3x3": "△",
    "conv_1x1": "•",
    "conv_3x3": "◆",
}


def plot_architecture(
    genotype: Genotype,
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150,
) -> str:
    """Plot the architecture as a directed acyclic graph.

    Creates side-by-side visualization of normal and reduction cells.

    Args:
        genotype: The genotype to visualize.
        filename: Save path (optional). Auto-generated if None.
        figsize: Figure dimensions.
        dpi: Figure resolution.

    Returns:
        Path to the saved image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, edges, cell_type in [
        (ax1, genotype.normal, "Normal Cell"),
        (ax2, genotype.reduce, "Reduction Cell"),
    ]:
        _draw_cell(ax, edges, cell_type)

    plt.tight_layout()

    if filename is None:
        timestamp = int(__import__("time").time())
        filename = f"architecture_{timestamp}.png"

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"Architecture plot saved: {filename}")
    return filename


def _draw_cell(ax, edges, title):
    """Draw a single cell's DAG."""
    # Layout: input nodes at bottom, intermediate nodes above
    # We use a simple layered layout

    # Extract unique node indices referenced in edges
    from_nodes = set()
    to_nodes = set()
    for op_name, from_idx in edges:
        from_nodes.add(from_idx)

    # Node positions
    max_node = max(from_nodes) if from_nodes else 2

    # Position nodes in layers
    # s0 (0) and s1 (1) at bottom, intermediates above
    pos = {}
    pos[0] = (-0.3, 0.0)  # c_{k-2}
    pos[1] = (0.3, 0.0)  # c_{k-1}

    # Count edges per target to determine which edges exist
    from collections import defaultdict

    # DARTS format: edges are ordered, each intermediate node has 2 edges
    target_count = defaultdict(int)
    for op_name, from_idx in edges:
        target_count[from_idx] += 1

    # Determine target nodes
    # In DARTS genotype, edges are in order for node 2, then node 3, etc.
    # We don't have explicit target info, so we infer from structure
    intermediates = sorted(list(set(idx for _, idx in edges)))

    # Map from indices to unique intermediate node numbers
    node_set = sorted(set([0, 1] + [from_idx for _, from_idx in edges]))
    intermediate_nodes = [n for n in node_set if n >= 2]

    for i, n in enumerate(intermediate_nodes):
        pos[n] = (0.0, 0.4 * (i + 1))

    # Draw nodes
    for node, (x, y) in pos.items():
        if node < 2:
            # Input nodes
            ax.scatter(x, y, s=600, c="#2d3436", zorder=5)
            ax.text(
                x,
                y,
                f"c_{{k-{2 - node}}}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        else:
            # Intermediate nodes (sum)
            ax.scatter(x, y, s=400, c="#0984e3", zorder=5)
            ax.text(
                x,
                y,
                f"n={node}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    # Draw edges with operation labels
    # We need to figure out which edge goes where
    # In DARTS, edges target nodes in order: node 2 gets first 2 edges,
    # node 3 gets next 2 edges, etc.
    target_node_idx = 2
    edges_written = 0

    for op_name, from_idx in edges:
        # Determine target node: every 2 edges moves to next node
        target = 2 + edges_written // 2
        if target not in intermediate_nodes:
            target = intermediate_nodes[-1] if intermediate_nodes else 2

        from_pos = pos.get(from_idx, (0, 0))
        to_pos = pos.get(target, (0, 0.5))

        # Draw edge
        color = OP_COLORS.get(op_name, "#b2bec3")
        ax.annotate(
            "",
            xy=to_pos,
            xytext=from_pos,
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5 + 0.5 * (op_name not in ["none"]),
                connectionstyle="arc3,rad=0.1",
            ),
        )

        # Label at midpoint
        mx = (from_pos[0] + to_pos[0]) / 2
        my = (from_pos[1] + to_pos[1]) / 2
        offset_x = 0.05 if from_pos[0] != to_pos[0] else 0.08
        ax.text(
            mx + offset_x,
            my,
            OP_SYMBOLS.get(op_name, op_name),
            fontsize=7,
            color=color,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )

        edges_written += 1

    # Legend for operations used
    ops_used = set(op for op, _ in edges)
    legend_elements = [
        mpatches.Patch(
            facecolor=OP_COLORS.get(op, "#b2bec3"),
            edgecolor="black",
            label=op,
        )
        for op in sorted(ops_used)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=6,
        framealpha=0.8,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_search_trajectory(
    history: List[Dict[str, Any]],
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> str:
    """Plot the search trajectory — fitness over time.

    Args:
        history: List of metric dicts from SearchAlgorithm.history.
        filename: Save path.
        figsize: Figure dimensions.

    Returns:
        Path to saved image.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if not history:
        logger.warning("Empty history — nothing to plot.")
        return ""

    # Extract data
    steps = [h.get("step", h.get("epoch", i)) for i, h in enumerate(history)]
    fitness = [
        h.get("accuracy", h.get("fitness", h.get("val_accuracy", 0.0))) for h in history
    ]

    # Fitness over time
    ax1.plot(steps, fitness, color="#0984e3", linewidth=1.5, alpha=0.7)
    ax1.scatter(steps, fitness, color="#0984e3", s=10, alpha=0.5)

    # Running best
    best_so_far = np.maximum.accumulate(fitness)
    ax1.plot(
        steps,
        best_so_far,
        color="#e17055",
        linewidth=2,
        linestyle="--",
        label="Best so far",
    )

    ax1.set_xlabel("Step / Epoch")
    ax1.set_ylabel("Fitness (accuracy)")
    ax1.set_title("Search Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution of final population (if available)
    if isinstance(fitness, list) and len(fitness) > 1:
        ax2.hist(fitness, bins=20, color="#00b894", alpha=0.7, edgecolor="black")
        ax2.axvline(
            np.max(fitness),
            color="#e17055",
            linestyle="--",
            linewidth=2,
            label=f"Best: {np.max(fitness):.4f}",
        )
        ax2.set_xlabel("Fitness")
        ax2.set_ylabel("Count")
        ax2.set_title("Fitness Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if filename is None:
        filename = f"search_trajectory_{int(__import__('time').time())}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Trajectory plot saved: {filename}")
    return filename


def plot_operation_importance(
    genotype: Genotype,
    filename: Optional[str] = None,
) -> str:
    """Plot histogram of operation frequencies in a genotype.

    Args:
        genotype: The genotype to analyze.
        filename: Save path.

    Returns:
        Path to saved image.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Count operations
    all_ops = [op for op, _ in genotype.normal + genotype.reduce]
    op_counts = {}
    for op in all_ops:
        op_counts[op] = op_counts.get(op, 0) + 1

    ops = list(op_counts.keys())
    counts = list(op_counts.values())
    colors = [OP_COLORS.get(op, "#b2bec3") for op in ops]

    bars = ax.barh(range(len(ops)), counts, color=colors, edgecolor="black")
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops)
    ax.set_xlabel("Count")
    ax.set_title("Operation Usage in Architecture")

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    if filename is None:
        filename = f"op_importance_{int(__import__('time').time())}.png"

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    return filename


def render_cell_graph(
    genotype: Genotype,
    output_format: str = "text",
) -> str:
    """Render a cell as ASCII/text graph.

    Args:
        genotype: The genotype.
        output_format: "text" or "dot" (Graphviz).

    Returns:
        String representation.
    """
    if output_format == "dot":
        return _to_dot(genotype)
    return _to_text(genotype)


def _to_text(genotype: Genotype) -> str:
    """ASCII representation of the cell."""
    lines = ["Cell Architecture (DARTS format):", "─" * 50]

    for cell_type, edges in [("Normal", genotype.normal), ("Reduce", genotype.reduce)]:
        lines.append(f"\n{cell_type} Cell:")
        for i, (op, from_idx) in enumerate(edges):
            target = 2 + i // 2  # DARTS convention
            lines.append(f"  [{target}] <- {op:18s} <- [{from_idx}]")

    if genotype.accuracy:
        lines.append(f"\nAccuracy: {genotype.accuracy:.4f}")
    if genotype.params:
        lines.append(f"\nParams: {genotype.params:.1f}M")
    if genotype.flops:
        lines.append(f"\nFLOPs: {genotype.flops:.1f}M")

    return "\n".join(lines)


def _to_dot(genotype: Genotype) -> str:
    """Graphviz DOT format string."""
    lines = ["digraph Cell {"]
    lines.append("  rankdir=BT;")
    lines.append("  node [shape=box, style=filled, fillcolor=lightblue];")

    for node in [0, 1]:
        lines.append(
            f'  n{node} [label="c_k-{2-node}", fillcolor="#2d3436", fontcolor=white];'
        )

    for i, (op, from_idx) in enumerate(genotype.normal):
        target = 2 + i // 2
        color = OP_COLORS.get(op, "#b2bec3")
        lines.append(
            f"  n{from_idx} -> n{target} "
            f'[label="{op}", color="{color}", fontsize=10];'
        )

    lines.append("}")
    return "\n".join(lines)
