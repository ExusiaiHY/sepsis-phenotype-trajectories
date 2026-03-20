"""
visualization.py - S2-light temporal phenotype trajectory figures.

All figures use descriptive language only. No causal or treatment-response framing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger("s2light.viz")

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


def plot_per_window_prevalence(
    window_labels: np.ndarray,
    window_starts: list[int],
    window_len: int,
    save_path: Path,
    k: int = 4,
) -> None:
    """Stacked bar: cluster proportions at each window position."""
    N, W = window_labels.shape
    fig, ax = plt.subplots(figsize=(10, 5))

    fracs = np.zeros((W, k))
    for wi in range(W):
        for c in range(k):
            fracs[wi, c] = (window_labels[:, wi] == c).mean()

    x = np.arange(W)
    bottom = np.zeros(W)
    for c in range(k):
        ax.bar(x, fracs[:, c], bottom=bottom, color=COLORS[c % len(COLORS)],
               label=f"Phenotype {c}", edgecolor="white", linewidth=0.5)
        bottom += fracs[:, c]

    labels = [f"[{s},{s + window_len})" for s in window_starts]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Window (hours)")
    ax.set_ylabel("Proportion")
    ax.set_title("Phenotype Prevalence by Window Position")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.2, axis="y")

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_sankey_transitions(
    window_labels: np.ndarray,
    save_path: Path,
    k: int = 4,
    min_flow_frac: float = 0.01,
) -> None:
    """
    Simplified Sankey-style flow diagram using matplotlib.
    Shows flows between adjacent windows as curved bands.
    """
    N, W = window_labels.shape
    fig, ax = plt.subplots(figsize=(14, 7))

    # Compute flows between each pair of adjacent windows
    node_x_positions = np.linspace(0.05, 0.95, W)
    node_height = 0.85
    node_width = 0.04

    # For each window, compute cluster positions (stacked vertically)
    node_positions = {}  # (window_idx, cluster) -> (y_center, height)
    for wi in range(W):
        counts = np.array([(window_labels[:, wi] == c).sum() for c in range(k)])
        fracs = counts / N
        y_bottom = 0.05
        for c in range(k):
            h = fracs[c] * node_height
            node_positions[(wi, c)] = (y_bottom + h / 2, h)
            y_bottom += h + 0.01

    # Draw nodes
    for wi in range(W):
        x = node_x_positions[wi]
        for c in range(k):
            y_center, h = node_positions[(wi, c)]
            rect = plt.Rectangle(
                (x - node_width / 2, y_center - h / 2), node_width, h,
                color=COLORS[c % len(COLORS)], ec="white", lw=0.5, zorder=3,
            )
            ax.add_patch(rect)

    # Draw flows
    for wi in range(W - 1):
        x0 = node_x_positions[wi] + node_width / 2
        x1 = node_x_positions[wi + 1] - node_width / 2

        for src in range(k):
            src_y, src_h = node_positions[(wi, src)]
            src_count = (window_labels[:, wi] == src).sum()
            if src_count == 0:
                continue

            flow_offset_src = -src_h / 2
            for dst in range(k):
                flow_count = ((window_labels[:, wi] == src) & (window_labels[:, wi + 1] == dst)).sum()
                flow_frac = flow_count / N
                if flow_frac < min_flow_frac:
                    continue

                flow_h = (flow_count / src_count) * src_h
                y0 = src_y + flow_offset_src + flow_h / 2
                flow_offset_src += flow_h

                dst_y, dst_h = node_positions[(wi + 1, dst)]

                # Color by source with alpha
                color = COLORS[src % len(COLORS)]
                alpha = 0.3 if src == dst else 0.5

                # Simple straight band
                from matplotlib.patches import FancyArrowPatch
                ax.fill([x0, x0, x1, x1],
                        [y0 - flow_h / 2, y0 + flow_h / 2,
                         y0 + flow_h / 2, y0 - flow_h / 2],
                        color=color, alpha=alpha, ec="none")

    # Labels
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xticks(node_x_positions)
    ax.set_xticklabels([f"W{i}" for i in range(W)], fontsize=11)
    ax.set_yticks([])
    ax.set_title("Phenotype Transitions Across Rolling Windows\n"
                 "(Descriptive temporal phenotype trajectories)", fontsize=13)

    handles = [mpatches.Patch(color=COLORS[c], label=f"Phenotype {c}") for c in range(k)]
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    ax.set_frame_on(False)

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_mortality_by_trajectory(
    trajectory_stats: dict,
    save_path: Path,
) -> None:
    """Bar chart: mortality rate by trajectory category (all-cohort descriptive)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cats = trajectory_stats["mortality_descriptives"]["all_cohort"]
    names = []
    rates = []
    counts = []

    for key in ["stable", "single_transition", "multi_transition"]:
        entry = cats[key]
        label = key.replace("_", " ").title()
        names.append(f"{label}\n(n={entry['n']})")
        rates.append(entry["mortality_rate"] if entry["mortality_rate"] is not None else 0)
        counts.append(entry["n"])

    bars = ax.bar(range(len(names)), rates, color=[COLORS[0], COLORS[1], COLORS[3]],
                  edgecolor="white", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("In-Hospital Mortality Rate")
    ax.set_title("Mortality by Trajectory Category\n"
                 "(Descriptive, all-cohort summary)")
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, max(rates) * 1.3 if rates else 0.3)

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
