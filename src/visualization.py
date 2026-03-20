"""
visualization.py - Visualization module

Responsibilities:
1. Cluster scatter plots (UMAP/t-SNE 2D view)
2. Optimal K selection curves
3. Subtype clinical feature heatmaps
4. Kaplan-Meier survival curves
5. Missing pattern visualization
6. Variable trajectory comparison plots
7. Summary dashboard

All figures use unified styling and auto-save to outputs/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts and servers
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from utils import setup_logger, resolve_path, ensure_dir

logger = setup_logger(__name__)


def _setup_style(config: dict) -> None:
    """Set global plotting style."""
    viz_cfg = config["visualization"]
    try:
        plt.style.use(viz_cfg.get("style", "seaborn-v0_8-whitegrid"))
    except OSError:
        plt.style.use("default")
    import platform
    if platform.system() == "Darwin":
        zh_font = "PingFang SC"
    elif platform.system() == "Windows":
        zh_font = "Microsoft YaHei"
    else:
        zh_font = "WenQuanYi Micro Hei"
    plt.rcParams.update({
        "figure.figsize": viz_cfg.get("figsize", [10, 8]),
        "figure.dpi": viz_cfg.get("dpi", 150),
        "font.sans-serif": [zh_font, "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def _save_fig(fig, name: str, config: dict) -> str:
    """Save figure to output directory."""
    out_dir = ensure_dir(resolve_path(config["paths"]["output_figures"]))
    fmt = config["visualization"].get("save_format", "png")
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=config["visualization"]["dpi"])
    plt.close(fig)
    logger.info(f"Figure saved: {path}")
    return str(path)


# ============================================================
# 1. Cluster Scatter Plot
# ============================================================

def plot_cluster_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    config: dict,
    title: str = "Sepsis Subtype Cluster Scatter Plot",
    true_labels: np.ndarray | None = None,
) -> str:
    """Plot clustering results in 2D reduced space. Dual subplot if true labels provided."""
    _setup_style(config)
    palette = config["visualization"].get("palette", "Set2")
    method = config["reduction"]["method"].upper()

    if true_labels is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        scatter1 = ax1.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=labels, cmap=palette, alpha=0.6, s=20, edgecolors="none"
        )
        ax1.set_title("Discovered Subtypes", fontsize=14)
        ax1.set_xlabel(f"{method}-1")
        ax1.set_ylabel(f"{method}-2")
        plt.colorbar(scatter1, ax=ax1, label="Cluster Label")

        scatter2 = ax2.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=true_labels, cmap=palette, alpha=0.6, s=20, edgecolors="none"
        )
        ax2.set_title("True Subtypes (Simulated)", fontsize=14)
        ax2.set_xlabel(f"{method}-1")
        ax2.set_ylabel(f"{method}-2")
        plt.colorbar(scatter2, ax=ax2, label="True Label")

        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=labels, cmap=palette, alpha=0.6, s=25, edgecolors="none"
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
        plt.colorbar(scatter, ax=ax, label="Cluster Label")

    return _save_fig(fig, "cluster_scatter", config)


# ============================================================
# 2. Optimal K Selection Curves
# ============================================================

def plot_k_selection(k_scores: pd.DataFrame, optimal_k: int, config: dict) -> str:
    """Plot evaluation metrics across different K values."""
    _setup_style(config)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("silhouette", "Silhouette Score (higher=better)", True),
        ("calinski_harabasz", "CH Index (higher=better)", True),
        ("davies_bouldin", "DB Index (lower=better)", False),
    ]

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics):
        ax.plot(k_scores["k"], k_scores[metric], "o-", linewidth=2, markersize=8)
        ax.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7, label=f"Optimal K={optimal_k}")
        ax.set_xlabel("Number of Clusters K", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(k_scores["k"].values)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Optimal Cluster Count Selection", fontsize=16)
    fig.tight_layout()
    return _save_fig(fig, "k_selection", config)


# ============================================================
# 3. Subtype Feature Heatmap
# ============================================================

def plot_subtype_heatmap(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    config: dict,
    top_n: int = 20,
) -> str:
    """
    Plot a heatmap of key features across subtypes.
    Shows the top_n most discriminative features (highest cross-cluster variance).
    """
    _setup_style(config)

    df = feature_df.copy()
    df["cluster"] = labels

    cluster_means = df.groupby("cluster").mean()
    cluster_z = (cluster_means - cluster_means.mean()) / cluster_means.std()

    feat_var = cluster_z.var(axis=0).sort_values(ascending=False)
    top_features = feat_var.head(top_n).index.tolist()
    plot_data = cluster_z[top_features].T

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.4)))
    im = ax.imshow(plot_data.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(cluster_means)))
    ax.set_xticklabels([f"Subtype {i}" for i in cluster_means.index], fontsize=11)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_title(f"Subtype Feature Profile (Top {top_n} Discriminative Features)", fontsize=14)

    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.8)
    fig.tight_layout()
    return _save_fig(fig, "subtype_heatmap", config)


# ============================================================
# 4. Kaplan-Meier Survival Curves
# ============================================================

def plot_survival_curves(
    labels: np.ndarray,
    patient_info: pd.DataFrame,
    config: dict,
) -> str:
    """Plot Kaplan-Meier survival curves per subtype, or mortality bar chart as fallback."""
    _setup_style(config)
    surv_cfg = config["evaluation"]["survival"]
    time_col = surv_cfg["time_col"]
    event_col = surv_cfg["event_col"]

    try:
        from lifelines import KaplanMeierFitter

        fig, ax = plt.subplots(figsize=(10, 7))
        kmf = KaplanMeierFitter()
        n_clusters = len(np.unique(labels))
        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

        for k in range(n_clusters):
            mask = labels == k
            n_k = mask.sum()
            kmf.fit(
                patient_info.loc[mask, time_col],
                event_observed=patient_info.loc[mask, event_col],
                label=f"Subtype {k} (n={n_k})",
            )
            kmf.plot_survival_function(ax=ax, color=colors[k], linewidth=2)

        ax.set_title("Kaplan-Meier Survival Curves by Subtype", fontsize=14)
        ax.set_xlabel("ICU Length of Stay (hours)", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    except ImportError:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        n_clusters = len(np.unique(labels))

        mortality_rates = []
        los_means = []
        for k in range(n_clusters):
            mask = labels == k
            mortality_rates.append(patient_info.loc[mask, event_col].mean())
            los_means.append(patient_info.loc[mask, time_col].mean())

        colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
        x = range(n_clusters)

        ax1.bar(x, mortality_rates, color=colors, edgecolor="black", alpha=0.8)
        ax1.set_xlabel("Subtype", fontsize=12)
        ax1.set_ylabel("28-Day Mortality Rate", fontsize=12)
        ax1.set_title("Mortality by Subtype", fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Subtype {k}" for k in x])

        ax2.bar(x, los_means, color=colors, edgecolor="black", alpha=0.8)
        ax2.set_xlabel("Subtype", fontsize=12)
        ax2.set_ylabel("Mean ICU LOS (hours)", fontsize=12)
        ax2.set_title("ICU Length of Stay by Subtype", fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Subtype {k}" for k in x])

        fig.suptitle("Subtype Outcome Stratification", fontsize=16)
        fig.tight_layout()

    return _save_fig(fig, "survival_curves", config)


# ============================================================
# 5. Missing Pattern Visualization
# ============================================================

def plot_missing_pattern(missing_stats: pd.DataFrame, config: dict) -> str:
    """Plot missing rate per variable as horizontal bars."""
    _setup_style(config)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        missing_stats["variable"],
        missing_stats["missing_rate"],
        color=plt.cm.YlOrRd(missing_stats["missing_rate"]),
        edgecolor="gray",
    )
    ax.set_xlabel("Missing Rate", fontsize=12)
    ax.set_title("Variable Missing Rate Distribution", fontsize=14)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    ax.legend()
    ax.invert_yaxis()

    for bar, rate in zip(bars, missing_stats["missing_rate"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=9)

    fig.tight_layout()
    return _save_fig(fig, "missing_pattern", config)


# ============================================================
# 6. Variable Trajectory Comparison
# ============================================================

def plot_trajectory_comparison(
    time_series_3d: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    config: dict,
    variables: list[str] | None = None,
    n_sample: int = 50,
) -> str:
    """Plot mean +/- std trajectory bands per subtype for key variables."""
    _setup_style(config)

    if variables is None:
        variables = ["heart_rate", "lactate", "map", "creatinine"]

    variables = [v for v in variables if v in feature_names]
    n_vars = len(variables)
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]

    for ax, var_name in zip(axes, variables):
        col_idx = feature_names.index(var_name)
        time_axis = np.arange(time_series_3d.shape[1])

        for k in range(n_clusters):
            mask = labels == k
            subset = time_series_3d[mask, :, col_idx]
            mean_traj = np.nanmean(subset, axis=0)
            std_traj = np.nanstd(subset, axis=0)

            ax.plot(time_axis, mean_traj, color=colors[k], linewidth=2,
                    label=f"Subtype {k}")
            ax.fill_between(time_axis, mean_traj - std_traj, mean_traj + std_traj,
                            color=colors[k], alpha=0.15)

        ax.set_title(var_name, fontsize=13)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Key Variable Trajectories by Subtype", fontsize=16, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "trajectory_comparison", config)


# ============================================================
# 7. Summary Dashboard
# ============================================================

def plot_summary_dashboard(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    k_scores: pd.DataFrame,
    optimal_k: int,
    patient_info: pd.DataFrame,
    config: dict,
) -> str:
    """Generate a 2x2 summary dashboard with scatter, K selection, mortality, and LOS."""
    _setup_style(config)
    n_clusters = len(np.unique(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Top-left: cluster scatter
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels,
                          cmap="Set2", alpha=0.6, s=15, edgecolors="none")
    ax1.set_title("UMAP Cluster Visualization", fontsize=13)
    plt.colorbar(scatter, ax=ax1, label="Subtype")

    # Top-right: K selection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k_scores["k"], k_scores["silhouette"], "o-", linewidth=2, color="#2196F3")
    ax2.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("K")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"Optimal K = {optimal_k}", fontsize=13)
    ax2.set_xticks(k_scores["k"].values)

    # Bottom-left: mortality comparison
    ax3 = fig.add_subplot(gs[1, 0])
    mortality_rates = [patient_info.loc[labels == k, "mortality_28d"].mean()
                       for k in range(n_clusters)]
    ax3.bar(range(n_clusters), mortality_rates, color=colors, edgecolor="black")
    ax3.set_xlabel("Subtype")
    ax3.set_ylabel("28-Day Mortality Rate")
    ax3.set_title("Mortality Stratification", fontsize=13)
    ax3.set_xticks(range(n_clusters))

    # Bottom-right: ICU LOS boxplot
    ax4 = fig.add_subplot(gs[1, 1])
    los_data = [patient_info.loc[labels == k, "icu_los"].values for k in range(n_clusters)]
    bp = ax4.boxplot(los_data, patch_artist=True, labels=[f"Subtype {k}" for k in range(n_clusters)])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax4.set_ylabel("ICU Length of Stay (hours)")
    ax4.set_title("ICU LOS Distribution by Subtype", fontsize=13)

    fig.suptitle("ICU Sepsis Dynamic Subtype Discovery - Summary Dashboard", fontsize=16, y=0.98)

    return _save_fig(fig, "summary_dashboard", config)
