"""
compare_vs_pca.py - Multi-seed clustering comparison: SS encoder vs PCA baseline.

Purpose:
  Run KMeans with multiple random seeds on both embedding types.
  Report mean ± std for all metrics. No claims of superiority without
  statistically meaningful separation.

Connects to:
  - data/s1/embeddings_ss.npy and embeddings_pca.npy
  - data/s0/static.csv for mortality labels and center_id

Expected output artifacts:
  data/s1/comparison_report.json
  Console table with all metrics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("s1.compare")

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]


def compare_embeddings(
    s0_dir: Path,
    s1_dir: Path,
    k_values: list[int] = (2, 4),
) -> dict:
    """
    Compare SS and PCA embeddings via multi-seed KMeans clustering.

    Returns comparison report dict.
    """
    s0_dir = Path(s0_dir)
    s1_dir = Path(s1_dir)

    # Load embeddings
    emb_ss = np.load(s1_dir / "embeddings_ss.npy")
    emb_pca = np.load(s1_dir / "embeddings_pca.npy")
    static = pd.read_csv(s0_dir / "static.csv")

    mortality = static["mortality_inhospital"].fillna(0).values
    center_ids = static["center_id"].values
    center_a_mask = center_ids == "center_a"
    center_b_mask = center_ids == "center_b"

    n_patients = len(static)
    logger.info(f"Comparing embeddings: SS {emb_ss.shape} vs PCA {emb_pca.shape}")
    logger.info(f"  Patients: {n_patients}, Mortality rate: {mortality.mean():.1%}")
    logger.info(f"  Center A: {center_a_mask.sum()}, Center B: {center_b_mask.sum()}")

    # Load split indices for reporting
    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)

    results = {}
    methods = {"SS_encoder": emb_ss, "PCA_baseline": emb_pca}

    for method_name, embeddings in methods.items():
        method_results = {}

        for k in k_values:
            seed_metrics = []

            for seed in SEEDS:
                km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=300)
                labels = km.fit_predict(embeddings)

                # Silhouette (full cohort)
                sil = silhouette_score(embeddings, labels)

                # Mortality per cluster
                cluster_morts = []
                for c in range(k):
                    cmask = labels == c
                    if cmask.sum() > 0:
                        cluster_morts.append(float(mortality[cmask].mean()))
                    else:
                        cluster_morts.append(0.0)

                mort_range = max(cluster_morts) - min(cluster_morts)

                # Center-wise silhouette
                sil_a = silhouette_score(embeddings[center_a_mask], labels[center_a_mask]) if center_a_mask.sum() > k else float("nan")
                sil_b = silhouette_score(embeddings[center_b_mask], labels[center_b_mask]) if center_b_mask.sum() > k else float("nan")

                # Center-wise cluster distribution similarity
                dist_a = np.bincount(labels[center_a_mask], minlength=k) / center_a_mask.sum()
                dist_b = np.bincount(labels[center_b_mask], minlength=k) / center_b_mask.sum()
                dist_l1 = float(np.abs(dist_a - dist_b).sum())

                seed_metrics.append({
                    "seed": seed,
                    "silhouette": sil,
                    "mort_range": mort_range,
                    "mort_min": min(cluster_morts),
                    "mort_max": max(cluster_morts),
                    "sil_center_a": sil_a,
                    "sil_center_b": sil_b,
                    "center_dist_l1": dist_l1,
                    "cluster_morts": cluster_morts,
                })

            # Aggregate across seeds
            agg = _aggregate_seeds(seed_metrics)
            method_results[f"K={k}"] = {
                "per_seed": seed_metrics,
                "aggregated": agg,
            }

        results[method_name] = method_results

    # Print comparison table
    _print_comparison_table(results, k_values)

    # Save report
    report_path = s1_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")

    return results


def _aggregate_seeds(metrics: list[dict]) -> dict:
    """Compute mean ± std across seeds for each metric."""
    keys = ["silhouette", "mort_range", "mort_min", "mort_max",
            "sil_center_a", "sil_center_b", "center_dist_l1"]
    agg = {}
    for key in keys:
        vals = [m[key] for m in metrics if np.isfinite(m[key])]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        else:
            agg[key] = {"mean": float("nan"), "std": float("nan")}
    return agg


def _print_comparison_table(results: dict, k_values: list[int]) -> None:
    """Print formatted comparison table."""
    methods = list(results.keys())

    for k in k_values:
        print(f"\n{'='*72}")
        print(f"  K={k} Clustering Comparison (mean ± std over {N_SEEDS} seeds)")
        print(f"{'='*72}")
        print(f"  {'Metric':<28s}", end="")
        for m in methods:
            print(f"  {m:>20s}", end="")
        print()
        print(f"  {'-'*70}")

        metrics_to_show = [
            ("Silhouette", "silhouette"),
            ("Mortality range", "mort_range"),
            ("Mortality min", "mort_min"),
            ("Mortality max", "mort_max"),
            ("Sil (Center A)", "sil_center_a"),
            ("Sil (Center B)", "sil_center_b"),
            ("Center dist L1", "center_dist_l1"),
        ]

        for label, key in metrics_to_show:
            print(f"  {label:<28s}", end="")
            for m in methods:
                agg = results[m][f"K={k}"]["aggregated"].get(key, {})
                mean = agg.get("mean", float("nan"))
                std = agg.get("std", float("nan"))
                if np.isfinite(mean):
                    if key in ("mort_range", "mort_min", "mort_max"):
                        print(f"  {mean:>8.1%} ± {std:>6.1%}  ", end="")
                    else:
                        print(f"  {mean:>8.4f} ± {std:>6.4f}  ", end="")
                else:
                    print(f"  {'N/A':>18s}  ", end="")
            print()

    print(f"{'='*72}")
