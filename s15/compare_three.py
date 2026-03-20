"""
compare_three.py - 3-way clustering comparison: PCA vs S1-masked vs S1.5-contrastive.

Multi-seed KMeans evaluation with diagnostics for all three representation families.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("s15.compare")

SEEDS = [42, 123, 456, 789, 2024]


def compare_three_methods(
    s0_dir: Path,
    s1_dir: Path,
    s15_dir: Path,
    k_values: list[int] = (2, 4),
) -> dict:
    """Run 3-way comparison across PCA, S1-masked, S1.5-contrastive."""
    s0_dir, s1_dir, s15_dir = Path(s0_dir), Path(s1_dir), Path(s15_dir)

    static = pd.read_csv(s0_dir / "static.csv")
    mortality = static["mortality_inhospital"].fillna(0).values
    center_ids = static["center_id"].values
    center_a = center_ids == "center_a"
    center_b = center_ids == "center_b"

    methods = {
        "PCA_baseline": np.load(s1_dir / "embeddings_pca.npy"),
        "S1_masked": np.load(s1_dir / "embeddings_ss.npy"),
        "S15_contrastive": np.load(s15_dir / "embeddings_s15.npy"),
    }

    results = {}
    for method_name, emb in methods.items():
        method_results = {}

        for k in k_values:
            seed_metrics = []
            for seed in SEEDS:
                km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=300)
                labels = km.fit_predict(emb)

                sil = silhouette_score(emb, labels)

                cluster_morts = []
                for c in range(k):
                    cm = labels == c
                    cluster_morts.append(float(mortality[cm].mean()) if cm.sum() > 0 else 0.0)

                mort_range = max(cluster_morts) - min(cluster_morts)

                sil_a = silhouette_score(emb[center_a], labels[center_a]) if center_a.sum() > k else float("nan")
                sil_b = silhouette_score(emb[center_b], labels[center_b]) if center_b.sum() > k else float("nan")

                dist_a = np.bincount(labels[center_a], minlength=k) / center_a.sum()
                dist_b = np.bincount(labels[center_b], minlength=k) / center_b.sum()
                dist_l1 = float(np.abs(dist_a - dist_b).sum())

                seed_metrics.append({
                    "seed": seed, "silhouette": sil, "mort_range": mort_range,
                    "mort_min": min(cluster_morts), "mort_max": max(cluster_morts),
                    "sil_center_a": sil_a, "sil_center_b": sil_b,
                    "center_dist_l1": dist_l1,
                })

            agg = _agg(seed_metrics)
            method_results[f"K={k}"] = {"per_seed": seed_metrics, "aggregated": agg}

        results[method_name] = method_results

    _print_table(results, k_values)

    return results


def _agg(metrics):
    keys = ["silhouette", "mort_range", "mort_min", "mort_max",
            "sil_center_a", "sil_center_b", "center_dist_l1"]
    agg = {}
    for key in keys:
        vals = [m[key] for m in metrics if np.isfinite(m[key])]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else {"mean": float("nan"), "std": float("nan")}
    return agg


def _print_table(results, k_values):
    methods = list(results.keys())
    for k in k_values:
        print(f"\n{'='*84}")
        print(f"  K={k} Clustering (mean ± std, {len(SEEDS)} seeds)")
        print(f"{'='*84}")
        print(f"  {'Metric':<24s}", end="")
        for m in methods:
            print(f" {m:>18s}", end="")
        print()
        print(f"  {'-'*82}")

        for label, key in [("Silhouette", "silhouette"), ("Mort range", "mort_range"),
                           ("Mort min", "mort_min"), ("Mort max", "mort_max"),
                           ("Sil Center A", "sil_center_a"), ("Sil Center B", "sil_center_b"),
                           ("Center dist L1", "center_dist_l1")]:
            print(f"  {label:<24s}", end="")
            for m in methods:
                a = results[m][f"K={k}"]["aggregated"].get(key, {})
                mean, std = a.get("mean", float("nan")), a.get("std", float("nan"))
                if np.isfinite(mean):
                    if "mort" in key:
                        print(f" {mean:>7.1%}±{std:>5.1%}   ", end="")
                    else:
                        print(f" {mean:>7.4f}±{std:>5.4f}   ", end="")
                else:
                    print(f" {'N/A':>18s}", end="")
            print()
    print(f"{'='*84}")
