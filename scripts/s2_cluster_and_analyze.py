#!/usr/bin/env python3
"""
s2_cluster_and_analyze.py - Clustering + transitions + sanity checks + figures.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s2_cluster_and_analyze.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger("s2")

    with open(PROJECT_ROOT / "config" / "s2_config.yaml") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s2_dir = PROJECT_ROOT / cfg["paths"]["s2_dir"]
    fig_dir = s2_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rw = cfg["rolling_windows"]
    k = cfg["clustering"]["k"]

    # Load data
    rolling_emb = np.load(s2_dir / "rolling_embeddings.npy")
    static = pd.read_csv(s0_dir / "static.csv")
    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)
    with open(s2_dir / "rolling_meta.json") as f:
        rolling_meta = json.load(f)

    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")

    logger.info(f"Rolling embeddings: {rolling_emb.shape}")
    logger.info(f"Static: {len(static)} patients")

    # ============================================================
    # Phase 1: Temporal clustering
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Temporal Clustering")
    logger.info("=" * 60)

    from s2light.temporal_clustering import fit_and_assign, save_kmeans_model

    window_labels, cluster_quality, km = fit_and_assign(
        rolling_emb, splits, k=k,
        n_init=cfg["clustering"]["n_init"],
        seed=cfg["clustering"]["seed"],
    )

    np.save(s2_dir / "window_labels.npy", window_labels)
    save_kmeans_model(km, s2_dir / "kmeans_model.json")

    # ============================================================
    # Phase 2: Window-level data quality
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Window-Level Data Quality")
    logger.info("=" * 60)

    window_starts = rolling_meta["window_starts"]
    window_len = rw["window_len"]
    N, W = window_labels.shape

    window_quality = []
    for wi, start in enumerate(window_starts):
        # Observation density for this window
        window_mask = masks[:, start:start + window_len, :]
        obs_density = window_mask.mean(axis=(1, 2))

        n_sparse = int((obs_density < 0.05).sum())

        wq = {
            "window_idx": wi,
            "hours": f"[{start},{start + window_len})",
            "obs_density_mean": round(float(obs_density.mean()), 4),
            "obs_density_std": round(float(obs_density.std()), 4),
            "n_extremely_sparse": n_sparse,
            "silhouette": cluster_quality["per_window"][wi]["silhouette"],
            "cluster_fractions": cluster_quality["per_window"][wi]["cluster_fractions"],
        }
        window_quality.append(wq)
        logger.info(f"  W{wi} {wq['hours']}: obs_density={wq['obs_density_mean']:.3f}, "
                     f"sil={wq['silhouette']:.4f}, sparse={n_sparse}")

    # ============================================================
    # Phase 3: Transition analysis
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Transition Analysis")
    logger.info("=" * 60)

    from s2light.transition_analysis import compute_transitions

    transition_report = compute_transitions(window_labels, static, splits, k=k)

    pl = transition_report["patient_level"]
    el = transition_report["event_level"]

    logger.info(f"Patient-level: stable={pl['stable_fraction']:.1%}, "
                f"single={pl['single_transition_fraction']:.1%}, "
                f"multi={pl['multi_transition_fraction']:.1%}")
    logger.info(f"Event-level: non-self={el['non_self_transition_events']}/{el['total_transition_events']} "
                f"({el['non_self_fraction']:.1%})")
    logger.info(f"Transition entropy ratio: {el['entropy_ratio']:.3f}")
    logger.info(f"Stability flag: {transition_report['stability_assessment']['flag']}")

    # Mortality by trajectory
    md = transition_report["mortality_descriptives"]["all_cohort"]
    for cat_name, cat_data in md.items():
        rate = cat_data['mortality_rate']
        n = cat_data['n']
        logger.info(f"  {cat_name}: n={n}, mortality={rate:.1%}" if rate else f"  {cat_name}: n={n}")

    # Per-split
    for split_name in ["train", "val", "test"]:
        sd = transition_report["mortality_descriptives"][split_name]
        logger.info(f"  [{split_name}] stable: n={sd['stable']['n']}, "
                     f"mort={sd['stable']['mortality_rate']}")

    # ============================================================
    # Phase 4: Save all results
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Save Results")
    logger.info("=" * 60)

    with open(s2_dir / "transition_matrix.json", "w") as f:
        json.dump(transition_report["event_level"], f, indent=2, default=str)

    with open(s2_dir / "trajectory_stats.json", "w") as f:
        json.dump(transition_report, f, indent=2, default=str)

    sanity = {
        "n_patients": N,
        "n_windows_per_patient": W,
        "window_quality": window_quality,
        "cluster_quality": cluster_quality,
        "stability_assessment": transition_report["stability_assessment"],
        "patient_level_summary": transition_report["patient_level"],
        "event_level_summary": {
            "total": el["total_transition_events"],
            "non_self": el["non_self_transition_events"],
            "non_self_fraction": el["non_self_fraction"],
            "entropy_ratio": el["entropy_ratio"],
        },
    }
    with open(s2_dir / "sanity_checks.json", "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    # ============================================================
    # Phase 5: Visualization
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Visualization")
    logger.info("=" * 60)

    from s2light.visualization import (
        plot_per_window_prevalence,
        plot_sankey_transitions,
        plot_mortality_by_trajectory,
    )

    plot_per_window_prevalence(
        window_labels, window_starts, window_len,
        fig_dir / "per_window_prevalence.png", k=k,
    )

    plot_sankey_transitions(
        window_labels, fig_dir / "sankey_transitions.png", k=k,
    )

    plot_mortality_by_trajectory(
        transition_report, fig_dir / "mortality_by_trajectory.png",
    )

    # ============================================================
    # Summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("S2-LIGHT COMPLETE")
    logger.info(f"  Patients: {N}, Windows: {W}, K: {k}")
    logger.info(f"  Stable: {pl['stable_fraction']:.1%}")
    logger.info(f"  Non-self transitions: {el['non_self_fraction']:.1%}")
    logger.info(f"  Figures: {fig_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
