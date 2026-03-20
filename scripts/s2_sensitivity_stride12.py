#!/usr/bin/env python3
"""
s2_sensitivity_stride12.py - Stride=12h sensitivity analysis for S2-light.

Runs the full S2-light pipeline with stride=12h (3 windows, 50% overlap)
and produces a side-by-side comparison with the primary stride=6h results.

Handles label permutation matching between the two clustering runs via
centroid cosine similarity before any cross-run comparison.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s2_sensitivity_stride12.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger("s2.sensitivity")

    s0_dir = PROJECT_ROOT / "data" / "s0"
    s2_dir = PROJECT_ROOT / "data" / "s2"
    sens_dir = s2_dir / "sensitivity_stride12"
    sens_dir.mkdir(parents=True, exist_ok=True)

    static = pd.read_csv(s0_dir / "static.csv")
    mortality = static["mortality_inhospital"].fillna(0).values
    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)

    K = 4

    # ============================================================
    # Phase 1: Extract stride=12h rolling embeddings
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Extract stride=12h rolling embeddings")
    logger.info("=" * 60)

    from s2light.rolling_embeddings import extract_rolling_embeddings

    emb_12, meta_12 = extract_rolling_embeddings(
        s0_dir=s0_dir,
        encoder_ckpt=PROJECT_ROOT / "data" / "s15" / "checkpoints" / "pretrain_best.pt",
        output_path=sens_dir / "rolling_embeddings.npy",
        window_len=24, stride=12, seq_len=48,
        device="cpu",
    )
    logger.info(f"Stride=12h embeddings: {emb_12.shape}")

    # ============================================================
    # Phase 2: Cluster stride=12h windows
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Temporal Clustering (stride=12h)")
    logger.info("=" * 60)

    from s2light.temporal_clustering import fit_and_assign, save_kmeans_model

    labels_12, quality_12, km_12 = fit_and_assign(
        emb_12, splits, k=K, n_init=20, seed=42,
    )
    np.save(sens_dir / "window_labels.npy", labels_12)
    save_kmeans_model(km_12, sens_dir / "kmeans_model.json")

    # ============================================================
    # Phase 3: Match cluster labels to stride=6h via centroids
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: Label Permutation Matching")
    logger.info("=" * 60)

    # Load stride=6h centroids
    with open(s2_dir / "kmeans_model.json") as f:
        km6_data = json.load(f)
    centers_6h = np.array(km6_data["centers"])  # (K, D)
    centers_12h = km_12.cluster_centers_         # (K, D)

    # Cost matrix: cosine distance between all centroid pairs
    cost = cdist(centers_12h, centers_6h, metric="cosine")  # (K, K)
    row_ind, col_ind = linear_sum_assignment(cost)

    # permutation_map[stride12h_label] = matched_stride6h_label
    permutation_map = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    match_costs = {f"{r}→{c}": round(float(cost[r, c]), 4) for r, c in zip(row_ind, col_ind)}

    logger.info(f"Cluster matching (cosine distance):")
    for r, c in zip(row_ind, col_ind):
        logger.info(f"  Stride12h cluster {r} → Stride6h cluster {c} (distance={cost[r, c]:.4f})")

    # Relabel stride=12h to align with stride=6h
    labels_12_aligned = np.vectorize(permutation_map.get)(labels_12)

    # ============================================================
    # Phase 4: Transition analysis (stride=12h, aligned labels)
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Transition Analysis (stride=12h)")
    logger.info("=" * 60)

    from s2light.transition_analysis import compute_transitions

    trans_12 = compute_transitions(labels_12_aligned, static, splits, k=K)

    pl = trans_12["patient_level"]
    el = trans_12["event_level"]
    logger.info(f"Patient-level: stable={pl['stable_fraction']:.1%}, "
                f"single={pl['single_transition_fraction']:.1%}, "
                f"multi={pl['multi_transition_fraction']:.1%}")
    logger.info(f"Event-level: non-self={el['non_self_transition_events']}/{el['total_transition_events']} "
                f"({el['non_self_fraction']:.1%})")

    with open(sens_dir / "transition_matrix.json", "w") as f:
        json.dump(el, f, indent=2, default=str)
    with open(sens_dir / "trajectory_stats.json", "w") as f:
        json.dump(trans_12, f, indent=2, default=str)

    # ============================================================
    # Phase 5: Load stride=6h results for comparison
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Side-by-Side Comparison")
    logger.info("=" * 60)

    with open(s2_dir / "trajectory_stats.json") as f:
        trans_6 = json.load(f)

    pl6 = trans_6["patient_level"]
    el6 = trans_6["event_level"]

    # Stride=6h stable phenotype mortality (already aligned to its own labels)
    mort_stable_6h = trans_6["mortality_by_stable_phenotype"]

    # Stride=12h stable phenotype mortality (using aligned labels)
    mort_stable_12h = {}
    for c in range(K):
        stable_in_c = np.all(labels_12_aligned == c, axis=1)
        n = int(stable_in_c.sum())
        rate = round(float(mortality[stable_in_c].mean()), 4) if n > 0 else None
        mort_stable_12h[f"cluster_{c}"] = {"n": n, "mortality_rate": rate}

    # === Build comparison table ===
    comparison = {
        "label_matching": {
            "method": "Hungarian algorithm on cosine distance between KMeans centroids",
            "permutation_map": permutation_map,
            "match_costs": match_costs,
        },
        "side_by_side": {},
    }

    metrics = [
        ("Windows per patient", 5, trans_12["n_windows"]),
        ("Adjacent pairs per patient", 4, trans_12["n_windows"] - 1),
        ("Overall window silhouette", quality_12.get("overall_silhouette", None),
         quality_12.get("overall_silhouette", None)),
        ("Stable patient fraction", pl6["stable_fraction"], pl["stable_fraction"]),
        ("Single-transition fraction", pl6["single_transition_fraction"], pl["single_transition_fraction"]),
        ("Multi-transition fraction", pl6["multi_transition_fraction"], pl["multi_transition_fraction"]),
        ("Non-self transition proportion", el6["non_self_fraction"], el["non_self_fraction"]),
        ("Non-self transition count", el6["non_self_transition_events"], el["non_self_transition_events"]),
        ("Transition entropy ratio", el6["entropy_ratio"], el["entropy_ratio"]),
    ]

    # Add mortality by stable phenotype
    for c in range(K):
        key6 = f"cluster_{c}"
        m6 = mort_stable_6h.get(key6, {}).get("mortality_rate", None)
        m12 = mort_stable_12h.get(key6, {}).get("mortality_rate", None)
        n6 = mort_stable_6h.get(key6, {}).get("n", 0)
        n12 = mort_stable_12h.get(key6, {}).get("n", 0)
        metrics.append((f"Stable P{c} mortality (n6={n6},n12={n12})", m6, m12))

    # Mortality by trajectory category
    md6 = trans_6["mortality_descriptives"]["all_cohort"]
    md12 = trans_12["mortality_descriptives"]["all_cohort"]
    for cat in ["stable", "single_transition", "multi_transition"]:
        m6 = md6[cat]["mortality_rate"]
        m12 = md12[cat]["mortality_rate"]
        metrics.append((f"Mort ({cat})", m6, m12))

    # Fix the overall silhouette entry
    # Load stride=6h sanity for its silhouette
    with open(s2_dir / "sanity_checks.json") as f:
        sanity_6 = json.load(f)
    sil_6 = sanity_6["cluster_quality"]["overall_silhouette"]
    sil_12 = quality_12["overall_silhouette"]
    metrics[2] = ("Overall window silhouette", sil_6, sil_12)

    # Print table
    print(f"\n{'='*80}")
    print(f"  STRIDE SENSITIVITY: 6h vs 12h")
    print(f"{'='*80}")
    print(f"  {'Metric':<45s} {'Stride=6h':>14s} {'Stride=12h':>14s}")
    print(f"  {'-'*78}")

    for name, v6, v12 in metrics:
        s6 = f"{v6:.1%}" if isinstance(v6, float) and v6 < 1.0 and "fraction" in name.lower() or "mort" in name.lower() or "proportion" in name.lower() else str(v6) if v6 is not None else "N/A"
        s12 = f"{v12:.1%}" if isinstance(v12, float) and v12 < 1.0 and "fraction" in name.lower() or "mort" in name.lower() or "proportion" in name.lower() else str(v12) if v12 is not None else "N/A"

        # Better formatting
        if isinstance(v6, float) and "silhouette" in name.lower():
            s6 = f"{v6:.4f}"
            s12 = f"{v12:.4f}" if isinstance(v12, float) else "N/A"
        elif isinstance(v6, float) and ("mort" in name.lower() or "fraction" in name.lower() or "proportion" in name.lower()):
            s6 = f"{v6:.1%}" if v6 is not None else "N/A"
            s12 = f"{v12:.1%}" if v12 is not None else "N/A"
        elif isinstance(v6, float) and "entropy" in name.lower():
            s6 = f"{v6:.3f}"
            s12 = f"{v12:.3f}" if isinstance(v12, float) else "N/A"
        elif isinstance(v6, int):
            s6 = f"{v6:,}"
            s12 = f"{v12:,}" if isinstance(v12, int) else str(v12)
        else:
            s6 = str(v6) if v6 is not None else "N/A"
            s12 = str(v12) if v12 is not None else "N/A"

        print(f"  {name:<45s} {s6:>14s} {s12:>14s}")

    print(f"{'='*80}")

    # === Robustness assessment ===
    print(f"\n{'='*80}")
    print(f"  ROBUSTNESS ASSESSMENT")
    print(f"{'='*80}")

    # Criterion 1: Non-self transition proportion
    prop_diff = abs(el6["non_self_fraction"] - el["non_self_fraction"])
    prop_robust = prop_diff < 0.03
    print(f"  Non-self transition proportion: 6h={el6['non_self_fraction']:.1%}, "
          f"12h={el['non_self_fraction']:.1%}, diff={prop_diff:.1%} "
          f"{'[ROBUST]' if prop_robust else '[SENSITIVE]'}")

    # Criterion 2: Stable phenotype mortality ordering
    mort_6_vals = [(c, mort_stable_6h[f"cluster_{c}"]["mortality_rate"])
                   for c in range(K) if mort_stable_6h[f"cluster_{c}"]["mortality_rate"] is not None]
    mort_12_vals = [(c, mort_stable_12h[f"cluster_{c}"]["mortality_rate"])
                    for c in range(K) if mort_stable_12h[f"cluster_{c}"]["mortality_rate"] is not None]

    order_6 = [c for c, _ in sorted(mort_6_vals, key=lambda x: x[1])]
    order_12 = [c for c, _ in sorted(mort_12_vals, key=lambda x: x[1])]
    order_preserved = order_6 == order_12
    print(f"  Mortality ordering: 6h={order_6}, 12h={order_12} "
          f"{'[PRESERVED]' if order_preserved else '[CHANGED]'}")

    # Criterion 3: Highest-risk phenotype
    highest_6 = max(mort_6_vals, key=lambda x: x[1])
    highest_12 = max(mort_12_vals, key=lambda x: x[1])
    same_highest = highest_6[0] == highest_12[0]
    print(f"  Highest-risk stable phenotype: 6h=P{highest_6[0]} ({highest_6[1]:.1%}), "
          f"12h=P{highest_12[0]} ({highest_12[1]:.1%}) "
          f"{'[SAME]' if same_highest else '[DIFFERENT]'}")

    # Criterion 4: Mortality range
    range_6 = max(v for _, v in mort_6_vals) - min(v for _, v in mort_6_vals)
    range_12 = max(v for _, v in mort_12_vals) - min(v for _, v in mort_12_vals)
    range_meaningful = range_12 > 0.15  # >15pp is clinically meaningful
    print(f"  Mortality range: 6h={range_6:.1%}, 12h={range_12:.1%} "
          f"{'[MEANINGFUL]' if range_meaningful else '[WEAK]'}")

    # Overall conclusion
    n_robust = sum([prop_robust, order_preserved, same_highest, range_meaningful])
    if n_robust == 4:
        conclusion = "ROBUST: All criteria passed. Stride=6h findings are confirmed under reduced overlap."
        manuscript_wording = "qualitatively consistent under reduced overlap (stride=12h, 50% overlap)"
    elif n_robust >= 3:
        conclusion = "MOSTLY ROBUST: 3/4 criteria passed. Minor sensitivity to overlap, but core findings hold."
        manuscript_wording = "largely consistent under reduced overlap, with minor quantitative differences"
    elif n_robust >= 2:
        conclusion = "PARTIALLY ROBUST: Core mortality stratification holds but transition rates are overlap-sensitive."
        manuscript_wording = "transition frequency is overlap-sensitive, but stable phenotype risk stratification remains consistent"
    else:
        conclusion = "NOT ROBUST: Findings are materially influenced by window overlap."
        manuscript_wording = "temporal transition findings require cautious interpretation due to overlap sensitivity"

    print(f"\n  CONCLUSION: {conclusion}")
    print(f"  MANUSCRIPT WORDING: \"{manuscript_wording}\"")
    print(f"{'='*80}")

    # Save comparison
    comparison["robustness_assessment"] = {
        "transition_proportion_robust": prop_robust,
        "transition_proportion_diff": round(prop_diff, 4),
        "mortality_ordering_preserved": order_preserved,
        "order_6h": order_6,
        "order_12h": order_12,
        "highest_risk_same": same_highest,
        "mortality_range_6h": round(range_6, 4),
        "mortality_range_12h": round(range_12, 4),
        "mortality_range_meaningful": range_meaningful,
        "criteria_passed": n_robust,
        "conclusion": conclusion,
        "manuscript_wording": manuscript_wording,
    }
    comparison["stride_6h_summary"] = {
        "stable_fraction": pl6["stable_fraction"],
        "non_self_proportion": el6["non_self_fraction"],
        "entropy_ratio": el6["entropy_ratio"],
        "mortality_by_stable_phenotype": mort_stable_6h,
    }
    comparison["stride_12h_summary"] = {
        "stable_fraction": pl["stable_fraction"],
        "non_self_proportion": el["non_self_fraction"],
        "entropy_ratio": el["entropy_ratio"],
        "mortality_by_stable_phenotype": mort_stable_12h,
    }

    with open(s2_dir / "sensitivity_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"Comparison saved: {s2_dir / 'sensitivity_comparison.json'}")


if __name__ == "__main__":
    main()
