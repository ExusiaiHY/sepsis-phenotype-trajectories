#!/usr/bin/env python3
"""
s3_cross_center_validation.py - Cross-center temporal phenotype validation.

S3 asks: do the temporal phenotype trajectories discovered on Center A
(train/val) hold when applied to Center B (test)?

Design:
  - The KMeans model was fit on Center A rolling-window embeddings in S2.
  - The S1.5 encoder was pretrained on Center A data only.
  - Center B window labels were already assigned by the same KMeans in S2.
  - S3 extracts Center-B-specific results and compares them to Center A.

No new training. No new encoder. This is pure evaluation.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s3_cross_center_validation.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("s3")


def main():
    s0_dir = PROJECT_ROOT / "data" / "s0"
    s2_dir = PROJECT_ROOT / "data" / "s2"
    s3_dir = PROJECT_ROOT / "data" / "s3"
    s3_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    static = pd.read_csv(s0_dir / "static.csv")
    mortality = static["mortality_inhospital"].fillna(0).values
    center_ids = static["center_id"].values

    with open(s0_dir / "splits.json") as f:
        splits = json.load(f)

    rolling_emb = np.load(s2_dir / "rolling_embeddings.npy")  # (N, 5, 128)
    window_labels = np.load(s2_dir / "window_labels.npy")      # (N, 5)

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])
    center_a_idx = np.where(center_ids == "center_a")[0]
    center_b_idx = np.where(center_ids == "center_b")[0]

    N, W, D = rolling_emb.shape
    K = 4

    logger.info("=" * 65)
    logger.info("S3: CROSS-CENTER TEMPORAL PHENOTYPE VALIDATION")
    logger.info("=" * 65)
    logger.info(f"Center A: {len(center_a_idx)} patients (train+val)")
    logger.info(f"Center B: {len(center_b_idx)} patients (test)")

    report = {
        "center_a_n": len(center_a_idx),
        "center_b_n": len(center_b_idx),
        "n_windows": W,
        "k": K,
    }

    # ============================================================
    # 1. Per-center phenotype prevalence at each window
    # ============================================================
    logger.info("\n--- 1. Per-Window Phenotype Prevalence by Center ---")

    prevalence_a = np.zeros((W, K))
    prevalence_b = np.zeros((W, K))
    for wi in range(W):
        for c in range(K):
            prevalence_a[wi, c] = (window_labels[center_a_idx, wi] == c).mean()
            prevalence_b[wi, c] = (window_labels[center_b_idx, wi] == c).mean()

    prevalence_l1_per_window = []
    for wi in range(W):
        l1 = float(np.abs(prevalence_a[wi] - prevalence_b[wi]).sum())
        prevalence_l1_per_window.append(l1)
        logger.info(f"  W{wi}: A={prevalence_a[wi].round(3).tolist()} "
                     f"B={prevalence_b[wi].round(3).tolist()} L1={l1:.4f}")

    mean_l1 = float(np.mean(prevalence_l1_per_window))
    logger.info(f"  Mean L1 across windows: {mean_l1:.4f}")

    report["prevalence"] = {
        "center_a": prevalence_a.round(4).tolist(),
        "center_b": prevalence_b.round(4).tolist(),
        "l1_per_window": [round(x, 4) for x in prevalence_l1_per_window],
        "mean_l1": round(mean_l1, 4),
    }

    # ============================================================
    # 2. Per-center transition statistics
    # ============================================================
    logger.info("\n--- 2. Transition Statistics by Center ---")

    def compute_center_transitions(idx, label):
        labels_sub = window_labels[idx]
        mort_sub = mortality[idx]
        n = len(idx)

        # Patient-level
        n_trans = np.array([
            sum(1 for t in range(W - 1) if labels_sub[i, t] != labels_sub[i, t + 1])
            for i in range(n)
        ])
        stable = n_trans == 0
        single = n_trans == 1
        multi = n_trans >= 2

        # Event-level
        trans_counts = np.zeros((K, K), dtype=int)
        for t in range(W - 1):
            for i in range(n):
                trans_counts[labels_sub[i, t], labels_sub[i, t + 1]] += 1
        total = trans_counts.sum()
        non_self = total - np.trace(trans_counts)

        # Mortality by stable phenotype
        mort_stable = {}
        for c in range(K):
            mask = stable & (labels_sub[:, 0] == c)
            nc = int(mask.sum())
            rate = round(float(mort_sub[mask].mean()), 4) if nc > 0 else None
            mort_stable[f"P{c}"] = {"n": nc, "mortality": rate}

        # Mortality by trajectory category
        mort_by_cat = {
            "stable": {"n": int(stable.sum()), "mortality": round(float(mort_sub[stable].mean()), 4) if stable.sum() > 0 else None},
            "single": {"n": int(single.sum()), "mortality": round(float(mort_sub[single].mean()), 4) if single.sum() > 0 else None},
            "multi": {"n": int(multi.sum()), "mortality": round(float(mort_sub[multi].mean()), 4) if multi.sum() > 0 else None},
        }

        result = {
            "n_patients": n,
            "stable_fraction": round(float(stable.mean()), 4),
            "single_fraction": round(float(single.mean()), 4),
            "multi_fraction": round(float(multi.mean()), 4),
            "non_self_proportion": round(float(non_self / max(total, 1)), 4),
            "non_self_count": int(non_self),
            "total_events": int(total),
            "transition_matrix": trans_counts.tolist(),
            "mortality_by_stable_phenotype": mort_stable,
            "mortality_by_category": mort_by_cat,
        }

        logger.info(f"  [{label}] n={n}, stable={result['stable_fraction']:.1%}, "
                     f"non-self={result['non_self_proportion']:.1%}")
        for c in range(K):
            ms = mort_stable[f"P{c}"]
            logger.info(f"    Stable P{c}: n={ms['n']}, mortality={ms['mortality']}")

        return result

    report["center_a"] = compute_center_transitions(center_a_idx, "Center A")
    report["center_b"] = compute_center_transitions(center_b_idx, "Center B")

    # ============================================================
    # 3. Per-center window-level silhouette
    # ============================================================
    logger.info("\n--- 3. Per-Window Silhouette by Center ---")

    sil_a_per_w = []
    sil_b_per_w = []
    for wi in range(W):
        emb_a = rolling_emb[center_a_idx, wi, :]
        lab_a = window_labels[center_a_idx, wi]
        emb_b = rolling_emb[center_b_idx, wi, :]
        lab_b = window_labels[center_b_idx, wi]

        sa = float(silhouette_score(emb_a, lab_a)) if len(np.unique(lab_a)) > 1 else float("nan")
        sb = float(silhouette_score(emb_b, lab_b)) if len(np.unique(lab_b)) > 1 else float("nan")
        sil_a_per_w.append(round(sa, 4))
        sil_b_per_w.append(round(sb, 4))
        logger.info(f"  W{wi}: A={sa:.4f}, B={sb:.4f}")

    report["silhouette_per_window"] = {
        "center_a": sil_a_per_w,
        "center_b": sil_b_per_w,
    }

    # ============================================================
    # 4. Cross-center robustness assessment
    # ============================================================
    logger.info("\n--- 4. Cross-Center Robustness Assessment ---")

    ca = report["center_a"]
    cb = report["center_b"]

    # Criterion 1: Stable fraction similarity
    stable_diff = abs(ca["stable_fraction"] - cb["stable_fraction"])
    stable_similar = stable_diff < 0.05
    logger.info(f"  Stable fraction: A={ca['stable_fraction']:.1%}, B={cb['stable_fraction']:.1%}, "
                f"diff={stable_diff:.1%} {'[SIMILAR]' if stable_similar else '[DIFFERENT]'}")

    # Criterion 2: Non-self transition proportion similarity
    trans_diff = abs(ca["non_self_proportion"] - cb["non_self_proportion"])
    trans_similar = trans_diff < 0.03
    logger.info(f"  Non-self proportion: A={ca['non_self_proportion']:.1%}, B={cb['non_self_proportion']:.1%}, "
                f"diff={trans_diff:.1%} {'[SIMILAR]' if trans_similar else '[DIFFERENT]'}")

    # Criterion 3: Mortality ordering of stable phenotypes
    def mortality_order(mort_dict):
        items = [(k, v["mortality"]) for k, v in mort_dict.items() if v["mortality"] is not None and v["n"] >= 10]
        return [k for k, _ in sorted(items, key=lambda x: x[1])]

    order_a = mortality_order(ca["mortality_by_stable_phenotype"])
    order_b = mortality_order(cb["mortality_by_stable_phenotype"])
    order_match = order_a == order_b
    logger.info(f"  Mortality ordering: A={order_a}, B={order_b} "
                f"{'[MATCH]' if order_match else '[MISMATCH]'}")

    # Criterion 4: Highest-risk phenotype same
    highest_a = max(ca["mortality_by_stable_phenotype"].items(),
                    key=lambda x: x[1]["mortality"] if x[1]["mortality"] is not None else 0)
    highest_b = max(cb["mortality_by_stable_phenotype"].items(),
                    key=lambda x: x[1]["mortality"] if x[1]["mortality"] is not None else 0)
    highest_same = highest_a[0] == highest_b[0]
    logger.info(f"  Highest-risk: A={highest_a[0]} ({highest_a[1]['mortality']:.1%}), "
                f"B={highest_b[0]} ({highest_b[1]['mortality']:.1%}) "
                f"{'[SAME]' if highest_same else '[DIFFERENT]'}")

    # Criterion 5: Mortality range clinically meaningful on Center B
    mort_vals_b = [v["mortality"] for v in cb["mortality_by_stable_phenotype"].values()
                   if v["mortality"] is not None and v["n"] >= 10]
    range_b = max(mort_vals_b) - min(mort_vals_b) if len(mort_vals_b) >= 2 else 0
    range_meaningful = range_b > 0.15
    logger.info(f"  Center B mortality range: {range_b:.1%} "
                f"{'[MEANINGFUL]' if range_meaningful else '[WEAK]'}")

    # Criterion 6: Prevalence L1 acceptably low
    prev_ok = mean_l1 < 0.10
    logger.info(f"  Mean prevalence L1: {mean_l1:.4f} {'[LOW]' if prev_ok else '[HIGH]'}")

    n_pass = sum([stable_similar, trans_similar, order_match, highest_same, range_meaningful, prev_ok])

    if n_pass == 6:
        conclusion = "STRONG CROSS-CENTER GENERALIZATION: All 6 criteria passed."
        claim_level = "robust cross-center generalization"
    elif n_pass >= 5:
        conclusion = "GOOD CROSS-CENTER GENERALIZATION: 5/6 criteria passed."
        claim_level = "cross-center generalization with minor quantitative differences"
    elif n_pass >= 4:
        conclusion = "MODERATE CROSS-CENTER CONSISTENCY: 4/6 criteria passed."
        claim_level = "cross-center consistency in core phenotype structure"
    elif n_pass >= 3:
        conclusion = "PARTIAL CROSS-CENTER CONSISTENCY: 3/6 criteria passed."
        claim_level = "preliminary cross-center consistency"
    else:
        conclusion = "WEAK CROSS-CENTER EVIDENCE: <3 criteria passed."
        claim_level = "limited cross-center evidence"

    logger.info(f"\n  CONCLUSION: {conclusion}")
    logger.info(f"  CLAIM LEVEL: \"{claim_level}\"")

    report["robustness"] = {
        "stable_fraction_similar": stable_similar,
        "stable_fraction_diff": round(stable_diff, 4),
        "transition_proportion_similar": trans_similar,
        "transition_proportion_diff": round(trans_diff, 4),
        "mortality_ordering_match": order_match,
        "order_a": order_a,
        "order_b": order_b,
        "highest_risk_same": highest_same,
        "center_b_mortality_range": round(range_b, 4),
        "range_meaningful": range_meaningful,
        "prevalence_l1_low": prev_ok,
        "mean_prevalence_l1": round(mean_l1, 4),
        "criteria_passed": n_pass,
        "conclusion": conclusion,
        "claim_level": claim_level,
    }

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n{'='*75}")
    print(f"  CROSS-CENTER COMPARISON: Center A (train) vs Center B (test)")
    print(f"{'='*75}")
    print(f"  {'Metric':<40s} {'Center A':>14s} {'Center B':>14s}")
    print(f"  {'-'*73}")
    print(f"  {'Patients':<40s} {ca['n_patients']:>14d} {cb['n_patients']:>14d}")
    print(f"  {'Stable fraction':<40s} {ca['stable_fraction']:>13.1%} {cb['stable_fraction']:>13.1%}")
    print(f"  {'Single-transition fraction':<40s} {ca['single_fraction']:>13.1%} {cb['single_fraction']:>13.1%}")
    print(f"  {'Multi-transition fraction':<40s} {ca['multi_fraction']:>13.1%} {cb['multi_fraction']:>13.1%}")
    print(f"  {'Non-self transition proportion':<40s} {ca['non_self_proportion']:>13.1%} {cb['non_self_proportion']:>13.1%}")

    for c in range(K):
        pa = ca['mortality_by_stable_phenotype'][f'P{c}']
        pb = cb['mortality_by_stable_phenotype'][f'P{c}']
        ma = f"{pa['mortality']:.1%}" if pa['mortality'] is not None else "N/A"
        mb = f"{pb['mortality']:.1%}" if pb['mortality'] is not None else "N/A"
        print(f"  {'Stable P' + str(c) + ' mortality':<40s} {ma:>14s} {mb:>14s}")

    print(f"  {'Mortality ordering':<40s} {str(order_a):>14s} {str(order_b):>14s}")
    print(f"  {'Mean prevalence L1':<40s} {'':>14s} {mean_l1:>14.4f}")
    print(f"{'='*75}")
    print(f"  Criteria passed: {n_pass}/6")
    print(f"  {conclusion}")
    print(f"  Claim level: \"{claim_level}\"")
    print(f"{'='*75}")

    # Save
    with open(s3_dir / "cross_center_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nReport saved: {s3_dir / 'cross_center_report.json'}")


if __name__ == "__main__":
    main()
