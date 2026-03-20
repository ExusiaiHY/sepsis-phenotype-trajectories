#!/usr/bin/env python3
"""
outcome_audit.py - Audit mortality labels: outcomes file vs proxy.

Purpose:
  Report outcome file coverage, compare proxy-derived mortality against
  ground truth, and quantify the discrepancy.

How to run:
  cd project
  python3.14 scripts/outcome_audit.py

Expected output:
  Console report + data/s0/outcome_audit_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s0.schema import CONTINUOUS_INDEX


def main():
    s0_dir = PROJECT_ROOT / "data" / "s0"
    static = pd.read_csv(s0_dir / "static.csv")
    raw_cont = np.load(s0_dir / "raw_aligned" / "continuous.npy")
    masks = np.load(s0_dir / "raw_aligned" / "masks_continuous.npy")

    n = len(static)
    print("=" * 60)
    print("OUTCOME AUDIT REPORT")
    print("=" * 60)

    # 1. Outcome file coverage
    mort_source = static["mortality_source"].value_counts()
    print(f"\n1. Mortality Label Sources (n={n})")
    for src, count in mort_source.items():
        print(f"   {src}: {count} ({count/n:.1%})")

    mort = static["mortality_inhospital"]
    n_available = mort.notna().sum()
    n_missing = mort.isna().sum()
    mort_rate = mort.mean() if n_available > 0 else float("nan")

    print(f"\n2. Mortality Statistics")
    print(f"   Available labels: {n_available} ({n_available/n:.1%})")
    print(f"   Missing labels:   {n_missing}")
    print(f"   Mortality rate:   {mort_rate:.1%}")

    # 3. Compute proxy mortality for comparison
    gcs_idx = CONTINUOUS_INDEX["gcs"]
    map_idx = CONTINUOUS_INDEX["map"]

    proxy_mort = np.zeros(n, dtype=int)
    for i in range(n):
        gcs_vals = raw_cont[i, :, gcs_idx]
        map_vals = raw_cont[i, :, map_idx]

        valid_gcs = gcs_vals[np.isfinite(gcs_vals)]
        valid_map = map_vals[np.isfinite(map_vals)]

        has_low_gcs = len(valid_gcs) > 0 and np.min(valid_gcs) <= 5
        has_low_map = len(valid_map) >= 3 and np.sum(valid_map < 55) >= 3

        if has_low_gcs or has_low_map:
            proxy_mort[i] = 1

    proxy_rate = proxy_mort.mean()
    print(f"\n3. Proxy Mortality (GCS<=5 or MAP<55 sustained)")
    print(f"   Proxy mortality rate: {proxy_rate:.1%}")
    print(f"   Ground truth rate:    {mort_rate:.1%}")
    print(f"   Discrepancy:          {abs(proxy_rate - mort_rate):.1%}")

    # 4. Agreement analysis (where both available)
    if n_available > 0:
        gt = mort.values.astype(float)
        valid_mask = np.isfinite(gt)
        gt_valid = gt[valid_mask].astype(int)
        proxy_valid = proxy_mort[valid_mask]

        agree = (gt_valid == proxy_valid).sum()
        disagree = (gt_valid != proxy_valid).sum()

        tp = ((gt_valid == 1) & (proxy_valid == 1)).sum()
        fp = ((gt_valid == 0) & (proxy_valid == 1)).sum()
        fn = ((gt_valid == 1) & (proxy_valid == 0)).sum()
        tn = ((gt_valid == 0) & (proxy_valid == 0)).sum()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)

        print(f"\n4. Proxy vs Ground Truth Agreement")
        print(f"   Agreement:    {agree}/{n_available} ({agree/n_available:.1%})")
        print(f"   Disagreement: {disagree}/{n_available} ({disagree/n_available:.1%})")
        print(f"   Sensitivity (proxy detects true death): {sensitivity:.1%}")
        print(f"   Specificity (proxy detects true survival): {specificity:.1%}")
        print(f"   PPV (proxy death is true death): {ppv:.1%}")
        print(f"\n   Confusion matrix:")
        print(f"                    GT=survived  GT=died")
        print(f"   Proxy=survived:  {tn:>8d}     {fn:>6d}")
        print(f"   Proxy=died:      {fp:>8d}     {tp:>6d}")

    # 5. Per-center breakdown
    print(f"\n5. Per-Center Mortality")
    for center in sorted(static["center_id"].dropna().unique()):
        subset = static[static["center_id"] == center]
        m = subset["mortality_inhospital"]
        rate = m.mean() if m.notna().any() else float("nan")
        print(f"   {center}: {len(subset)} patients, mortality={rate:.1%}")

    # Save report
    report = {
        "n_patients": n,
        "mortality_sources": mort_source.to_dict(),
        "mortality_rate_ground_truth": float(mort_rate) if np.isfinite(mort_rate) else None,
        "mortality_rate_proxy": float(proxy_rate),
        "discrepancy": float(abs(proxy_rate - mort_rate)),
        "proxy_sensitivity": float(sensitivity) if n_available > 0 else None,
        "proxy_specificity": float(specificity) if n_available > 0 else None,
        "proxy_ppv": float(ppv) if n_available > 0 else None,
    }
    report_path = s0_dir / "outcome_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
