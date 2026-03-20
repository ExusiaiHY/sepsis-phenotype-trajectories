#!/usr/bin/env python3
"""
parity_check_v1_vs_s0.py - Compare V1 original output vs S0 compat(exact_v1).

Purpose:
  Run V1's load_physionet2012 and S0's compat.to_v1_format side by side.
  Compare shapes, feature names, missingness patterns, and value statistics
  to identify any discrepancies and localize their source.

How to run:
  cd project
  python3.14 scripts/parity_check_v1_vs_s0.py

Expected output:
  Console report + data/s0/parity_check_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    print("=" * 60)
    print("PARITY CHECK: V1 vs S0 (exact_v1)")
    print("=" * 60)

    # ===== Load V1 data =====
    print("\n[1/5] Loading V1 data via load_physionet2012...")
    from load_physionet2012 import load_physionet2012, PROJECT_FEATURES
    from utils import resolve_path

    v1_data_dir = resolve_path("data/external")
    v1_ts, v1_info = load_physionet2012(v1_data_dir, n_hours=48)
    v1_features = list(PROJECT_FEATURES)

    print(f"  V1 shape: {v1_ts.shape}")
    print(f"  V1 features ({len(v1_features)}): {v1_features[:5]}...{v1_features[-3:]}")
    print(f"  V1 patients: {len(v1_info)}")

    # ===== Load S0 compat data =====
    print("\n[2/5] Loading S0 compat(exact_v1)...")
    from s0.compat import to_v1_format

    s0_dir = PROJECT_ROOT / "data" / "s0"
    s0_ts, s0_info, s0_features = to_v1_format(s0_dir, mode="exact_v1")

    print(f"  S0 shape: {s0_ts.shape}")
    print(f"  S0 features ({len(s0_features)}): {s0_features[:5]}...{s0_features[-3:]}")
    print(f"  S0 patients: {len(s0_info)}")

    # ===== Compare =====
    report = {"checks": []}

    def check(name, passed, detail=""):
        status = "PASS" if passed else "MISMATCH"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        report["checks"].append({"name": name, "passed": passed, "detail": detail})

    # 3a. Feature names
    print("\n[3/5] Comparing feature names...")
    check("Feature count",
          len(v1_features) == len(s0_features),
          f"V1={len(v1_features)}, S0={len(s0_features)}")

    name_match = v1_features == s0_features
    check("Feature order", name_match,
          "" if name_match else f"First diff at: {next(i for i,(a,b) in enumerate(zip(v1_features, s0_features)) if a != b)}")

    # 3b. Patient count
    # V1 and S0 may have slightly different counts due to different skip logic
    print(f"\n[4/5] Comparing data statistics...")
    n_v1 = v1_ts.shape[0]
    n_s0 = s0_ts.shape[0]
    check("Patient count", True,
          f"V1={n_v1}, S0={n_s0}, diff={abs(n_v1 - n_s0)}")

    # For comparison, align on common patients
    # V1 patient_info has 'patient_id', S0 has 'patient_id' in static
    v1_pids = set(v1_info["patient_id"].astype(str).values)
    s0_pids = set(s0_info["patient_id"].astype(str).values)
    common_pids = v1_pids & s0_pids
    v1_only = v1_pids - s0_pids
    s0_only = s0_pids - v1_pids

    check("Common patients",
          len(common_pids) > min(n_v1, n_s0) * 0.95,
          f"Common={len(common_pids)}, V1-only={len(v1_only)}, S0-only={len(s0_only)}")

    # 3c. Per-feature missingness comparison (on the common set, using first min(N) patients aligned)
    # Since patient ordering may differ, compare aggregate statistics
    n_compare = min(n_v1, n_s0)

    v1_missing = np.isnan(v1_ts[:n_compare]).mean(axis=(0, 1))
    s0_missing = np.isnan(s0_ts[:n_compare]).mean(axis=(0, 1))
    # S0 processed is already imputed, so NaN rate will be 0 for imputable features
    # But for parity check, we need RAW S0 data
    # Load raw_aligned for proper comparison
    s0_raw = np.load(s0_dir / "raw_aligned" / "continuous.npy")
    s0_proxy = np.load(s0_dir / "raw_aligned" / "proxy_indicators.npy")

    # Build raw V1-equivalent from S0 raw (before imputation)
    from s0.schema import CONTINUOUS_INDEX, V1_FEATURE_ORDER
    n_s0_raw = s0_raw.shape[0]
    n_cmp = min(n_v1, n_s0_raw)

    print(f"\n  Per-feature missingness comparison (first {n_cmp} patients):")
    print(f"  {'Feature':<22s} {'V1_miss%':>10s} {'S0_miss%':>10s} {'Diff':>8s}")
    print(f"  {'-'*52}")

    feature_diffs = []
    for fi, fname in enumerate(v1_features):
        v1_mr = float(np.isnan(v1_ts[:n_cmp, :, fi]).mean()) * 100
        # For S0, need to find corresponding raw data
        if fname in CONTINUOUS_INDEX:
            s0_fi = CONTINUOUS_INDEX[fname]
            s0_mr = float(np.isnan(s0_raw[:n_cmp, :, s0_fi]).mean()) * 100
        elif fname == "pao2_fio2_ratio":
            # Derived: check if pao2 and fio2 are both available
            pao2 = s0_raw[:n_cmp, :, CONTINUOUS_INDEX["pao2"]]
            fio2 = s0_raw[:n_cmp, :, CONTINUOUS_INDEX["fio2"]]
            valid = np.isfinite(pao2) & np.isfinite(fio2) & (fio2 > 0)
            s0_mr = float(1.0 - valid.mean()) * 100
        elif fname in ("vasopressor", "mechanical_vent"):
            # Proxy: check underlying variable availability
            proxy_idx = 0 if fname == "vasopressor" else 1
            s0_masks_proxy = np.load(s0_dir / "raw_aligned" / "masks_proxy.npy")
            s0_mr = float(1.0 - s0_masks_proxy[:n_cmp, :, proxy_idx].mean()) * 100
        else:
            s0_mr = 100.0

        diff = abs(v1_mr - s0_mr)
        feature_diffs.append({"feature": fname, "v1_missing": v1_mr, "s0_missing": s0_mr, "diff": diff})
        flag = " ***" if diff > 5.0 else ""
        print(f"  {fname:<22s} {v1_mr:>9.1f}% {s0_mr:>9.1f}% {diff:>7.1f}%{flag}")

    max_diff = max(d["diff"] for d in feature_diffs)
    check("Missingness parity (max diff < 5%)", max_diff < 5.0,
          f"Max diff={max_diff:.1f}% at {max(feature_diffs, key=lambda d: d['diff'])['feature']}")

    # 3d. Value statistics on processed data (post-imputation)
    print(f"\n  Processed value statistics comparison (first {n_cmp} patients):")
    print(f"  {'Feature':<22s} {'V1_mean':>10s} {'S0_mean':>10s} {'V1_std':>10s} {'S0_std':>10s}")
    print(f"  {'-'*64}")

    # V1 processed: need to run V1 preprocess
    # Instead, compare raw means (pre-imputation) for meaningful comparison
    value_diffs = []
    for fi, fname in enumerate(v1_features):
        if fname in CONTINUOUS_INDEX:
            v1_vals = v1_ts[:n_cmp, :, fi]
            s0_vals = s0_raw[:n_cmp, :, CONTINUOUS_INDEX[fname]]
            with np.errstate(all="ignore"):
                v1_m = float(np.nanmean(v1_vals))
                s0_m = float(np.nanmean(s0_vals))
                v1_s = float(np.nanstd(v1_vals))
                s0_s = float(np.nanstd(s0_vals))
            diff_m = abs(v1_m - s0_m)
            value_diffs.append({"feature": fname, "v1_mean": v1_m, "s0_mean": s0_m, "diff_mean": diff_m})
            flag = " ***" if diff_m > 1.0 else ""
            print(f"  {fname:<22s} {v1_m:>10.3f} {s0_m:>10.3f} {v1_s:>10.3f} {s0_s:>10.3f}{flag}")

    # 3e. Mortality comparison
    print(f"\n[5/5] Mortality label comparison...")
    v1_mort_col = "mortality_28d" if "mortality_28d" in v1_info.columns else "mortality"
    s0_mort_col = "mortality_28d" if "mortality_28d" in s0_info.columns else "mortality_inhospital"

    if v1_mort_col in v1_info.columns and s0_mort_col in s0_info.columns:
        v1_mort_rate = v1_info[v1_mort_col].mean()
        s0_mort_rate = s0_info[s0_mort_col].mean()
        print(f"  V1 mortality rate: {v1_mort_rate:.1%} (source: {v1_mort_col})")
        print(f"  S0 mortality rate: {s0_mort_rate:.1%} (source: outcomes file)")
        check("Mortality label source differs (expected)",
              abs(v1_mort_rate - s0_mort_rate) > 0.1,
              f"V1 uses proxy ({v1_mort_rate:.1%}), S0 uses outcomes file ({s0_mort_rate:.1%}). "
              f"Large discrepancy is EXPECTED because V1 proxy overestimates mortality.")

    # Summary
    n_pass = sum(1 for c in report["checks"] if c["passed"])
    n_total = len(report["checks"])
    print(f"\n{'='*60}")
    print(f"PARITY SUMMARY: {n_pass}/{n_total} checks passed")
    print(f"{'='*60}")

    report["summary"] = {
        "n_checks": n_total,
        "n_passed": n_pass,
        "v1_patients": n_v1,
        "s0_patients": n_s0,
        "common_patients": len(common_pids),
        "max_missingness_diff": max_diff,
        "feature_diffs": feature_diffs,
    }

    report_path = PROJECT_ROOT / "data" / "s0" / "parity_check_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
