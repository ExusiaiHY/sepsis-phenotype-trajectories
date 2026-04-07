#!/usr/bin/env python3
"""
s6_build_enhanced_mimic.py - Stage 6 Phase 1: Enhanced MIMIC-IV extraction + split subtype labels

This script:
  1. Builds enhanced analysis tables from the local MIMIC-IV DuckDB
     (adds blood differential, inflammation markers, enzymes,
      microbiology cultures, ventilation status, ferritin/D-dimer).
  2. Generates split subtype labels (gold, proxy, score, mask).
  3. Reshapes the enhanced time-series into 3D tensors compatible with
     the existing S1.5 representation-learning and S5 real-time pipeline.
  4. Exports a fused patient_info file that contains both the original
     outcomes AND the new multi-task subtype labels.

Usage:
  export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  python scripts/s6_build_enhanced_mimic.py \
      --db-path db/mimic4_real.db \
      --processed-dir data/processed_mimic_enhanced \
      --hours 48

Integration with existing pipeline:
  python src/main.py --source mimic \
      --processed-dir data/processed_mimic_enhanced \
      --tag mimic_enhanced
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_enhanced_analysis_table import build_enhanced_analysis_tables
from subtype_label_engine import build_subtype_labels


def reshape_to_3d(ts_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Reshape long-form time-series DataFrame into (n_patients, n_timesteps, n_features).
    Uses vectorized indexing for speed on large cohorts.
    """
    stay_ids = sorted(ts_df["stay_id"].unique())
    n_patients = len(stay_ids)
    n_timesteps = ts_df["hr"].nunique()
    n_features = len(feature_cols)

    stay_to_idx = {sid: i for i, sid in enumerate(stay_ids)}

    # Sort once and map stay_id -> row index via vectorized map
    ts_sorted = ts_df.sort_values(["stay_id", "hr"]).reset_index(drop=True)
    row_idx = ts_sorted["stay_id"].map(stay_to_idx).values
    col_idx = ts_sorted["hr"].values

    tensor = np.full((n_patients, n_timesteps, n_features), np.nan, dtype=np.float32)
    vals = ts_sorted[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan).astype(np.float32)
    tensor[row_idx, col_idx, :] = vals
    return tensor


def main():
    parser = argparse.ArgumentParser(description="Build enhanced MIMIC-IV tables with split subtype labels")
    parser.add_argument("--db-path", type=str, default=str(PROJECT_ROOT / "db" / "mimic4_real.db"))
    parser.add_argument("--processed-dir", type=str, default=str(PROJECT_ROOT / "data" / "processed_mimic_enhanced"))
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"])
    parser.add_argument("--skip-build", action="store_true", help="Skip table build if outputs already exist")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    static_enhanced_path = processed_dir / f"patient_static_enhanced.{args.format}"
    ts_enhanced_path = processed_dir / f"patient_timeseries_enhanced.{args.format}"

    # Step 1: Build enhanced tables
    if not (args.skip_build and static_enhanced_path.exists() and ts_enhanced_path.exists()):
        print("\n>>> Step 1/3: Building enhanced analysis tables from DuckDB...")
        build_enhanced_analysis_tables(
            db_path=db_path,
            output_dir=processed_dir,
            n_hours=args.hours,
            output_format=args.format,
        )
    else:
        print("\n>>> Step 1/3: Skipping build (files exist)")

    # Step 2: Generate split subtype labels
    print("\n>>> Step 2/3: Generating split subtype labels...")
    patient_info = build_subtype_labels(
        static_path=static_enhanced_path,
        timeseries_path=ts_enhanced_path,
        output_dir=processed_dir,
        output_format=args.format,
    )

    # Step 3: Reshape time-series to 3D tensor and save fused artifacts
    print("\n>>> Step 3/3: Reshaping to 3D tensor and saving fused artifacts...")
    if args.format == "parquet":
        ts_df = pd.read_parquet(ts_enhanced_path)
    else:
        ts_df = pd.read_csv(ts_enhanced_path)

    meta_cols = {"stay_id", "subject_id", "hr", "grid_time"}
    feature_cols = [c for c in ts_df.columns if c not in meta_cols]

    time_series_3d = reshape_to_3d(ts_df, feature_cols)
    print(f"  3D tensor shape: {time_series_3d.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Save tensor
    np.save(processed_dir / "time_series_enhanced.npy", time_series_3d)

    # Save patient_info (rename to match existing pipeline expectations)
    patient_info_out = processed_dir / "patient_info_enhanced.csv"
    patient_info.to_csv(patient_info_out, index=False)

    # Also save a feature-name manifest for downstream loaders
    manifest = {
        "feature_names": feature_cols,
        "n_patients": int(time_series_3d.shape[0]),
        "n_timesteps": int(time_series_3d.shape[1]),
        "n_features": int(time_series_3d.shape[2]),
        "subtype_columns": [
            "gold_mals_label",
            "gold_immunoparalysis_label",
            "proxy_immune_state",
            "proxy_immune_state_label",
            "proxy_clinical_phenotype",
            "proxy_clinical_phenotype_label",
            "proxy_trajectory_phenotype",
            "proxy_trajectory_phenotype_label",
            "proxy_fluid_strategy",
            "proxy_fluid_strategy_label",
            "score_mals",
            "score_immunoparalysis",
            "score_alpha",
            "score_beta",
            "score_gamma",
            "score_delta",
            "score_trajectory_a",
            "score_trajectory_b",
            "score_trajectory_c",
            "score_trajectory_d",
            "score_restrictive_fluid_benefit",
            "score_resuscitation_fluid_benefit",
        ],
        "subtype_artifacts": {
            "targets_npz": "sepsis_multitask_targets.npz",
            "schema_json": "sepsis_multitask_schema.json",
        },
        "outcome_columns": [
            "mortality_28d",
            "is_sepsis3",
            "los_icu_days",
        ],
    }
    with open(processed_dir / "enhanced_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Saved fused artifacts to {processed_dir}:")
    print(f"    - time_series_enhanced.npy")
    print(f"    - patient_info_enhanced.csv")
    print(f"    - patient_static_with_subtypes.{args.format}")
    print(f"    - sepsis_multitask_targets.npz")
    print(f"    - sepsis_multitask_schema.json")
    print(f"    - enhanced_manifest.json")

    print("\n" + "=" * 65)
    print("NEXT STEPS:")
    print("  1. Run the existing pipeline on enhanced data:")
    print(f"     python src/main.py --source mimic --processed-dir {processed_dir} --tag mimic_enhanced")
    print("  2. Or train the multi-task S5 model using the new subtype labels.")
    print("=" * 65)


if __name__ == "__main__":
    main()
