"""
s6_run_full_optimization.py - Unified pipeline for model optimization.

Runs the complete S6 optimization pipeline:
  1. Compute informative missingness features from S0 masks
  2. Run causal phenotyping pipeline (CATE estimation + organ scoring)
  3. Assign mechanism-based phenotype names
  4. Generate comprehensive validation report

Usage:
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/s6_run_full_optimization.py

Inputs:
  - data/s0/processed/     (continuous, masks, proxy_indicators)
  - data/s0/static.csv     (patient metadata with mortality)
  - data/s0/splits.json    (train/val/test splits)
  - data/s2/window_labels.npy (KMeans cluster assignments)

Outputs:
  - data/s6/missingness_summary.json
  - data/s6/missingness_enhanced.npy
  - data/s6/phenotype_assignments.csv
  - data/s6/organ_scores.csv
  - data/s6/cate_scores.npy
  - data/s6/causal_phenotyping_report.json
  - data/s6/optimization_report.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6_optimization.missingness_encoder import (
    compute_missingness_features,
    compute_patient_missingness_summary,
)
from s6_optimization.baseline_comparison import generate_baseline_comparison_report
from s6_optimization.causal_phenotyping import run_causal_phenotyping_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("s6.orchestrator")


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("S6 OPTIMIZATION PIPELINE — Model Enhancement")
    logger.info("=" * 70)

    # Load config
    config_path = PROJECT_ROOT / "config" / "s6_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s2_dir = PROJECT_ROOT / cfg["paths"]["s2_dir"]
    output_dir = PROJECT_ROOT / cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline": "s6_full_optimization",
        "config": cfg,
        "stages": {},
    }

    # ================================================================
    # STAGE 1: Informative Missingness Feature Engineering
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 1: Informative Missingness Features")
    logger.info("=" * 60)
    t1 = time.time()

    masks = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
    masks_array = np.array(masks)  # load into memory for computation

    # Compute missingness summary
    miss_summary = compute_patient_missingness_summary(masks_array)
    miss_summary_path = output_dir / "missingness_summary.json"
    with open(miss_summary_path, "w") as f:
        json.dump(miss_summary, f, indent=2)
    logger.info("Missingness summary saved: %s", miss_summary_path)

    # Compute enhanced missingness features
    enhanced_masks = compute_missingness_features(
        masks_array,
        gap_window=cfg["missingness"]["gap_window"],
    )
    enhanced_path = output_dir / "missingness_enhanced.npy"
    np.save(enhanced_path, enhanced_masks)
    logger.info("Enhanced mask tensor: %s -> %s", masks_array.shape, enhanced_masks.shape)

    del masks_array, enhanced_masks  # free memory

    report["stages"]["missingness"] = {
        "duration_sec": round(time.time() - t1, 1),
        "summary": miss_summary,
        "artifacts": {
            "summary": str(miss_summary_path),
            "enhanced_masks": str(enhanced_path),
        },
    }

    # ================================================================
    # STAGE 2: Causal Phenotyping Pipeline
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Causal Phenotyping (CATE + Organ Scores + Naming)")
    logger.info("=" * 60)
    t2 = time.time()

    causal_report = run_causal_phenotyping_pipeline(
        s0_dir=s0_dir,
        s2_dir=s2_dir,
        output_dir=output_dir,
        splits_path=s0_dir / "splits.json",
        causal_method=cfg["causal"]["method"],
        treatment_horizon=cfg["causal"]["treatment_horizon"],
        organ_horizon=cfg["organ_scoring"]["horizon"],
        phenotype_config=cfg["phenotype_naming"],
    )

    report["stages"]["causal_phenotyping"] = {
        "duration_sec": round(time.time() - t2, 1),
        "report": causal_report,
    }

    # ================================================================
    # STAGE 3: Final Summary & Validation
    # ================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Baseline Comparison")
    logger.info("=" * 60)
    t3 = time.time()

    comparison_report = generate_baseline_comparison_report(
        static_path=s0_dir / "static.csv",
        window_labels_path=s2_dir / "window_labels.npy",
        phenotype_assignments_path=output_dir / "phenotype_assignments.csv",
        output_path=output_dir / "baseline_comparison.json",
        min_group_size=50,
    )
    report["stages"]["baseline_comparison"] = {
        "duration_sec": round(time.time() - t3, 1),
        "report": comparison_report,
    }

    logger.info(
        "Mortality range: baseline=%.4f -> optimized=%.4f",
        comparison_report["baseline"]["supported_mortality_range"],
        comparison_report["optimized"]["supported_mortality_range"],
    )
    logger.info(
        "Center L1: baseline=%.4f -> optimized=%.4f",
        comparison_report["baseline"]["center_distribution_l1"],
        comparison_report["optimized"]["center_distribution_l1"],
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Final Summary")
    logger.info("=" * 60)

    total_time = round(time.time() - t_start, 1)
    report["total_duration_sec"] = total_time

    # Print summary
    logger.info("")
    logger.info("PHENOTYPE DISTRIBUTION:")
    for key, val in causal_report.get("phenotype_validation", {}).items():
        name = val.get("phenotype_name", key)
        logger.info(
            "  %-45s n=%5d (%.1f%%)  mortality=%.1f%%  SOFA=%.1f  CATE=%.4f",
            key, val["n"], 100 * val["fraction"],
            100 * val["mortality_rate"],
            val["mean_sofa"], val["mean_cate"],
        )

    logger.info("")
    logger.info("CATE SUMMARY:")
    cate_s = causal_report.get("cate_summary", {})
    logger.info("  Mean=%.4f  Std=%.4f  Q10=%.4f  Q50=%.4f  Q90=%.4f",
                cate_s.get("mean", 0), cate_s.get("std", 0),
                cate_s.get("q10", 0), cate_s.get("q50", 0), cate_s.get("q90", 0))

    # Save final report
    report_path = output_dir / "optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("S6 OPTIMIZATION COMPLETE — Total time: %.1f seconds", total_time)
    logger.info("Report: %s", report_path)
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    main()
