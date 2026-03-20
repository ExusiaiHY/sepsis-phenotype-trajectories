#!/usr/bin/env python3
"""
s0_prepare.py - S0 data pipeline entry point.

Purpose:
  Run the full S0 pipeline: extract → preprocess → split → manifest.

How to run:
  cd project
  python3.14 scripts/s0_prepare.py
  python3.14 scripts/s0_prepare.py --config config/s0_config.yaml
  python3.14 scripts/s0_prepare.py --split-method random

Expected output artifacts:
  data/s0/raw_aligned/*.npy       (6 files)
  data/s0/processed/*.npy         (6 files + preprocess_stats.json)
  data/s0/static.csv
  data/s0/feature_dict.json
  data/s0/splits.json
  data/s0/data_manifest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s0.schema import schema_to_feature_dict
from s0.physionet2012_extractor import extract_physionet2012
from s0.preprocessor import preprocess_raw_aligned
from s0.splits import build_splits
from s0.manifest import generate_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="S0 Data Pipeline")
    parser.add_argument("--config", type=str, default="config/s0_config.yaml")
    parser.add_argument("--split-method", type=str, default=None,
                        choices=["cross_center", "random"])
    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main():
    args = parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.split_method:
        config["splits"]["method"] = args.split_method

    setup_logging(config.get("runtime", {}).get("log_level", "INFO"))
    logger = logging.getLogger("s0_prepare")

    logger.info("=" * 60)
    logger.info("S0 Data Pipeline — PhysioNet 2012")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    data_dir = PROJECT_ROOT / config["data"]["data_dir"]
    output_dir = PROJECT_ROOT / config["data"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    n_hours = config["data"]["n_hours"]
    pn_cfg = config["data"]["physionet2012"]

    # ============================================================
    # Step 1: Extract + align
    # ============================================================
    logger.info("[Step 1/5] Extracting PhysioNet 2012 data...")
    extraction_stats = extract_physionet2012(
        data_dir=data_dir,
        output_dir=output_dir,
        n_hours=n_hours,
        sets=pn_cfg["sets"],
        center_a_sets=set(pn_cfg["center_a_sets"]),
        center_b_sets=set(pn_cfg["center_b_sets"]),
        min_measurements=pn_cfg.get("min_measurements", 1),
    )

    # ============================================================
    # Step 2: Save feature dictionary
    # ============================================================
    logger.info("[Step 2/5] Saving feature dictionary...")
    feat_dict = schema_to_feature_dict()
    with open(output_dir / "feature_dict.json", "w") as f:
        json.dump(feat_dict, f, indent=2)

    # ============================================================
    # Step 3: Preprocess
    # ============================================================
    logger.info("[Step 3/5] Preprocessing...")
    prep_cfg = config["preprocess"]
    preprocess_stats = preprocess_raw_aligned(
        input_dir=output_dir / "raw_aligned",
        output_dir=output_dir / "processed",
        max_forward_fill_hours=prep_cfg["max_forward_fill_hours"],
        outlier_sigma=prep_cfg["outlier_sigma"],
        normalization=prep_cfg["normalization"],
    )

    # ============================================================
    # Step 4: Split
    # ============================================================
    logger.info("[Step 4/5] Building splits...")
    split_cfg = config["splits"]
    build_splits(
        static_path=output_dir / "static.csv",
        output_path=output_dir / "splits.json",
        method=split_cfg["method"],
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        seed=split_cfg.get("seed", 42),
        stratify_by=split_cfg.get("stratify_by", "mortality_inhospital"),
    )

    # ============================================================
    # Step 5: Manifest
    # ============================================================
    logger.info("[Step 5/5] Generating manifest...")
    generate_manifest(output_dir, extraction_stats, preprocess_stats, config)

    # ============================================================
    # Summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("S0 pipeline complete!")
    logger.info(f"  Patients: {extraction_stats['n_patients']}")
    logger.info(f"  Hours: {n_hours}")
    logger.info(f"  Continuous features: {extraction_stats['n_continuous']}")
    logger.info(f"  Intervention channels: {extraction_stats['n_interventions']}")
    logger.info(f"  Proxy channels: {extraction_stats['n_proxy']}")
    logger.info(f"  Missing rate (continuous): {extraction_stats['overall_continuous_missing_rate']:.1%}")
    logger.info(f"  Outcome labels: {'outcomes file' if extraction_stats['has_outcome_labels'] else 'proxy'}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
