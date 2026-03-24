#!/usr/bin/env python3
"""
s15_train_calibrated_stacking.py - Train calibration-aware stacking classifier.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    python3 scripts/s15_train_calibrated_stacking.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.calibrated_stacking import train_calibrated_stacking


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train calibration-aware stacking mortality classifier"
    )
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prior-rate", type=float, default=0.142)
    parser.add_argument("--no-posthoc", action="store_true",
                        help="Skip post-hoc Platt scaling")
    parser.add_argument("--original-specs", action="store_true",
                        help="Use original base specs instead of calibrated ones")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg.get("runtime", {}).get("log_level", "INFO")),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    output_dir = Path(args.output_dir) if args.output_dir else s15_dir / "calibrated_stacking"

    report = train_calibrated_stacking(
        s0_dir=s0_dir,
        splits_path=s0_dir / "splits.json",
        output_dir=output_dir,
        embeddings_path=s15_dir / "embeddings_s15.npy",
        prior_rate=args.prior_rate,
        use_calibrated_specs=not args.original_specs,
        apply_posthoc_calibration=not args.no_posthoc,
    )

    logger = logging.getLogger("scripts.s15_train_calibrated_stacking")
    cal = report["calibrated_test_metrics"]
    cls = report["classification_at_threshold"]
    logger.info("=" * 60)
    logger.info("CALIBRATED STACKING RESULTS")
    logger.info("=" * 60)
    logger.info("  Brier=%.4f  ECE=%.4f  AUROC=%.4f",
                cal["brier"], cal["ece"], cal["auroc"] or 0.0)
    logger.info("  Threshold=%.4f  Recall=%.4f  Precision=%.4f  F1=%.4f",
                report["threshold"], cls["recall"], cls["precision"], cls["f1"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
