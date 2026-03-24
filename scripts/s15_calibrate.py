#!/usr/bin/env python3
"""
s15_calibrate.py - Run post-hoc calibration on the stacking mortality classifier.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    python3 scripts/s15_calibrate.py \
      --model-dir data/s15_trainval/stacking_accuracy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.calibration import run_calibration_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration for stacking mortality classifier"
    )
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing stacking_mortality_classifier.pkl",
    )
    parser.add_argument("--prior-rate", type=float, default=0.142,
                        help="Clinical prior mortality rate (default: 0.142)")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <model-dir>/calibration)")
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
    logger = logging.getLogger("scripts.s15_calibrate")

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    model_path = model_dir / "stacking_mortality_classifier.pkl"

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "calibration"

    report = run_calibration_pipeline(
        model_path=model_path,
        s0_dir=s0_dir,
        splits_path=s0_dir / "splits.json",
        embeddings_path=s15_dir / "embeddings_s15.npy",
        output_dir=output_dir,
        prior_rate=args.prior_rate,
        n_bins=args.n_bins,
    )

    logger.info("=" * 60)
    logger.info("CALIBRATION COMPARISON RESULTS")
    logger.info("=" * 60)
    for row in report["ranking"]:
        logger.info(
            "  %-30s  Brier=%.4f  ECE=%.4f  AUROC=%.4f  Recall=%.4f",
            row["method"],
            row["brier"],
            row["ece"],
            row["auroc"],
            row["recall"],
        )
    logger.info("-" * 60)
    logger.info("  RECOMMENDED: %s", report["recommendation"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
