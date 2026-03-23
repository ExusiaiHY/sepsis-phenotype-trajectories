#!/usr/bin/env python3
"""
s15_validate_stacking_classifier.py - Bootstrap, calibration, and explanation for stacking model.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_validate_stacking_classifier.py --model-dir data/s15_trainval/stacking_accuracy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.stacking_validation import validate_stacking_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Validate OOF stacking mortality classifier")
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--model-dir", required=True, help="Directory containing stacking_mortality_classifier.pkl")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=20)
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
    model_dir = Path(args.model_dir)
    model_path = model_dir / "stacking_mortality_classifier.pkl"

    report = validate_stacking_classifier(
        model_path=model_path,
        s0_dir=s0_dir,
        splits_path=s0_dir / "splits.json",
        output_dir=model_dir,
        embeddings_path=s15_dir / "embeddings_s15.npy",
        n_bootstrap=args.bootstrap,
        n_bins=args.n_bins,
        permutation_repeats=args.permutation_repeats,
    )

    logger = logging.getLogger("scripts.s15_validate_stacking_classifier")
    logger.info(
        "test accuracy=%.4f balanced_acc=%.4f f1=%.4f auroc=%.4f",
        report["splits"]["test"]["accuracy"],
        report["splits"]["test"]["balanced_accuracy"],
        report["splits"]["test"]["f1"],
        report["splits"]["test"]["auroc"],
    )
    logger.info(
        "test calibration brier=%.4f ece=%.4f",
        report["test_calibration"]["brier"],
        report["test_calibration"]["ece"],
    )


if __name__ == "__main__":
    main()
