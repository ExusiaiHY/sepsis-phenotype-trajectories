#!/usr/bin/env python3
"""
s15_train_stacking_classifier.py - Train leakage-aware OOF stacking classifier.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_train_stacking_classifier.py --config config/s15_trainval_config.yaml
  ./.venv/bin/python scripts/s15_train_stacking_classifier.py --threshold-metric balanced_accuracy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.stacking_classifier import train_stacking_mortality_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train OOF stacking downstream mortality classifier")
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--embeddings", default=None, help="Optional override for embeddings .npy path")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    parser.add_argument("--label-col", default="mortality_inhospital")
    parser.add_argument(
        "--threshold-metric",
        default="accuracy",
        choices=["accuracy", "balanced_accuracy", "f1"],
    )
    parser.add_argument("--n-splits", type=int, default=5)
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
    embeddings_path = Path(args.embeddings) if args.embeddings else (s15_dir / "embeddings_s15.npy")
    output_dir = Path(args.output_dir) if args.output_dir else (s15_dir / f"stacking_{args.threshold_metric}")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"No embeddings found at {embeddings_path}. Run s15_extract.py first.")

    report = train_stacking_mortality_classifier(
        s0_dir=s0_dir,
        splits_path=s0_dir / "splits.json",
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        label_col=args.label_col,
        threshold_metric=args.threshold_metric,
        n_splits=args.n_splits,
    )

    logger = logging.getLogger("scripts.s15_train_stacking_classifier")
    for metric_name in ("accuracy", "balanced_accuracy", "f1"):
        point = report["operating_points"][metric_name]["test"]
        logger.info(
            "%s operating point -> test accuracy=%.4f balanced_acc=%.4f f1=%.4f recall=%.4f auroc=%s",
            metric_name,
            point["accuracy"],
            point["balanced_accuracy"],
            point["f1"],
            point["recall"],
            point["auroc"],
        )


if __name__ == "__main__":
    main()
