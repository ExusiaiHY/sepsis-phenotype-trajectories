#!/usr/bin/env python3
"""
s15_train_classifier.py - Train a supervised mortality classifier on S1.5 embeddings.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_train_classifier.py
  ./.venv/bin/python scripts/s15_train_classifier.py --config config/s15_trainval_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.classification_eval import train_mortality_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised mortality classifier on S1.5 embeddings")
    parser.add_argument("--config", default="config/s15_config.yaml")
    parser.add_argument("--embeddings", default=None,
                        help="Optional override for embeddings .npy path")
    parser.add_argument("--label-col", default="mortality_inhospital")
    parser.add_argument("--class-weight", default="balanced",
                        help="Classifier class_weight; use 'none' to disable")
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threshold-metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "accuracy", "f1"])
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

    if not embeddings_path.exists():
        raise FileNotFoundError(f"No embeddings found at {embeddings_path}. Run s15_extract.py first.")

    report = train_mortality_classifier(
        embeddings=np.load(embeddings_path),
        static_path=s0_dir / "static.csv",
        splits_path=s0_dir / "splits.json",
        output_dir=s15_dir,
        label_col=args.label_col,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        c_value=args.c_value,
        max_iter=args.max_iter,
        threshold_metric=args.threshold_metric,
    )

    for split_name in ("val", "test"):
        metrics = report["splits"][split_name]
        logging.getLogger("s15.train_classifier").info(
            "%s accuracy=%.4f balanced_acc=%.4f f1=%.4f auroc=%s",
            split_name,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["auroc"],
        )


if __name__ == "__main__":
    main()
