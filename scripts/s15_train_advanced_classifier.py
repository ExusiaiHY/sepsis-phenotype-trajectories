#!/usr/bin/env python3
"""
s15_train_advanced_classifier.py - Train stronger downstream mortality models.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_train_advanced_classifier.py --config config/s15_trainval_config.yaml
  ./.venv/bin/python scripts/s15_train_advanced_classifier.py --model-type hgb --feature-set stats_mask_proxy_static
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.advanced_classifier import train_advanced_mortality_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train advanced downstream mortality classifier")
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--embeddings", default=None,
                        help="Optional override for embeddings .npy path")
    parser.add_argument("--output-dir", default=None,
                        help="Optional override for output directory")
    parser.add_argument("--label-col", default="mortality_inhospital")
    parser.add_argument("--model-type", default="hgb",
                        choices=["logreg", "hgb", "hgb_ensemble"])
    parser.add_argument("--feature-set", default="stats_mask_proxy_static",
                        choices=["embeddings", "embeddings_static", "stats_mask_proxy_static", "fused_all"])
    parser.add_argument("--threshold-metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "accuracy", "f1"])
    parser.add_argument("--hgb-max-depth", type=int, default=5)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--hgb-max-iter", type=int, default=300)
    parser.add_argument("--ensemble-weight-step", type=float, default=0.05)
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
    output_dir = Path(args.output_dir) if args.output_dir else (s15_dir / "advanced_classifier")

    if args.model_type in {"logreg", "hgb_ensemble"} or args.feature_set in {"embeddings", "embeddings_static", "fused_all"}:
        if not embeddings_path.exists():
            raise FileNotFoundError(f"No embeddings found at {embeddings_path}. Run s15_extract.py first.")
        embeddings_arg = embeddings_path
    else:
        embeddings_arg = embeddings_path if embeddings_path.exists() else None

    report = train_advanced_mortality_classifier(
        s0_dir=s0_dir,
        splits_path=s0_dir / "splits.json",
        output_dir=output_dir,
        embeddings_path=embeddings_arg,
        label_col=args.label_col,
        model_type=args.model_type,
        feature_set=args.feature_set,
        threshold_metric=args.threshold_metric,
        hgb_max_depth=args.hgb_max_depth,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_max_iter=args.hgb_max_iter,
        ensemble_weight_step=args.ensemble_weight_step,
    )

    logger = logging.getLogger("s15.train_advanced_classifier")
    for split_name in ("val", "test"):
        metrics = report["splits"][split_name]
        logger.info(
            "%s accuracy=%.4f balanced_acc=%.4f f1=%.4f recall=%.4f auroc=%s",
            split_name,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["recall"],
            metrics["auroc"],
        )

    if "ensemble_selection" in report:
        logger.info(
            "ensemble fused_weight=%.2f stats_weight=%.2f threshold=%.2f",
            report["ensemble_selection"]["selected_fused_weight"],
            report["ensemble_selection"]["selected_stats_weight"],
            report["threshold_selection"]["selected_threshold"],
        )


if __name__ == "__main__":
    main()
