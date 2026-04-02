#!/usr/bin/env python3
"""
s6_train_multitask_student.py - Train the multi-task S6 realtime student.

Usage:
  export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE
  python scripts/s6_train_multitask_student.py \
      --data-dir data/processed_mimic_enhanced \
      --output-dir data/s6_multitask_mimic \
      --epochs 10 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6.multitask_model import train_multitask_student


def get_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train S6 multi-task realtime student")
    parser.add_argument("--data-dir", type=str, default="data/processed_mimic_enhanced")
    parser.add_argument("--output-dir", type=str, default="data/s6_multitask_mimic")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--student-arch", type=str, default="transformer")
    parser.add_argument("--student-d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-immune", type=float, default=1.0)
    parser.add_argument("--lambda-organ", type=float, default=1.0)
    parser.add_argument("--lambda-fluid", type=float, default=1.0)
    parser.add_argument("--lambda-mortality", type=float, default=1.0)
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--immune-boost", type=float, default=1.0)
    parser.add_argument("--organ-boost", type=float, default=1.0)
    parser.add_argument("--fluid-boost", type=float, default=1.0)
    parser.add_argument("--phase1-epochs", type=int, default=0)
    parser.add_argument("--init-strict", action="store_true")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    init_checkpoint = PROJECT_ROOT / args.init_checkpoint if args.init_checkpoint else None

    report = train_multitask_student(
        data_dir=data_dir,
        output_dir=output_dir,
        init_checkpoint_path=init_checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        lambda_mortality=args.lambda_mortality,
        lambda_immune=args.lambda_immune,
        lambda_organ=args.lambda_organ,
        lambda_fluid=args.lambda_fluid,
        init_checkpoint_strict=args.init_strict,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        immune_boost=args.immune_boost,
        organ_boost=args.organ_boost,
        fluid_boost=args.fluid_boost,
        phase1_epochs=args.phase1_epochs,
        seed=args.seed,
        device=get_device(args.device),
        student_arch=args.student_arch,
        student_d_model=args.student_d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    logger = logging.getLogger("s6.train")
    logger.info("=" * 60)
    logger.info("S6 Multi-task training complete")
    logger.info("Test mortality AUROC: %s", report["splits"]["test"]["mortality"]["auroc"])
    logger.info("Test immune macro-F1: %s", report["splits"]["test"]["immune"]["macro_f1"])
    logger.info("Test organ macro-F1:  %s", report["splits"]["test"]["organ"]["macro_f1"])
    logger.info("Test fluid macro-F1:  %s", report["splits"]["test"]["fluid"]["macro_f1"])
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
