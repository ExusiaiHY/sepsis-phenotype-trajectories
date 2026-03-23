#!/usr/bin/env python3
"""
s15_finetune_supervised.py - End-to-end supervised fine-tuning for mortality prediction.

How to run:
  cd project
  ./.venv/bin/python scripts/s15_finetune_supervised.py
  ./.venv/bin/python scripts/s15_finetune_supervised.py --aux-data-dir data/s19_bridge
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.finetune_supervised import train_end_to_end_classifier


def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end supervised fine-tuning on S1.5 encoder")
    parser.add_argument("--config", default="config/s15_trainval_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label-col", default="mortality_inhospital")
    parser.add_argument("--aux-data-dir", default="data/s19_bridge")
    parser.add_argument("--aux-label-col", default="sepsis_label")
    parser.add_argument("--disable-aux", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--aux-epochs", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=2.0e-4)
    parser.add_argument("--lr-head", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold-metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "accuracy", "f1"])
    parser.add_argument("--monitor-metric", default="auroc",
                        choices=["accuracy", "balanced_accuracy", "f1", "auroc"])
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--head-dropout", type=float, default=0.3)
    parser.add_argument("--device", default=None)
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
    output_dir = Path(args.output_dir) if args.output_dir else (s15_dir / "finetune_supervised")
    checkpoint = Path(args.checkpoint) if args.checkpoint else (s15_dir / "checkpoints" / "pretrain_best.pt")
    aux_dir = None if args.disable_aux else (PROJECT_ROOT / args.aux_data_dir)
    if aux_dir is not None and not aux_dir.exists():
        logging.getLogger("scripts.s15_finetune_supervised").warning(
            "Auxiliary data dir %s does not exist; continuing without auxiliary stage.",
            aux_dir,
        )
        aux_dir = None

    report = train_end_to_end_classifier(
        s0_dir=s0_dir,
        output_dir=output_dir,
        pretrained_checkpoint=checkpoint if checkpoint.exists() else None,
        label_col=args.label_col,
        aux_data_dir=aux_dir,
        aux_label_col=args.aux_label_col,
        batch_size=args.batch_size,
        epochs=args.epochs,
        aux_epochs=args.aux_epochs,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        patience=args.patience,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        grad_clip=args.grad_clip,
        threshold_metric=args.threshold_metric,
        monitor_metric=args.monitor_metric,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        device=args.device or get_device(cfg.get("runtime", {}).get("device", "auto")),
        seed=cfg.get("pretraining", {}).get("seed", 42),
    )

    logger = logging.getLogger("scripts.s15_finetune_supervised")
    metrics = report["main_task"]["splits"]["test"]
    logger.info(
        "main test accuracy=%.4f balanced_acc=%.4f recall=%.4f f1=%.4f auroc=%s",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["recall"],
        metrics["f1"],
        metrics["auroc"],
    )
    if "auxiliary_stage" in report:
        aux_metrics = report["auxiliary_stage"]["splits"]["test"]
        logger.info(
            "aux test accuracy=%.4f balanced_acc=%.4f recall=%.4f auroc=%s",
            aux_metrics["accuracy"],
            aux_metrics["balanced_accuracy"],
            aux_metrics["recall"],
            aux_metrics["auroc"],
        )


if __name__ == "__main__":
    main()
