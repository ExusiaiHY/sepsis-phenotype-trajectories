#!/usr/bin/env python3
"""
s1_pretrain.py - Entry point for S1 self-supervised pretraining.

How to run:
  cd project
  KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s1_pretrain.py
  KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s1_pretrain.py --epochs 30 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
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
    p = argparse.ArgumentParser(description="S1 Pretraining")
    p.add_argument("--config", type=str, default="config/s1_config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, config.get("runtime", {}).get("log_level", "INFO")),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    enc = config["encoder"]
    pt = config["pretraining"]

    device = args.device or get_device(config.get("runtime", {}).get("device", "auto"))
    epochs = args.epochs or pt["epochs"]
    lr = args.lr or pt["lr"]
    bs = args.batch_size or pt["batch_size"]

    s0_dir = PROJECT_ROOT / config["paths"]["s0_dir"]
    s1_dir = PROJECT_ROOT / config["paths"]["s1_dir"]

    from s1.pretrain import pretrain

    pretrain(
        s0_dir=s0_dir,
        output_dir=s1_dir,
        n_features=enc["n_features"],
        d_model=enc["d_model"],
        n_heads=enc["n_heads"],
        n_layers=enc["n_layers"],
        d_ff=enc["d_ff"],
        dropout=enc["dropout"],
        mask_ratio=pt["mask_ratio"],
        epochs=epochs,
        batch_size=bs,
        lr=lr,
        weight_decay=pt["weight_decay"],
        patience=pt["patience"],
        grad_clip=pt["grad_clip"],
        device=device,
        seed=pt["seed"],
    )


if __name__ == "__main__":
    main()
