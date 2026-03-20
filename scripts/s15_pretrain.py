#!/usr/bin/env python3
"""
s15_pretrain.py - S1.5 pretraining entry point.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s15_pretrain.py
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s15_pretrain.py --epochs 30 --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device(pref="auto"):
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
    p = argparse.ArgumentParser(description="S1.5 Contrastive Pretraining")
    p.add_argument("--config", default="config/s15_config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg.get("runtime", {}).get("log_level", "INFO")),
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )

    device = args.device or get_device(cfg.get("runtime", {}).get("device", "auto"))
    enc = cfg["encoder"]
    con = cfg["contrastive"]
    pt = cfg["pretraining"]

    from s15.pretrain_contrastive import pretrain_contrastive

    pretrain_contrastive(
        s0_dir=PROJECT_ROOT / cfg["paths"]["s0_dir"],
        output_dir=PROJECT_ROOT / cfg["paths"]["s15_dir"],
        n_features=enc["n_features"], d_model=enc["d_model"],
        n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        d_ff=enc["d_ff"], dropout=enc["dropout"],
        view_len=con["view_len"], mask_ratio=con["mask_ratio"],
        temperature=con["temperature"], proj_dim=con["proj_dim"],
        max_lambda=con["max_lambda"], warmup_epochs=con["warmup_epochs"],
        epochs=args.epochs or pt["epochs"], batch_size=pt["batch_size"],
        lr=args.lr or pt["lr"], weight_decay=pt["weight_decay"],
        patience=pt["patience"], grad_clip=pt["grad_clip"],
        device=device, seed=pt["seed"],
    )


if __name__ == "__main__":
    main()
