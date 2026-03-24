#!/usr/bin/env python3
"""s4_train_treatment_aware.py - Train Stage 4 treatment-aware classifier."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s4.treatment_aware_model import train_treatment_aware_classifier


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def get_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 4 treatment-aware classifier")
    parser.add_argument("--config", default="config/s4_config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--s0-dir", default=None)
    parser.add_argument("--treatment-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--note-embeddings", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    with open(_resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    report = train_treatment_aware_classifier(
        s0_dir=_resolve(args.s0_dir or cfg["paths"]["s0_dir"]),
        treatment_dir=_resolve(args.treatment_dir or cfg["paths"]["treatment_dir"]),
        output_dir=_resolve(args.output_dir or cfg["paths"]["output_dir"]),
        pretrained_checkpoint=_resolve(args.pretrained_checkpoint or cfg["paths"].get("pretrained_checkpoint")),
        note_embeddings_path=_resolve(args.note_embeddings or cfg["paths"].get("note_embeddings")),
        label_col=cfg["training"].get("label_col", "mortality_inhospital"),
        batch_size=int(cfg["training"].get("batch_size", 128)),
        epochs=int(cfg["training"].get("epochs", 12)),
        lr_encoder=float(cfg["training"].get("lr_encoder", 2.0e-4)),
        lr_head=float(cfg["training"].get("lr_head", 1.0e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-4)),
        patience=int(cfg["training"].get("patience", 4)),
        freeze_base_epochs=int(cfg["training"].get("freeze_base_epochs", 0)),
        grad_clip=float(cfg["training"].get("grad_clip", 1.0)),
        threshold_metric=cfg["training"].get("threshold_metric", "balanced_accuracy"),
        monitor_metric=cfg["training"].get("monitor_metric", "auroc"),
        seed=int(cfg["training"].get("seed", 42)),
        device=get_device(args.device or cfg.get("runtime", {}).get("device", "auto")),
        d_model=int(cfg["model"].get("d_model", 128)),
        n_heads=int(cfg["model"].get("n_heads", 4)),
        n_layers=int(cfg["model"].get("n_layers", 2)),
        d_ff=int(cfg["model"].get("d_ff", 256)),
        dropout=float(cfg["model"].get("dropout", 0.2)),
        treatment_layers=int(cfg["model"].get("treatment_layers", 1)),
        head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 128)),
        head_dropout=float(cfg["model"].get("head_dropout", 0.2)),
    )
    logging.getLogger("s4.train").info(
        "Saved treatment-aware model. Test AUROC=%s",
        report["splits"]["test"].get("auroc"),
    )


if __name__ == "__main__":
    main()
