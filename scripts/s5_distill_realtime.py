#!/usr/bin/env python3
"""s5_distill_realtime.py - Distill and evaluate the Stage 5 realtime student."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s5.realtime_model import distill_realtime_student


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
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Distill Stage 5 realtime student model")
    parser.add_argument("--config", default="config/s5_config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--s0-dir", default=None)
    parser.add_argument("--treatment-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--teacher-embeddings", default=None)
    parser.add_argument("--teacher-probabilities", default=None)
    parser.add_argument("--note-embeddings", default=None)
    parser.add_argument("--output-dir", default=None)
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

    report = distill_realtime_student(
        s0_dir=_resolve(args.s0_dir or cfg["paths"]["s0_dir"]),
        treatment_dir=_resolve(args.treatment_dir or cfg["paths"]["treatment_dir"]),
        output_dir=_resolve(args.output_dir or cfg["paths"]["output_dir"]),
        init_checkpoint_path=_resolve(
            args.init_checkpoint
            or cfg["paths"].get("init_checkpoint")
            or cfg["paths"].get("base_student_artifact")
        ),
        teacher_embeddings_path=_resolve(args.teacher_embeddings or cfg["paths"].get("teacher_embeddings")),
        teacher_probabilities_path=_resolve(args.teacher_probabilities or cfg["paths"].get("teacher_probabilities")),
        note_embeddings_path=_resolve(args.note_embeddings or cfg["paths"].get("note_embeddings")),
        label_col=cfg["training"].get("label_col", "mortality_inhospital"),
        batch_size=int(cfg["training"].get("batch_size", 128)),
        epochs=int(cfg["training"].get("epochs", 10)),
        lr=float(cfg["training"].get("lr", 1.0e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1.0e-4)),
        patience=int(cfg["training"].get("patience", 4)),
        bce_weight=float(cfg["training"].get("bce_weight", 1.0)),
        pos_weight=float(cfg["training"]["pos_weight"]) if cfg["training"].get("pos_weight") is not None else None,
        horizon_augmentation_min_h=int(cfg["training"].get("horizon_augmentation_min_h", 0)),
        distill_weight=float(cfg["training"].get("distill_weight", 1.0)),
        distill_cosine_weight=float(cfg["training"].get("distill_cosine_weight", 0.0)),
        distill_prob_weight=float(cfg["training"].get("distill_prob_weight", 0.0)),
        distill_temperature=float(cfg["training"].get("distill_temperature", 1.0)),
        apply_temperature_scaling=bool(cfg["training"].get("apply_temperature_scaling", False)),
        init_checkpoint_strict=bool(cfg["training"].get("init_checkpoint_strict", True)),
        threshold_metric=cfg["training"].get("threshold_metric", "balanced_accuracy"),
        target_positive_rate=cfg["training"].get("target_positive_rate"),
        seed=int(cfg["training"].get("seed", 42)),
        device=get_device(args.device or cfg.get("runtime", {}).get("device", "auto")),
        student_arch=cfg["model"].get("student_arch", "transformer"),
        student_d_model=int(cfg["model"].get("student_d_model", 64)),
        teacher_dim=int(cfg["model"].get("teacher_dim", 128)),
        n_heads=int(cfg["model"].get("n_heads", 4)),
        n_layers=int(cfg["model"].get("n_layers", 1)),
        d_ff=int(cfg["model"].get("d_ff", 128)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        treatment_layers=int(cfg["model"].get("treatment_layers", 1)),
        head_hidden_dim=int(cfg["model"].get("head_hidden_dim", 64)),
        head_dropout=float(cfg["model"].get("head_dropout", 0.1)),
        tcn_kernel_size=int(cfg["model"].get("tcn_kernel_size", 3)),
        tcn_dilations=tuple(int(v) for v in cfg["model"].get("tcn_dilations", [1, 2, 4, 8])),
    )
    logging.getLogger("s5.student").info(
        "Saved realtime student. Test AUROC=%s latency_ms=%s",
        report["splits"]["test"].get("auroc"),
        report["deployment"]["cpu_latency_ms_per_sample"],
    )


if __name__ == "__main__":
    main()
