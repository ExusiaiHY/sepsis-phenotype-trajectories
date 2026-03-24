#!/usr/bin/env python3
"""
s2_extract_rolling.py - Extract rolling-window embeddings.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s2_extract_rolling.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def get_device(pref="auto"):
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
    parser = argparse.ArgumentParser(description="Extract rolling-window embeddings")
    parser.add_argument("--config", default="config/s2_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from s2light.rolling_embeddings import extract_rolling_embeddings

    requested_device = args.device if args.device is not None else cfg.get("runtime", {}).get("device", "auto")
    device = get_device(requested_device)
    batch_size = int(args.batch_size or cfg.get("runtime", {}).get("batch_size", 128))
    s0_dir = resolve_project_path(cfg["paths"]["s0_dir"])
    s15_encoder = resolve_project_path(cfg["paths"]["s15_encoder"])
    s2_dir = resolve_project_path(cfg["paths"]["s2_dir"])
    s2_dir.mkdir(parents=True, exist_ok=True)
    rw = cfg["rolling_windows"]

    emb, meta = extract_rolling_embeddings(
        s0_dir=s0_dir,
        encoder_ckpt=s15_encoder,
        output_path=s2_dir / "rolling_embeddings.npy",
        window_len=rw["window_len"],
        stride=rw["stride"],
        seq_len=rw["seq_len"],
        device=device,
        batch_size=batch_size,
    )

    with open(s2_dir / "rolling_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.getLogger("s2").info("Rolling extraction complete.")


if __name__ == "__main__":
    main()
