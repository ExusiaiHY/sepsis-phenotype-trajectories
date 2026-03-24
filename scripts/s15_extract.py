#!/usr/bin/env python3
"""
s15_extract.py - Extract S1.5 embeddings from pretrained contrastive encoder.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s15_extract.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s1.encoder import ICUTransformerEncoder


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def get_device(pref="auto"):
    if pref != "auto":
        return pref
    try:
        import torch as t
        if t.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract S1.5 embeddings from pretrained encoder")
    parser.add_argument("--config", default="config/s15_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger("s15.extract")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s0_dir = resolve_project_path(cfg["paths"]["s0_dir"])
    s15_dir = resolve_project_path(cfg["paths"]["s15_dir"])
    s15_dir.mkdir(parents=True, exist_ok=True)
    requested_device = args.device if args.device is not None else cfg.get("runtime", {}).get("device", "auto")
    device = get_device(requested_device)
    batch_size = int(args.batch_size or cfg.get("runtime", {}).get("batch_size", 128))

    # Load checkpoint
    ckpt_override = cfg["paths"].get("s15_checkpoint")
    ckpt_path = resolve_project_path(ckpt_override) if ckpt_override else (s15_dir / "checkpoints" / "pretrain_best.pt")
    if not ckpt_path.exists():
        logger.error(f"No checkpoint at {ckpt_path}. Run s15_pretrain.py first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    enc_cfg = ckpt["config"]

    encoder = ICUTransformerEncoder(
        n_features=enc_cfg["n_features"], d_model=enc_cfg["d_model"],
        n_heads=enc_cfg["n_heads"], n_layers=enc_cfg["n_layers"],
        d_ff=enc_cfg["d_ff"], dropout=0.0,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.train(False)
    d_model = int(enc_cfg["d_model"])

    # Load data
    continuous = np.load(s0_dir / "processed" / "continuous.npy", mmap_mode="r")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
    n_patients = continuous.shape[0]

    logger.info(f"Extracting S1.5 embeddings for {n_patients} patients...")

    embeddings = np.empty((n_patients, d_model), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_patients, batch_size):
            end = min(start + batch_size, n_patients)
            x_np = np.array(continuous[start:end], dtype=np.float32, copy=True)
            m_np = np.array(masks[start:end], dtype=np.float32, copy=True)
            x = torch.from_numpy(x_np).to(device)
            m = torch.from_numpy(m_np).to(device)
            emb = encoder(x, m)
            embeddings[start:end] = emb.cpu().numpy().astype(np.float32, copy=False)

    out_path = s15_dir / "embeddings_s15.npy"
    np.save(out_path, embeddings)
    logger.info(f"S1.5 embeddings saved: {embeddings.shape} → {out_path}")


if __name__ == "__main__":
    main()
