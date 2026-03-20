#!/usr/bin/env python3
"""
s15_extract.py - Extract S1.5 embeddings from pretrained contrastive encoder.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s15_extract.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s1.encoder import ICUTransformerEncoder


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


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger("s15.extract")

    with open(PROJECT_ROOT / "config" / "s15_config.yaml") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    device = get_device(cfg.get("runtime", {}).get("device", "auto"))

    # Load checkpoint
    ckpt_path = s15_dir / "checkpoints" / "pretrain_best.pt"
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

    # Load data
    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")
    n_patients = continuous.shape[0]

    logger.info(f"Extracting S1.5 embeddings for {n_patients} patients...")

    all_emb = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, n_patients, batch_size):
            end = min(start + batch_size, n_patients)
            x = torch.from_numpy(continuous[start:end]).float().to(device)
            m = torch.from_numpy(masks[start:end]).float().to(device)
            emb = encoder(x, m)
            all_emb.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    out_path = s15_dir / "embeddings_s15.npy"
    np.save(out_path, embeddings)
    logger.info(f"S1.5 embeddings saved: {embeddings.shape} → {out_path}")


if __name__ == "__main__":
    main()
