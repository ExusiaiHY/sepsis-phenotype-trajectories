#!/usr/bin/env python3
"""
s2_extract_rolling.py - Extract rolling-window embeddings.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s2_extract_rolling.py
"""
from __future__ import annotations

import json
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
    except Exception:
        pass
    return "cpu"


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

    with open(PROJECT_ROOT / "config" / "s2_config.yaml") as f:
        cfg = yaml.safe_load(f)

    from s2light.rolling_embeddings import extract_rolling_embeddings

    device = get_device(cfg.get("runtime", {}).get("device", "auto"))
    s2_dir = PROJECT_ROOT / cfg["paths"]["s2_dir"]
    rw = cfg["rolling_windows"]

    emb, meta = extract_rolling_embeddings(
        s0_dir=PROJECT_ROOT / cfg["paths"]["s0_dir"],
        encoder_ckpt=PROJECT_ROOT / cfg["paths"]["s15_encoder"],
        output_path=s2_dir / "rolling_embeddings.npy",
        window_len=rw["window_len"],
        stride=rw["stride"],
        seq_len=rw["seq_len"],
        device=device,
    )

    with open(s2_dir / "rolling_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.getLogger("s2").info("Rolling extraction complete.")


if __name__ == "__main__":
    main()
