#!/usr/bin/env python3
"""
s1_extract.py - Extract both SS and PCA embeddings.

How to run:
  cd project
  KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s1_extract.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device(preference="auto"):
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


def main():
    with open(PROJECT_ROOT / "config" / "s1_config.yaml") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    s0_dir = PROJECT_ROOT / config["paths"]["s0_dir"]
    s1_dir = PROJECT_ROOT / config["paths"]["s1_dir"]
    device = get_device(config.get("runtime", {}).get("device", "auto"))

    from s1.extract_embeddings import extract_ss_embeddings, extract_pca_embeddings

    extract_ss_embeddings(s0_dir, s1_dir, device=device)
    extract_pca_embeddings(s0_dir, s1_dir, n_components=config["pca_baseline"]["n_components"])


if __name__ == "__main__":
    main()
