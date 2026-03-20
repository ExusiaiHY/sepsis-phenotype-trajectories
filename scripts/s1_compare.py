#!/usr/bin/env python3
"""
s1_compare.py - Compare SS encoder vs PCA baseline clustering.

How to run:
  cd project
  python3.14 scripts/s1_compare.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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

    from s1.compare_vs_pca import compare_embeddings

    compare_embeddings(s0_dir, s1_dir, k_values=config["evaluation"]["k_values"])


if __name__ == "__main__":
    main()
