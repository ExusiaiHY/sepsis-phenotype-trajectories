#!/usr/bin/env python3
"""
s15_diagnostics.py - Run probes on all three embedding families + compare.

How to run:
  cd project
  python3.14 scripts/s15_diagnostics.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.diagnostics import run_all_diagnostics, print_diagnostics_comparison


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

    with open(PROJECT_ROOT / "config" / "s15_config.yaml") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s1_dir = PROJECT_ROOT / cfg["paths"]["s1_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]

    static_path = s0_dir / "static.csv"
    splits_path = s0_dir / "splits.json"
    masks_path = s0_dir / "processed" / "masks_continuous.npy"

    reports = []

    # PCA
    reports.append(run_all_diagnostics(
        np.load(s1_dir / "embeddings_pca.npy"),
        static_path, splits_path, masks_path,
        s15_dir / "diagnostics_pca.json", label="PCA",
    ))

    # S1 masked
    reports.append(run_all_diagnostics(
        np.load(s1_dir / "embeddings_ss.npy"),
        static_path, splits_path, masks_path,
        s15_dir / "diagnostics_s1_masked.json", label="S1_masked",
    ))

    # S1.5 contrastive
    reports.append(run_all_diagnostics(
        np.load(s15_dir / "embeddings_s15.npy"),
        static_path, splits_path, masks_path,
        s15_dir / "diagnostics_s15_contrastive.json", label="S15_contrastive",
    ))

    print_diagnostics_comparison(reports)


if __name__ == "__main__":
    main()
