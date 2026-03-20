#!/usr/bin/env python3
"""
s15_compare.py - 3-way clustering comparison.

How to run:
  cd project
  python3.14 scripts/s15_compare.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s15.compare_three import compare_three_methods


def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

    with open(PROJECT_ROOT / "config" / "s15_config.yaml") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s1_dir = PROJECT_ROOT / cfg["paths"]["s1_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]

    results = compare_three_methods(s0_dir, s1_dir, s15_dir, k_values=cfg["evaluation"]["k_values"])

    out_path = s15_dir / "comparison_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logging.getLogger("s15").info(f"3-way comparison saved: {out_path}")


if __name__ == "__main__":
    main()
