#!/usr/bin/env python3
"""
s19_prepare.py - Build an S0-compatible bridge dataset from local Sepsis 2019 stubs.

How to run:
  cd project
  ./.venv/bin/python scripts/s19_prepare.py
  ./.venv/bin/python scripts/s19_prepare.py --max-patients 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from load_sepsis2019 import PROJECT_FEATURES, load_sepsis2019
from s15.sepsis2019_bridge import build_bridge_bundle, write_bridge_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Sepsis 2019 bridge dataset")
    parser.add_argument("--data-dir", default="archive/sepsis2019_stubs")
    parser.add_argument("--output-dir", default="data/s19_bridge")
    parser.add_argument("--reference-stats", default="data/s0/processed/preprocess_stats.json")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("scripts.s19_prepare")

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    reference_stats = PROJECT_ROOT / args.reference_stats if args.reference_stats else None
    if reference_stats is not None and not reference_stats.exists():
        logger.warning("Reference preprocessing stats not found at %s; using self stats.", reference_stats)
        reference_stats = None

    logger.info("Loading Sepsis 2019 stubs from %s", data_dir)
    time_series, patient_info = load_sepsis2019(
        data_dir=data_dir,
        n_hours=48,
        max_patients=args.max_patients,
        use_cache=not args.disable_cache,
    )

    bundle = build_bridge_bundle(
        time_series=time_series,
        patient_info=patient_info,
        source_feature_names=PROJECT_FEATURES,
    )
    report = write_bridge_dataset(
        output_dir=output_dir,
        bundle=bundle,
        reference_stats_path=reference_stats,
        split_method="random",
        seed=args.seed,
        stratify_by="sepsis_label",
    )

    logger.info(
        "Bridge ready: patients=%d, sepsis prevalence=%.4f, coverage=%d/%d",
        report["n_patients"],
        report["sepsis_prevalence"],
        len(report["mapped_features"]),
        21,
    )


if __name__ == "__main__":
    main()
