#!/usr/bin/env python3
"""s4_prepare_treatments.py - Build Stage 4 treatment feature bundle."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s4.treatment_features import build_treatment_feature_bundle


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Stage 4 treatment-aware bundle")
    parser.add_argument("--config", default="config/s4_config.yaml")
    parser.add_argument("--source", choices=["mimic", "eicu"], default=None)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-hours", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--all-patients", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    config_path = _resolve(args.config)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    source = args.source or cfg["data"]["source"]
    report = build_treatment_feature_bundle(
        source=source,
        prepared_dir=_resolve(args.prepared_dir or cfg["paths"]["prepared_dir"]),
        raw_dir=_resolve(args.raw_dir or cfg["paths"]["raw_dir"]),
        output_dir=_resolve(args.output_dir or cfg["paths"]["treatment_dir"]),
        n_hours=int(args.n_hours or cfg["data"].get("n_hours", 48)),
        max_patients=args.max_patients if args.max_patients is not None else cfg["data"].get("max_patients"),
        tag=args.tag or cfg["data"].get("tag", "eicu_demo"),
        sepsis3_only=not args.all_patients and bool(cfg["data"].get("sepsis3_only", True)),
    )
    logging.getLogger("s4.prepare").info("Treatment bundle ready: %s", report["report_path"])


if __name__ == "__main__":
    main()
