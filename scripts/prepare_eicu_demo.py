#!/usr/bin/env python3
"""
prepare_eicu_demo.py - Build cached eICU demo artifacts for the legacy V1 pipeline.

How to run:
  cd project
  ./.venv/bin/python scripts/prepare_eicu_demo.py
  ./.venv/bin/python scripts/prepare_eicu_demo.py --data-dir /path/to/eicu-demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eicu_loader import prepare_eicu_demo_artifacts
from utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare cached eICU demo artifacts")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data-dir", default="data/external/eicu_demo")
    parser.add_argument("--output-dir", default="data/processed_eicu_demo")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--tag", default="eicu_demo")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    config["data"]["source"] = "eicu"
    config.setdefault("data", {}).setdefault("eicu", {})
    config["data"]["eicu"]["data_dir"] = args.data_dir
    config["data"]["eicu"]["n_timesteps"] = args.hours
    if args.max_patients is not None:
        config["data"]["eicu"]["max_patients"] = args.max_patients

    output_dir = PROJECT_ROOT / args.output_dir
    report = prepare_eicu_demo_artifacts(config, output_dir=output_dir, tag=args.tag)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
