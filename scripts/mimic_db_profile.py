#!/usr/bin/env python3
"""
mimic_db_profile.py - Generate a reproducible profile of the local DuckDB MIMIC database.

How to run:
  cd project
  ./.venv/bin/python scripts/mimic_db_profile.py
  ./.venv/bin/python scripts/mimic_db_profile.py --db-path archive/db/mimic4.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mimic_db_profile import build_mimic_profile, save_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Profile the local MIMIC DuckDB database")
    parser.add_argument("--db-path", default=None, help="Optional explicit path to mimic4.db")
    parser.add_argument("--output-dir", default="data/mimic_db_profile")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_mimic_profile(Path(args.db_path) if args.db_path else None)
    json_path, md_path = save_profile(report, PROJECT_ROOT / args.output_dir)
    print(f"Saved JSON profile to {json_path}")
    print(f"Saved Markdown summary to {md_path}")


if __name__ == "__main__":
    main()
