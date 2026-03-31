"""
s6_compare_runs.py - Compare two S6 run directories.

Usage:
  python3 scripts/s6_compare_runs.py data/s6 data/s6_rerun_20260401 \
      --output data/s6_rerun_20260401/iteration_vs_s6.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6_optimization.run_comparison import write_s6_run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two S6 run directories.")
    parser.add_argument("previous_run", help="Previous S6 run directory relative to project root")
    parser.add_argument("current_run", help="Current S6 run directory relative to project root")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path relative to project root",
    )
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    report = write_s6_run_comparison(
        PROJECT_ROOT / args.previous_run,
        PROJECT_ROOT / args.current_run,
        PROJECT_ROOT / args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
