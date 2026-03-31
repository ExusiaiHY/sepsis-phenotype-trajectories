"""
s6_search_severity_splits.py - Small-scale search over S6 severity split targets.

Example:
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE ./.venv/bin/python \
      scripts/s6_search_severity_splits.py \
      --run-dir data/s6_rerun_20260401 \
      --static-path data/s0/static.csv \
      --output data/s6_rerun_20260401/severity_split_search.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6_optimization.severity_split_search import write_severity_split_search_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search S6 severity split targets.")
    parser.add_argument("--run-dir", required=True, help="Existing S6 run directory relative to project root")
    parser.add_argument("--static-path", default="data/s0/static.csv", help="Static CSV path relative to project root")
    parser.add_argument("--output", required=True, help="Output JSON path relative to project root")
    parser.add_argument("--min-group-size", type=int, default=50)
    parser.add_argument("--max-combination-size", type=int, default=4)
    parser.add_argument("--split-mode", choices=["critical", "recovering", "both"], default="both")
    parser.add_argument("--min-candidate-size", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    report = write_severity_split_search_report(
        PROJECT_ROOT / args.run_dir,
        PROJECT_ROOT / args.static_path,
        PROJECT_ROOT / args.output,
        min_group_size=args.min_group_size,
        max_combination_size=args.max_combination_size,
        split_mode=args.split_mode,
        min_candidate_size=args.min_candidate_size,
        top_k=args.top_k,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
