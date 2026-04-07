"""
s6_alpha_sweep_report.py - Summarize and compare multiple S6 run directories.

Usage:
  ./.venv/bin/python scripts/s6_alpha_sweep_report.py \
      data/s6_rerun_20260401_round6 \
      data/s6_rerun_20260401_round7 \
      data/s6_rerun_20260401_alpha06 \
      --references data/s6_rerun_20260401_round6 data/s6_rerun_20260401_round7 \
      --output-json data/s6_alpha_sweep_report.json \
      --output-csv data/s6_alpha_sweep_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s6_optimization.run_comparison import compare_s6_runs, load_s6_run_summary


def _label_for_path(path: str | Path) -> str:
    return Path(path).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiple S6 run directories.")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directories relative to project root",
    )
    parser.add_argument(
        "--references",
        nargs="*",
        default=[],
        help="Reference run directories relative to project root for pairwise comparisons",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path relative to project root",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path relative to project root",
    )
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    run_paths = [PROJECT_ROOT / run for run in args.runs]
    ref_paths = [PROJECT_ROOT / run for run in args.references]

    run_summaries = []
    for run_path in run_paths:
        summary = load_s6_run_summary(run_path)
        run_summaries.append(
            {
                "label": _label_for_path(run_path),
                "summary": summary,
            }
        )

    comparisons = {}
    for run_path in run_paths:
        run_label = _label_for_path(run_path)
        comparisons[run_label] = {}
        for ref_path in ref_paths:
            ref_label = _label_for_path(ref_path)
            if run_path.resolve() == ref_path.resolve():
                continue
            comparisons[run_label][ref_label] = compare_s6_runs(ref_path, run_path)

    report = {
        "runs": run_summaries,
        "comparisons": comparisons,
    }

    if args.output_json:
        output_json = PROJECT_ROOT / args.output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if args.output_csv:
        output_csv = PROJECT_ROOT / args.output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label",
            "run_dir",
            "mean_sofa_total",
            "cate_std",
            "supported_group_count",
            "supported_mortality_range",
            "weighted_mortality_std",
            "center_distribution_l1",
            "center_mortality_deviation",
            "dominant_group_fraction",
            "rare_group_fraction",
            "group_count",
        ]
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for run in run_summaries:
                row = {"label": run["label"], **run["summary"]}
                writer.writerow({key: row.get(key) for key in fieldnames})

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
