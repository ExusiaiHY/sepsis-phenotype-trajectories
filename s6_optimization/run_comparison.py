"""
run_comparison.py - Compare two S6 run directories quantitatively.

This complements baseline_comparison.py:
  - baseline_comparison.py compares one S6 run against S2 dominant clusters
  - run_comparison.py compares two S6 runs against each other
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN_METRIC_DIRECTIONS = {
    "mean_sofa_total": "contextual",
    "cate_std": "lower",
    "supported_group_count": "higher",
    "supported_mortality_range": "higher",
    "weighted_mortality_std": "higher",
    "center_distribution_l1": "lower",
    "center_mortality_deviation": "lower",
    "dominant_group_fraction": "lower",
    "rare_group_fraction": "lower",
}


def load_s6_run_summary(run_dir: str | Path) -> dict:
    """Load the compact comparison summary for one S6 run directory."""
    run_dir = Path(run_dir)
    baseline_report = json.loads((run_dir / "baseline_comparison.json").read_text())
    causal_report = json.loads((run_dir / "causal_phenotyping_report.json").read_text())
    organ_scores = pd.read_csv(run_dir / "organ_scores.csv")

    optimized = baseline_report["optimized"]
    return {
        "run_dir": str(run_dir),
        "mean_sofa_total": round(float(organ_scores["sofa_total"].mean()), 4),
        "cate_std": round(float(causal_report["cate_summary"]["std"]), 4),
        "supported_group_count": int(optimized["supported_group_count"]),
        "supported_mortality_range": round(float(optimized["supported_mortality_range"]), 4),
        "weighted_mortality_std": round(float(optimized["weighted_mortality_std"]), 4),
        "center_distribution_l1": round(float(optimized["center_distribution_l1"]), 4),
        "center_mortality_deviation": round(float(optimized["center_mortality_deviation"]), 4),
        "dominant_group_fraction": round(float(optimized["dominant_group_fraction"]), 4),
        "rare_group_fraction": round(float(optimized["rare_group_fraction"]), 4),
        "group_count": int(optimized["group_count"]),
        "phenotype_counts": {
            item["label"]: int(item["n"]) for item in optimized["group_stats"]
        },
    }


def compare_s6_runs(
    previous_run_dir: str | Path,
    current_run_dir: str | Path,
) -> dict:
    """Compare two S6 runs and quantify which metrics improved or regressed."""
    previous = load_s6_run_summary(previous_run_dir)
    current = load_s6_run_summary(current_run_dir)

    metric_deltas = {}
    for metric, direction in RUN_METRIC_DIRECTIONS.items():
        prev = previous[metric]
        curr = current[metric]
        delta = float(curr - prev)
        relative_pct = None if prev == 0 else float((delta / prev) * 100.0)
        if direction == "contextual":
            improved = None
        elif direction == "higher":
            improved = delta > 0
        else:
            improved = delta < 0

        metric_deltas[metric] = {
            "previous": round(float(prev), 4),
            "current": round(float(curr), 4),
            "delta": round(delta, 4),
            "relative_pct": round(relative_pct, 2) if relative_pct is not None else None,
            "direction": direction,
            "improved": improved,
        }

    return {
        "previous_run": previous,
        "current_run": current,
        "metric_deltas": metric_deltas,
    }


def write_s6_run_comparison(
    previous_run_dir: str | Path,
    current_run_dir: str | Path,
    output_path: str | Path,
) -> dict:
    """Create and save a comparison report between two S6 runs."""
    report = compare_s6_runs(previous_run_dir, current_run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
