"""
baseline_comparison.py - Quantitative comparison between baseline clusters and S6 phenotypes.

Produces a stable comparison report so each S6 run can answer:
  - Did outcome separation improve?
  - Did cross-center consistency improve?
  - Did label collapse or rare-group fragmentation get worse?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIRECTIONS = {
    "supported_group_count": "higher",
    "supported_mortality_range": "higher",
    "weighted_mortality_std": "higher",
    "center_distribution_l1": "lower",
    "center_mortality_deviation": "lower",
    "dominant_group_fraction": "lower",
    "rare_group_fraction": "lower",
}


def dominant_cluster_labels(window_labels: np.ndarray) -> np.ndarray:
    """Collapse a per-window cluster trajectory to a dominant cluster label."""
    dominant = []
    for row in window_labels:
        values, counts = np.unique(row, return_counts=True)
        dominant.append(str(int(values[np.argmax(counts)])))
    return np.asarray(dominant, dtype=object)


def _distribution(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def _l1_distance(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return float(sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys))


def summarize_labeling(
    labels: np.ndarray | pd.Series,
    outcome: np.ndarray | pd.Series,
    center_ids: np.ndarray | pd.Series | None = None,
    min_group_size: int = 50,
) -> dict:
    """Summarize label quality with separation and consistency metrics."""
    label_series = pd.Series(labels, name="label").astype(str)
    outcome_series = pd.Series(outcome, name="outcome").astype(float)
    df = pd.DataFrame({"label": label_series, "outcome": outcome_series})
    if center_ids is not None:
        df["center_id"] = pd.Series(center_ids, name="center_id").astype(str)

    overall_rate = float(df["outcome"].mean())
    group_stats = (
        df.groupby("label", observed=True)["outcome"]
        .agg(n="size", mortality="mean")
        .reset_index()
        .sort_values("label")
    )
    supported = group_stats[group_stats["n"] >= min_group_size].copy()

    if supported.empty:
        supported_range = 0.0
        weighted_std = 0.0
    else:
        supported_range = float(supported["mortality"].max() - supported["mortality"].min())
        weights = supported["n"].to_numpy(dtype=float)
        mortality = supported["mortality"].to_numpy(dtype=float)
        weighted_std = float(np.sqrt(np.average((mortality - overall_rate) ** 2, weights=weights)))

    dominant_group_fraction = float(group_stats["n"].max() / len(df))
    rare_group_fraction = float(group_stats.loc[group_stats["n"] < min_group_size, "n"].sum() / len(df))

    summary = {
        "n_samples": int(len(df)),
        "overall_outcome_rate": round(overall_rate, 4),
        "group_count": int(len(group_stats)),
        "supported_group_count": int(len(supported)),
        "supported_mortality_range": round(supported_range, 4),
        "weighted_mortality_std": round(weighted_std, 4),
        "dominant_group_fraction": round(dominant_group_fraction, 4),
        "rare_group_fraction": round(rare_group_fraction, 4),
        "group_stats": [
            {
                "label": row["label"],
                "n": int(row["n"]),
                "mortality": round(float(row["mortality"]), 4),
            }
            for _, row in group_stats.iterrows()
        ],
    }

    if "center_id" not in df.columns:
        summary["center_distribution_l1"] = None
        summary["center_mortality_deviation"] = None
        return summary

    overall_dist = _distribution(df["label"])
    center_distances = {}
    weighted_l1 = 0.0
    center_mortality_deviations = []

    for center, center_df in df.groupby("center_id", observed=True):
        dist = _distribution(center_df["label"])
        l1 = _l1_distance(overall_dist, dist)
        center_distances[str(center)] = round(l1, 4)
        weighted_l1 += l1 * (len(center_df) / len(df))

    supported_lookup = supported.set_index("label")["mortality"].to_dict()
    for center, center_df in df.groupby("center_id", observed=True):
        by_center = (
            center_df.groupby("label", observed=True)["outcome"]
            .agg(n="size", mortality="mean")
            .reset_index()
        )
        for _, row in by_center.iterrows():
            label = row["label"]
            if label not in supported_lookup or row["n"] < max(10, min_group_size // 5):
                continue
            center_mortality_deviations.append(
                abs(float(row["mortality"]) - float(supported_lookup[label]))
            )

    summary["center_distribution_l1"] = round(weighted_l1, 4)
    summary["center_mortality_deviation"] = round(
        float(np.mean(center_mortality_deviations)) if center_mortality_deviations else 0.0,
        4,
    )
    summary["center_distribution_by_center"] = center_distances
    return summary


def compare_summaries(baseline: dict, optimized: dict) -> dict:
    """Compare baseline and optimized label summaries using fixed directions."""
    metric_deltas = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        base = baseline.get(metric)
        opt = optimized.get(metric)
        if base is None or opt is None:
            metric_deltas[metric] = {
                "baseline": base,
                "optimized": opt,
                "delta": None,
                "relative_pct": None,
                "improved": None,
                "direction": direction,
            }
            continue

        delta = float(opt - base)
        if base == 0:
            relative_pct = None
        else:
            relative_pct = float((delta / base) * 100.0)
        improved = delta > 0 if direction == "higher" else delta < 0

        metric_deltas[metric] = {
            "baseline": round(float(base), 4),
            "optimized": round(float(opt), 4),
            "delta": round(delta, 4),
            "relative_pct": round(relative_pct, 2) if relative_pct is not None else None,
            "improved": improved,
            "direction": direction,
        }

    return {"metric_deltas": metric_deltas}


def generate_baseline_comparison_report(
    *,
    static_path: Path,
    window_labels_path: Path,
    phenotype_assignments_path: Path,
    output_path: Path,
    min_group_size: int = 50,
) -> dict:
    """Create and save the baseline-vs-S6 comparison report."""
    static = pd.read_csv(static_path)
    window_labels = np.load(window_labels_path)
    phenotype_df = pd.read_csv(phenotype_assignments_path)

    outcome_col = "mortality_inhospital"
    center_ids = static["center_id"].values if "center_id" in static.columns else None

    baseline_labels = dominant_cluster_labels(window_labels)
    optimized_labels = phenotype_df["phenotype_key"].astype(str).values
    outcome = static[outcome_col].fillna(0).astype(float).values

    baseline = summarize_labeling(
        baseline_labels,
        outcome,
        center_ids=center_ids,
        min_group_size=min_group_size,
    )
    optimized = summarize_labeling(
        optimized_labels,
        outcome,
        center_ids=center_ids,
        min_group_size=min_group_size,
    )
    comparison = compare_summaries(baseline, optimized)

    report = {
        "baseline_labeling": "s2_dominant_cluster",
        "optimized_labeling": "s6_mechanism_based_phenotype",
        "min_group_size": int(min_group_size),
        "baseline": baseline,
        "optimized": optimized,
        **comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
