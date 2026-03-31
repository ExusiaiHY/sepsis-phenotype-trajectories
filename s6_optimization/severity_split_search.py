"""
severity_split_search.py - Search severity split targets for S6 phenotype relabeling.

This module performs a light-weight search over existing S6 phenotype assignments.
It does not rerun the full causal phenotyping pipeline. Instead, it simulates
cluster-aware severity suffixes on top of an existing run directory and scores
the tradeoff between mortality separation and cross-center consistency.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from s6_optimization.baseline_comparison import summarize_labeling


DEFAULT_SCORE_WEIGHTS = {
    "weighted_mortality_std": 3.0,
    "supported_mortality_range": 2.0,
    "supported_group_count": 0.25,
    "center_distribution_l1": -10.0,
    "center_mortality_deviation": -10.0,
    "dominant_group_fraction": -0.5,
}


@dataclass(frozen=True)
class SearchConfig:
    min_group_size: int = 50
    max_combination_size: int = 4
    split_mode: str = "both"
    min_candidate_size: int = 200


def _score_summary(summary: dict, weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    return round(
        sum(float(summary.get(metric, 0.0) or 0.0) * weight for metric, weight in weights.items()),
        6,
    )


def _cluster_risk_tiers(df: pd.DataFrame) -> tuple[int, int]:
    stats = (
        df.groupby("dominant_cluster", observed=True)["mortality_actual"]
        .mean()
        .sort_values()
    )
    recovering_cluster = int(stats.index[0])
    critical_cluster = int(stats.index[-1])
    return recovering_cluster, critical_cluster


def _simulate_labels(
    labels: pd.Series,
    dominant_clusters: pd.Series,
    targets: tuple[str, ...],
    *,
    split_mode: str,
    recovering_cluster: int,
    critical_cluster: int,
) -> pd.Series:
    simulated = labels.astype(str).copy()
    mask = simulated.isin(targets)
    if split_mode in {"critical", "both"}:
        simulated = simulated.where(
            ~(mask & (dominant_clusters == critical_cluster)),
            simulated + "_critical",
        )
    if split_mode in {"recovering", "both"}:
        simulated = simulated.where(
            ~(mask & (dominant_clusters == recovering_cluster)),
            simulated + "_recovering",
        )
    return simulated


def _candidate_targets(df: pd.DataFrame, cfg: SearchConfig) -> list[str]:
    counts = df["phenotype_key"].value_counts()
    return sorted(
        str(label)
        for label, n in counts.items()
        if n >= cfg.min_candidate_size
        and not str(label).endswith("_critical")
        and not str(label).endswith("_recovering")
    )


def search_severity_split_targets(
    run_dir: str | Path,
    static_path: str | Path,
    *,
    min_group_size: int = 50,
    max_combination_size: int = 4,
    split_mode: str = "both",
    min_candidate_size: int = 200,
    top_k: int = 10,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Search for phenotype targets that benefit from cluster-aware severity suffixes.
    """
    cfg = SearchConfig(
        min_group_size=min_group_size,
        max_combination_size=max_combination_size,
        split_mode=split_mode,
        min_candidate_size=min_candidate_size,
    )
    run_dir = Path(run_dir)
    static_path = Path(static_path)

    phenotype_df = pd.read_csv(run_dir / "phenotype_assignments.csv")
    static = pd.read_csv(static_path)
    phenotype_df["mortality_actual"] = static["mortality_inhospital"].fillna(0).astype(float)
    phenotype_df["center_id"] = static["center_id"].astype(str)

    recovering_cluster, critical_cluster = _cluster_risk_tiers(phenotype_df)
    candidates = _candidate_targets(phenotype_df, cfg)

    baseline_summary = summarize_labeling(
        phenotype_df["phenotype_key"].astype(str).values,
        phenotype_df["mortality_actual"].values,
        center_ids=phenotype_df["center_id"].values,
        min_group_size=cfg.min_group_size,
    )
    baseline_score = _score_summary(baseline_summary, weights=weights)

    evaluations = []
    for size in range(1, min(cfg.max_combination_size, len(candidates)) + 1):
        for targets in itertools.combinations(candidates, size):
            simulated = _simulate_labels(
                phenotype_df["phenotype_key"],
                phenotype_df["dominant_cluster"],
                targets,
                split_mode=cfg.split_mode,
                recovering_cluster=recovering_cluster,
                critical_cluster=critical_cluster,
            )
            summary = summarize_labeling(
                simulated.values,
                phenotype_df["mortality_actual"].values,
                center_ids=phenotype_df["center_id"].values,
                min_group_size=cfg.min_group_size,
            )
            score = _score_summary(summary, weights=weights)
            evaluations.append({
                "targets": list(targets),
                "score": score,
                "score_delta_vs_current": round(score - baseline_score, 6),
                "summary": {
                    key: summary[key]
                    for key in [
                        "supported_group_count",
                        "supported_mortality_range",
                        "weighted_mortality_std",
                        "center_distribution_l1",
                        "center_mortality_deviation",
                        "dominant_group_fraction",
                        "rare_group_fraction",
                    ]
                },
            })

    evaluations.sort(
        key=lambda item: (
            item["score"],
            item["summary"]["supported_mortality_range"],
            item["summary"]["weighted_mortality_std"],
        ),
        reverse=True,
    )

    report = {
        "run_dir": str(run_dir),
        "static_path": str(static_path),
        "search_config": {
            "min_group_size": cfg.min_group_size,
            "max_combination_size": cfg.max_combination_size,
            "split_mode": cfg.split_mode,
            "min_candidate_size": cfg.min_candidate_size,
            "weights": weights or DEFAULT_SCORE_WEIGHTS,
        },
        "cluster_risk_tiers": {
            "recovering_cluster": recovering_cluster,
            "critical_cluster": critical_cluster,
        },
        "current": {
            "targets": [],
            "score": baseline_score,
            "summary": {
                key: baseline_summary[key]
                for key in [
                    "supported_group_count",
                    "supported_mortality_range",
                    "weighted_mortality_std",
                    "center_distribution_l1",
                    "center_mortality_deviation",
                    "dominant_group_fraction",
                    "rare_group_fraction",
                ]
            },
        },
        "candidate_labels": candidates,
        "recommendation": evaluations[0] if evaluations else None,
        "top_results": evaluations[:top_k],
    }
    return report


def write_severity_split_search_report(
    run_dir: str | Path,
    static_path: str | Path,
    output_path: str | Path,
    **kwargs,
) -> dict:
    report = search_severity_split_targets(run_dir, static_path, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
