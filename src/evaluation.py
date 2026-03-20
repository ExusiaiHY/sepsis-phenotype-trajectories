"""
evaluation.py - Result evaluation module

Responsibilities:
1. Internal clustering quality metrics
2. External validation against ground truth labels (when available)
3. Survival stratification analysis (Kaplan-Meier + log-rank test)
4. Subtype clinical profile construction
5. Structured evaluation report generation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy import stats
from typing import Any

from utils import setup_logger, timer

logger = setup_logger(__name__)


# ============================================================
# Comprehensive Evaluation Entry Point
# ============================================================

@timer
def evaluate_all(
    X: np.ndarray,
    labels: np.ndarray,
    patient_info: pd.DataFrame,
    config: dict,
    feature_df: pd.DataFrame | None = None,
) -> dict:
    """
    Run all evaluations and return a structured report.

    Parameters
    ----------
    X : np.ndarray          Feature matrix (for internal metrics)
    labels : np.ndarray     Cluster labels
    patient_info : pd.DataFrame  Patient info (with outcomes and true subtypes)
    config : dict
    feature_df : pd.DataFrame | None  Raw features (for clinical profiling)

    Returns
    -------
    report : dict with all evaluation results.
    """
    report = {}

    # 1. Internal clustering metrics
    report["cluster_metrics"] = compute_internal_metrics(X, labels)

    # 2. External metrics (if true labels available)
    if "subtype_true" in patient_info.columns:
        report["external_metrics"] = compute_external_metrics(
            labels, patient_info["subtype_true"].values
        )

    # 3. Survival stratification
    if config["evaluation"]["survival"]["enabled"]:
        report["survival"] = survival_analysis(labels, patient_info, config)

    # 4. Subtype clinical profiles
    report["subtype_profiles"] = build_subtype_profiles(
        labels, patient_info, feature_df
    )

    # 5. Cluster size and balance
    report["cluster_sizes"] = _cluster_size_summary(labels)

    _log_summary(report)
    return report


# ============================================================
# Internal Clustering Metrics
# ============================================================

def compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal clustering evaluation metrics."""
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        logger.warning("Cluster count < 2, cannot compute internal metrics")
        return {}

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(silhouette_score(X, labels), 4),
        "calinski_harabasz_score": round(calinski_harabasz_score(X, labels), 2),
        "davies_bouldin_score": round(davies_bouldin_score(X, labels), 4),
    }

    from sklearn.metrics import silhouette_samples
    sample_scores = silhouette_samples(X, labels)
    for k in range(n_clusters):
        mask = labels == k
        metrics[f"silhouette_cluster_{k}"] = round(sample_scores[mask].mean(), 4)

    return metrics


# ============================================================
# External Metrics (with ground truth)
# ============================================================

def compute_external_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """
    Compute clustering-to-ground-truth alignment metrics.

    ARI (Adjusted Rand Index): [-1, 1], higher is better
    NMI (Normalized Mutual Info): [0, 1], higher is better
    """
    metrics = {
        "adjusted_rand_index": round(adjusted_rand_score(true_labels, pred_labels), 4),
        "normalized_mutual_info": round(
            normalized_mutual_info_score(true_labels, pred_labels), 4
        ),
    }

    cross_tab = pd.crosstab(
        pd.Series(true_labels, name="true_subtype"),
        pd.Series(pred_labels, name="cluster_label"),
    )
    metrics["contingency_table"] = cross_tab

    return metrics


# ============================================================
# Survival Stratification Analysis
# ============================================================

def survival_analysis(
    labels: np.ndarray,
    patient_info: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Perform survival stratification analysis across subtypes.

    Uses Kaplan-Meier estimation and log-rank test.
    Falls back to simplified chi-square/Kruskal-Wallis if lifelines is unavailable.
    """
    surv_cfg = config["evaluation"]["survival"]
    time_col = surv_cfg["time_col"]
    event_col = surv_cfg["event_col"]

    result = {"per_cluster": {}}
    n_clusters = len(np.unique(labels))

    # Per-cluster outcome statistics
    for k in range(n_clusters):
        mask = labels == k
        subset = patient_info[mask]
        result["per_cluster"][k] = {
            "n_patients": int(mask.sum()),
            "mortality_rate": round(subset[event_col].mean(), 4) if event_col in subset.columns else None,
            "mean_icu_los": round(subset[time_col].mean(), 2) if time_col in subset.columns else None,
            "median_icu_los": round(subset[time_col].median(), 2) if time_col in subset.columns else None,
        }

    # Try log-rank test with lifelines
    try:
        from lifelines.statistics import logrank_test
        from lifelines import KaplanMeierFitter

        p_matrix = np.ones((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                mask_i = labels == i
                mask_j = labels == j
                lr = logrank_test(
                    patient_info.loc[mask_i, time_col],
                    patient_info.loc[mask_j, time_col],
                    event_observed_A=patient_info.loc[mask_i, event_col],
                    event_observed_B=patient_info.loc[mask_j, event_col],
                )
                p_matrix[i, j] = lr.p_value
                p_matrix[j, i] = lr.p_value

        result["log_rank_p_matrix"] = pd.DataFrame(
            p_matrix,
            index=[f"Cluster_{i}" for i in range(n_clusters)],
            columns=[f"Cluster_{j}" for j in range(n_clusters)],
        )

        km_data = {}
        kmf = KaplanMeierFitter()
        for k in range(n_clusters):
            mask = labels == k
            kmf.fit(
                patient_info.loc[mask, time_col],
                event_observed=patient_info.loc[mask, event_col],
                label=f"Cluster {k}",
            )
            km_data[k] = {
                "timeline": kmf.timeline.tolist(),
                "survival_function": kmf.survival_function_.iloc[:, 0].tolist(),
                "median_survival": kmf.median_survival_time_,
            }
        result["km_data"] = km_data
        result["lifelines_available"] = True

    except ImportError:
        logger.warning("lifelines not installed, using simplified survival analysis")
        result["lifelines_available"] = False

        if event_col in patient_info.columns:
            contingency = pd.crosstab(labels, patient_info[event_col])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            result["chi2_test"] = {
                "chi2": round(chi2, 4),
                "p_value": round(p_val, 6),
                "dof": dof,
            }

        if time_col in patient_info.columns:
            groups = [patient_info.loc[labels == k, time_col].values
                      for k in range(n_clusters)]
            h_stat, p_val = stats.kruskal(*groups)
            result["kruskal_wallis_los"] = {
                "h_statistic": round(h_stat, 4),
                "p_value": round(p_val, 6),
            }

    return result


# ============================================================
# Subtype Clinical Profiles
# ============================================================

def build_subtype_profiles(
    labels: np.ndarray,
    patient_info: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a clinical profile summary for each subtype."""
    df = patient_info.copy()
    df["cluster"] = labels
    n_clusters = len(np.unique(labels))

    profiles = []
    for k in range(n_clusters):
        subset = df[df["cluster"] == k]
        profile = {
            "cluster": k,
            "n_patients": len(subset),
            "proportion": f"{len(subset) / len(df):.1%}",
        }

        if "age" in subset.columns:
            profile["age_mean"] = round(subset["age"].mean(), 1)
        if "gender" in subset.columns:
            profile["male_ratio"] = f"{(subset['gender'] == 'M').mean():.1%}"

        if "mortality_28d" in subset.columns:
            profile["mortality_28d"] = f"{subset['mortality_28d'].mean():.1%}"
        if "icu_los" in subset.columns:
            profile["icu_los_mean"] = round(subset["icu_los"].mean(), 1)
        if "shock_onset" in subset.columns:
            profile["shock_rate"] = f"{subset['shock_onset'].mean():.1%}"

        profiles.append(profile)

    return pd.DataFrame(profiles)


# ============================================================
# Helper Functions
# ============================================================

def _cluster_size_summary(labels: np.ndarray) -> dict:
    """Cluster size statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {
        "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
        "balance_ratio": round(counts.min() / counts.max(), 3),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
    }


def _log_summary(report: dict) -> None:
    """Print evaluation summary to log."""
    logger.info("=" * 50)
    logger.info("Evaluation Report Summary")
    logger.info("=" * 50)

    if "cluster_metrics" in report:
        m = report["cluster_metrics"]
        logger.info(f"  Clusters: {m.get('n_clusters', '?')}")
        logger.info(f"  Silhouette: {m.get('silhouette_score', '?')}")
        logger.info(f"  CH Index: {m.get('calinski_harabasz_score', '?')}")
        logger.info(f"  DB Index: {m.get('davies_bouldin_score', '?')}")

    if "external_metrics" in report:
        e = report["external_metrics"]
        logger.info(f"  ARI: {e.get('adjusted_rand_index', '?')}")
        logger.info(f"  NMI: {e.get('normalized_mutual_info', '?')}")

    if "survival" in report:
        surv = report["survival"]
        for k, info in surv.get("per_cluster", {}).items():
            logger.info(f"  Subtype {k}: mortality={info.get('mortality_rate', '?')}, "
                         f"ICU LOS mean={info.get('mean_icu_los', '?')}h")
