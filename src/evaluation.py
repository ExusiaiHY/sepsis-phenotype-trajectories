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

    if time_col not in patient_info.columns or event_col not in patient_info.columns:
        logger.warning(
            "Survival analysis skipped: missing required columns %s / %s",
            time_col,
            event_col,
        )
        result["lifelines_available"] = False
        result["n_valid_survival_rows"] = 0
        return result

    durations = pd.to_numeric(patient_info[time_col], errors="coerce")
    events = pd.to_numeric(patient_info[event_col], errors="coerce")
    valid_mask = durations.notna() & events.notna() & (durations > 0)
    result["n_valid_survival_rows"] = int(valid_mask.sum())

    def _add_simplified_survival_stats() -> None:
        """Populate coarse survival comparisons when KM/log-rank cannot run."""
        if valid_mask.any():
            contingency = pd.crosstab(labels[valid_mask.values], events.loc[valid_mask].astype(int))
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 1:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                result["chi2_test"] = {
                    "chi2": round(chi2, 4),
                    "p_value": round(p_val, 6),
                    "dof": dof,
                }

        groups = [durations.loc[(labels == k) & valid_mask.values].values for k in range(n_clusters)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            result["kruskal_wallis_los"] = {
                "h_statistic": round(h_stat, 4),
                "p_value": round(p_val, 6),
            }

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
                mask_i = (labels == i) & valid_mask.values
                mask_j = (labels == j) & valid_mask.values
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    p_matrix[i, j] = np.nan
                    p_matrix[j, i] = np.nan
                    continue
                lr = logrank_test(
                    durations.loc[mask_i],
                    durations.loc[mask_j],
                    event_observed_A=events.loc[mask_i],
                    event_observed_B=events.loc[mask_j],
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
            mask = (labels == k) & valid_mask.values
            if mask.sum() == 0:
                km_data[k] = {
                    "timeline": [],
                    "survival_function": [],
                    "median_survival": None,
                }
                continue
            kmf.fit(
                durations.loc[mask],
                event_observed=events.loc[mask],
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
        _add_simplified_survival_stats()
    except Exception as exc:
        logger.warning("lifelines survival analysis failed, using simplified survival analysis: %s", exc)
        result["lifelines_available"] = False
        _add_simplified_survival_stats()

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
