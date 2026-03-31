"""
phenotype_naming.py - Mechanism-based clinical phenotype assignment.

Replaces pure risk-ordering (cluster 0-3 = low-to-high mortality) with
clinically interpretable phenotype names based on three axes:

  1. Organ dysfunction dominance (SOFA sub-scores computed from raw physiology)
  2. Treatment responsiveness   (CATE from causal analysis: positive = responsive)
  3. Trajectory dynamics        (stable / improving / deteriorating)

This module is called AFTER KMeans clustering (s2light) and causal analysis (s4)
to relabel each patient's phenotype with a mechanism-based clinical name.

All mortality numbers use real outcomes (14.2% base rate), never proxy labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("s6.phenotype_naming")


DEFAULT_THRESHOLDS = {
    "low_severity_sofa_max": 2,
    "low_severity_mortality_max": 0.10,
    "cate_responsiveness_threshold": 0.02,
    "deterioration_slope_threshold": 0.01,
    "high_mortality_reference": 0.1424,
    "multi_organ_min_organs": 3,
    "severity_split_targets": [
        "respiratory_failure",
        "hemodynamic_unstable_proxy_responsive",
    ],
}


def resolve_thresholds(thresholds: dict | None = None) -> dict:
    """Merge caller-provided overrides with stable default thresholds."""
    merged = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        for key, value in thresholds.items():
            if value is not None:
                merged[key] = value
    return merged


# ============================================================
# SOFA Sub-Score Computation from Raw Physiology
# ============================================================

@dataclass(frozen=True)
class OrganScore:
    """Per-patient organ dysfunction severity from physiology features."""
    respiratory: float   # PaO2/FiO2 ratio based
    cardiovascular: float  # MAP-based (proxy for vasopressor need)
    hepatic: float       # Bilirubin-based
    renal: float         # Creatinine-based
    coagulation: float   # Platelet-based
    neurological: float  # GCS-based


# SOFA thresholds adapted from Sepsis-3 (Singer et al., JAMA 2016)
# Each function returns 0-4 SOFA sub-score from physiological value

def _sofa_respiratory(pao2_fio2_ratio: float) -> int:
    """SOFA respiratory: PaO2/FiO2 ratio."""
    if np.isnan(pao2_fio2_ratio) or pao2_fio2_ratio >= 400:
        return 0
    if pao2_fio2_ratio >= 300:
        return 1
    if pao2_fio2_ratio >= 200:
        return 2
    if pao2_fio2_ratio >= 100:
        return 3
    return 4


def _sofa_cardiovascular(mean_map: float) -> int:
    """SOFA cardiovascular: MAP threshold (simplified, no vasopressor dose)."""
    if np.isnan(mean_map) or mean_map >= 70:
        return 0
    return 1  # MAP < 70 = score 1; higher scores require vasopressor dose data


def _sofa_hepatic(bilirubin: float) -> int:
    """SOFA hepatic: total bilirubin (mg/dL)."""
    if np.isnan(bilirubin) or bilirubin < 1.2:
        return 0
    if bilirubin < 2.0:
        return 1
    if bilirubin < 6.0:
        return 2
    if bilirubin < 12.0:
        return 3
    return 4


def _sofa_renal(creatinine: float) -> int:
    """SOFA renal: serum creatinine (mg/dL)."""
    if np.isnan(creatinine) or creatinine < 1.2:
        return 0
    if creatinine < 2.0:
        return 1
    if creatinine < 3.5:
        return 2
    if creatinine < 5.0:
        return 3
    return 4


def _sofa_coagulation(platelets: float) -> int:
    """SOFA coagulation: platelet count (K/uL)."""
    if np.isnan(platelets) or platelets >= 150:
        return 0
    if platelets >= 100:
        return 1
    if platelets >= 50:
        return 2
    if platelets >= 20:
        return 3
    return 4


def _sofa_neurological(gcs: float) -> int:
    """SOFA neurological: Glasgow Coma Scale."""
    if np.isnan(gcs) or gcs >= 15:
        return 0
    if gcs >= 13:
        return 1
    if gcs >= 10:
        return 2
    if gcs >= 6:
        return 3
    return 4


def _validate_raw_units(continuous: np.ndarray, feature_names: list[str]) -> None:
    """
    Guard: reject z-score standardized data for SOFA computation.

    SOFA thresholds are defined in clinical units (GCS 3-15, MAP ~40-120 mmHg,
    Platelet ~20-400 K/uL). Z-scored data has mean≈0, std≈1, which would
    produce wildly incorrect SOFA sub-scores.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    # Check MAP: raw range is ~40-120 mmHg; z-scored would be ~-3 to +3
    if "map" in idx:
        map_vals = continuous[:, :, idx["map"]]
        observed = map_vals[np.isfinite(map_vals)]
        if len(observed) > 100:
            median = float(np.median(observed))
            if abs(median) < 5.0:
                raise ValueError(
                    f"SOFA computation requires RAW clinical units, but MAP median={median:.2f} "
                    f"looks like z-score standardized data. Use raw_aligned/continuous.npy, "
                    f"not processed/continuous.npy."
                )
    # Check GCS: raw range 3-15; z-scored ~-2 to +1
    if "gcs" in idx:
        gcs_vals = continuous[:, :, idx["gcs"]]
        observed = gcs_vals[np.isfinite(gcs_vals)]
        if len(observed) > 100:
            gcs_max = float(np.max(observed))
            if gcs_max < 16 and float(np.median(observed)) < 3.0:
                raise ValueError(
                    f"SOFA computation requires RAW clinical units, but GCS max={gcs_max:.2f} "
                    f"looks like z-score standardized data."
                )


def compute_organ_scores(
    continuous: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    horizon: int = 24,
) -> pd.DataFrame:
    """
    Compute per-patient organ dysfunction scores from RAW S0 physiology.

    IMPORTANT: `continuous` MUST be in original clinical units
    (raw_aligned/continuous.npy), NOT z-score standardized data.

    Uses the worst (maximum) value in the first `horizon` hours
    for deterioration-sensitive variables, mean for MAP.

    Parameters
    ----------
    continuous: (N, T, F) RAW physiology tensor (original clinical units)
    masks: (N, T, F) observation masks
    feature_names: list of feature names matching dim=-1
    horizon: hours to aggregate over (default: first 24h)

    Returns
    -------
    DataFrame with columns: respiratory, cardiovascular, hepatic,
                            renal, coagulation, neurological, sofa_total,
                            dominant_organ

    Raises
    ------
    ValueError: if input appears to be z-score standardized
    """
    _validate_raw_units(continuous, feature_names)

    idx = {name: i for i, name in enumerate(feature_names)}
    N = continuous.shape[0]
    T_use = min(horizon, continuous.shape[1])

    records = []
    for i in range(N):
        # Extract relevant channels, masked mean/max over horizon
        def _masked_stat(feat, stat="mean"):
            if feat not in idx:
                return np.nan
            vals = continuous[i, :T_use, idx[feat]]
            m = masks[i, :T_use, idx[feat]]
            observed = vals[m > 0.5]
            if len(observed) == 0:
                return np.nan
            return float(np.mean(observed)) if stat == "mean" else float(np.max(observed))

        def _masked_min(feat):
            if feat not in idx:
                return np.nan
            vals = continuous[i, :T_use, idx[feat]]
            m = masks[i, :T_use, idx[feat]]
            observed = vals[m > 0.5]
            if len(observed) == 0:
                return np.nan
            return float(np.min(observed))

        # PaO2/FiO2 ratio: use worst (minimum)
        pao2 = _masked_min("pao2")
        fio2 = _masked_stat("fio2", "max")
        if not np.isnan(pao2) and not np.isnan(fio2) and fio2 > 0.01:
            pf_ratio = pao2 / fio2
        else:
            pf_ratio = np.nan

        mean_map = _masked_stat("map", "mean")
        max_bili = _masked_stat("bilirubin", "max")
        max_creat = _masked_stat("creatinine", "max")
        min_plt = _masked_min("platelet")
        min_gcs = _masked_min("gcs")

        resp = _sofa_respiratory(pf_ratio)
        cardio = _sofa_cardiovascular(mean_map)
        hepat = _sofa_hepatic(max_bili)
        renal = _sofa_renal(max_creat)
        coag = _sofa_coagulation(min_plt)
        neuro = _sofa_neurological(min_gcs)
        total = resp + cardio + hepat + renal + coag + neuro

        # Dominant organ = highest sub-score (ties broken by clinical priority)
        organ_map = {
            "respiratory": resp,
            "cardiovascular": cardio,
            "hepatic": hepat,
            "renal": renal,
            "coagulation": coag,
            "neurological": neuro,
        }
        dominant = max(organ_map, key=organ_map.get)

        records.append({
            "respiratory": resp,
            "cardiovascular": cardio,
            "hepatic": hepat,
            "renal": renal,
            "coagulation": coag,
            "neurological": neuro,
            "sofa_total": total,
            "dominant_organ": dominant,
        })

    return pd.DataFrame(records)


# ============================================================
# Clinical Phenotype Assignment Rules
# ============================================================

# Phenotype name vocabulary (mechanism-based, NOT risk-ordered)
PHENOTYPE_NAMES = {
    "hemodynamic_unstable_proxy_responsive": "Hemodynamic Instability – Proxy-Responsive",
    "hemodynamic_unstable_proxy_refractory": "Hemodynamic Instability – Proxy-Refractory",
    "hemodynamic_unstable_proxy_responsive_critical": "Hemodynamic Instability – Proxy-Responsive Critical",
    "hemodynamic_unstable_proxy_responsive_recovering": "Hemodynamic Instability – Proxy-Responsive Recovering",
    "respiratory_failure": "Respiratory Failure Dominant",
    "respiratory_failure_critical": "Respiratory Failure – Critical Pattern",
    "respiratory_failure_recovering": "Respiratory Failure – Recovering Pattern",
    "hepatorenal_dysfunction": "Hepato-Renal Dysfunction",
    "coagulopathy_dominant": "Coagulopathy Dominant",
    "neurological_decline": "Neurological Decline",
    "mild_organ_stable": "Mild / Stable Organ Function",
    "multi_organ_deteriorating": "Multi-Organ Deteriorating",
}


def apply_cluster_severity_modifier(
    phenotype_key: str,
    dominant_cluster: int,
    cluster_mortality_order: dict[int, float],
    thresholds: dict | None = None,
) -> str:
    """
    Add a severity modifier for selected phenotypes based on dominant cluster risk tier.

    The highest-mortality dominant cluster is tagged as ``_critical`` and the
    lowest-mortality dominant cluster is tagged as ``_recovering``. This keeps
    the organ-level mechanism label but restores the strong mortality separation
    already present in the temporal cluster trajectories.
    """
    cfg = resolve_thresholds(thresholds)
    targets = {str(key) for key in cfg.get("severity_split_targets", [])}
    if phenotype_key not in targets or len(cluster_mortality_order) < 2:
        return phenotype_key

    ranked_clusters = sorted(
        ((int(cluster_id), float(mortality)) for cluster_id, mortality in cluster_mortality_order.items()),
        key=lambda item: item[1],
    )
    recovering_cluster = ranked_clusters[0][0]
    critical_cluster = ranked_clusters[-1][0]

    if int(dominant_cluster) == critical_cluster:
        return f"{phenotype_key}_critical"
    if int(dominant_cluster) == recovering_cluster:
        return f"{phenotype_key}_recovering"
    return phenotype_key


def assign_phenotype_by_causality(
    cluster_id: int,
    cate_score: float,
    mortality_risk: float,
    organ_scores: dict,
    trajectory_direction: str = "stable",
    thresholds: dict | None = None,
) -> str:
    """
    Mechanism-based phenotype name from cluster + causal + organ evidence.

    Parameters
    ----------
    cluster_id : int
        KMeans-assigned cluster (0..K-1).
    cate_score : float
        Conditional average treatment effect from causal analysis.
        >0 means treatment reduces mortality (responsive).
        <0 means treatment associated with higher mortality (refractory/harmful).
    mortality_risk : float
        Baseline mortality risk (against 14.2% cohort average).
    organ_scores : dict
        Keys: respiratory, cardiovascular, hepatic, renal, coagulation,
              neurological, sofa_total, dominant_organ
    trajectory_direction : str
        One of: "stable", "improving", "deteriorating"

    Returns
    -------
    str : mechanism-based phenotype name key
    """
    cfg = resolve_thresholds(thresholds)
    sofa = organ_scores.get("sofa_total", 0)
    dominant = organ_scores.get("dominant_organ", "cardiovascular")
    cardio = organ_scores.get("cardiovascular", 0)
    resp = organ_scores.get("respiratory", 0)
    hepat = organ_scores.get("hepatic", 0)
    renal = organ_scores.get("renal", 0)
    coag = organ_scores.get("coagulation", 0)
    neuro = organ_scores.get("neurological", 0)

    # Rule 1: Low severity, stable trajectory → mild/stable
    if (
        sofa <= cfg["low_severity_sofa_max"]
        and mortality_risk < cfg["low_severity_mortality_max"]
        and trajectory_direction != "deteriorating"
    ):
        return "mild_organ_stable"

    # Rule 2: Multi-organ deterioration (≥3 organs score ≥1, worsening trajectory)
    organs_involved = sum(1 for s in [resp, cardio, hepat, renal, coag, neuro] if s >= 1)
    if (
        organs_involved >= int(cfg["multi_organ_min_organs"])
        and trajectory_direction == "deteriorating"
    ):
        return "multi_organ_deteriorating"

    # Rule 3: Cardiovascular dominant
    if (
        dominant == "cardiovascular"
        or (cardio >= 1 and mortality_risk >= cfg["high_mortality_reference"])
    ):
        if cate_score > cfg["cate_responsiveness_threshold"]:
            return "hemodynamic_unstable_proxy_responsive"
        else:
            return "hemodynamic_unstable_proxy_refractory"

    # Rule 4: Respiratory failure dominant
    if dominant == "respiratory" and resp >= 2:
        return "respiratory_failure"

    # Rule 5: Hepato-renal syndrome
    if (hepat >= 2 or renal >= 2) and (hepat + renal >= 3):
        return "hepatorenal_dysfunction"

    # Rule 6: Coagulopathy
    if dominant == "coagulation" and coag >= 2:
        return "coagulopathy_dominant"

    # Rule 7: Neurological decline
    if dominant == "neurological" and neuro >= 2:
        return "neurological_decline"

    # Rule 8: Fallback — use CATE + mortality to determine proxy-responsiveness
    if mortality_risk >= cfg["high_mortality_reference"]:
        if cate_score > cfg["cate_responsiveness_threshold"]:
            return "hemodynamic_unstable_proxy_responsive"
        return "hemodynamic_unstable_proxy_refractory"

    return "mild_organ_stable"


def classify_trajectory_direction(
    window_labels: np.ndarray,
    cluster_mortality_order: dict[int, float],
    slope_threshold: float = DEFAULT_THRESHOLDS["deterioration_slope_threshold"],
) -> np.ndarray:
    """
    Classify each patient's trajectory as stable/improving/deteriorating.

    Uses the mortality ordering of clusters to determine direction:
    if a patient moves toward higher-mortality clusters → deteriorating.

    Parameters
    ----------
    window_labels: (N, W) cluster assignments per window
    cluster_mortality_order: {cluster_id: mortality_rate}

    Returns
    -------
    directions: (N,) array of strings
    """
    N, W = window_labels.shape
    directions = []

    for i in range(N):
        traj = window_labels[i]
        if np.all(traj == traj[0]):
            directions.append("stable")
            continue

        # Compute mortality-risk trajectory slope
        risks = np.array([cluster_mortality_order.get(int(c), 0.142) for c in traj])
        slope = np.polyfit(np.arange(W), risks, 1)[0]

        if slope > slope_threshold:
            directions.append("deteriorating")
        elif slope < -slope_threshold:
            directions.append("improving")
        else:
            directions.append("stable")

    return np.array(directions)


def assign_all_phenotypes(
    window_labels: np.ndarray,
    organ_scores_df: pd.DataFrame,
    cate_scores: np.ndarray,
    mortality_risks: np.ndarray,
    cluster_mortality_order: dict[int, float],
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Assign mechanism-based phenotype names to all patients.

    Parameters
    ----------
    window_labels: (N, W) KMeans cluster labels
    organ_scores_df: DataFrame from compute_organ_scores()
    cate_scores: (N,) CATE from causal analysis
    mortality_risks: (N,) predicted mortality risk
    cluster_mortality_order: {cluster_id: mortality_rate}

    Returns
    -------
    DataFrame with columns: patient_idx, dominant_cluster, trajectory_direction,
                            phenotype_key, phenotype_name, cate_score,
                            mortality_risk, sofa_total, dominant_organ
    """
    cfg = resolve_thresholds(thresholds)
    N = window_labels.shape[0]

    # Step 1: trajectory directions
    directions = classify_trajectory_direction(
        window_labels,
        cluster_mortality_order,
        slope_threshold=cfg["deterioration_slope_threshold"],
    )

    # Step 2: dominant cluster (mode across windows)
    dominant_clusters = []
    for i in range(N):
        vals, counts = np.unique(window_labels[i], return_counts=True)
        dominant_clusters.append(int(vals[np.argmax(counts)]))

    # Step 3: assign phenotype per patient
    results = []
    for i in range(N):
        organ = organ_scores_df.iloc[i].to_dict()
        phenotype_key = assign_phenotype_by_causality(
            cluster_id=dominant_clusters[i],
            cate_score=float(cate_scores[i]) if not np.isnan(cate_scores[i]) else 0.0,
            mortality_risk=float(mortality_risks[i]),
            organ_scores=organ,
            trajectory_direction=directions[i],
            thresholds=cfg,
        )
        phenotype_key = apply_cluster_severity_modifier(
            phenotype_key=phenotype_key,
            dominant_cluster=dominant_clusters[i],
            cluster_mortality_order=cluster_mortality_order,
            thresholds=cfg,
        )
        results.append({
            "patient_idx": i,
            "dominant_cluster": dominant_clusters[i],
            "trajectory_direction": directions[i],
            "phenotype_key": phenotype_key,
            "phenotype_name": PHENOTYPE_NAMES.get(phenotype_key, phenotype_key),
            "cate_score": float(cate_scores[i]) if not np.isnan(cate_scores[i]) else 0.0,
            "mortality_risk": float(mortality_risks[i]),
            "sofa_total": organ.get("sofa_total", 0),
            "dominant_organ": organ.get("dominant_organ", "unknown"),
        })

    df = pd.DataFrame(results)
    logger.info(
        "Phenotype distribution:\n%s",
        df["phenotype_key"].value_counts().to_string(),
    )
    return df
