"""
causal_phenotyping.py - End-to-end causal treatment effect pipeline for phenotype naming.

Integrates CausalML (uber/causalml) with the existing s4/ causal analysis framework
and the new phenotype naming module to produce clinically interpretable phenotype labels.

Pipeline:
  1. Load S2 window labels + S0 physiology + S0 proxy treatments
  2. Build causal analysis frame using proxy treatment indicators
  3. Estimate heterogeneous treatment effects (HTE) via CausalML T-Learner
  4. Compute organ dysfunction scores from raw physiology
  5. Assign mechanism-based phenotype names
  6. Generate validation report
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s0.schema import CONTINUOUS_NAMES, N_CONTINUOUS
from s6_optimization.phenotype_naming import (
    assign_all_phenotypes,
    compute_organ_scores,
    classify_trajectory_direction,
    PHENOTYPE_NAMES,
)

logger = logging.getLogger("s6.causal_phenotyping")


def estimate_cate_with_causalml(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    method: str = "t_learner",
) -> np.ndarray:
    """
    Estimate Conditional Average Treatment Effect using CausalML.

    Falls back to sklearn-based DML if CausalML fails.

    Parameters
    ----------
    X: (N, D) covariate matrix
    treatment: (N,) binary treatment indicator
    outcome: (N,) binary or continuous outcome
    method: 't_learner' or 'x_learner'

    Returns
    -------
    cate: (N,) individual treatment effect estimates
    """
    # Impute and scale
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_clean = pipe.fit_transform(X)
    treatment = np.asarray(treatment, dtype=int)
    outcome = np.asarray(outcome, dtype=float)

    # Remove NaN outcomes
    valid = ~np.isnan(outcome)
    X_clean = X_clean[valid]
    treatment = treatment[valid]
    outcome = outcome[valid]

    try:
        if method == "t_learner":
            from causalml.inference.meta import BaseTRegressor
            learner = BaseTRegressor(
                learner=HistGradientBoostingClassifier(
                    max_depth=5, max_iter=200, random_state=42,
                ),
                control_name=0,
            )
        else:
            from causalml.inference.meta import BaseXRegressor
            learner = BaseXRegressor(
                learner=HistGradientBoostingClassifier(
                    max_depth=5, max_iter=200, random_state=42,
                ),
                control_name=0,
            )

        cate = learner.fit_predict(
            X=X_clean,
            treatment=treatment.astype(str),
            y=outcome,
        )
        # fit_predict returns (N, n_treatment_groups) — take first column
        if cate.ndim == 2:
            cate = cate[:, 0]
        logger.info("CausalML %s: CATE mean=%.4f, std=%.4f", method, cate.mean(), cate.std())
        return cate

    except Exception as e:
        logger.warning("CausalML failed (%s), falling back to DML: %s", method, e)
        return _fallback_dml_cate(X_clean, treatment, outcome)


def _fallback_dml_cate(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    """Sklearn-based doubly-robust DML fallback (same as s4/causal_analysis.py)."""
    # Propensity model
    ps_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    ps_model.fit(X, treatment)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.05, 0.95)

    # Outcome model
    X_aug = np.column_stack([X, treatment])
    outcome_model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    outcome_model.fit(X_aug, outcome)
    mu1 = outcome_model.predict(np.column_stack([X, np.ones(len(X))]))
    mu0 = outcome_model.predict(np.column_stack([X, np.zeros(len(X))]))
    mu = outcome_model.predict(X_aug)

    # Doubly-robust pseudo-outcome
    pseudo = ((treatment - ps) / (ps * (1.0 - ps) + 1e-8)) * (outcome - mu) + (mu1 - mu0)

    # CATE model
    tau_model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    tau_model.fit(X, pseudo)
    cate = tau_model.predict(X)
    logger.info("DML fallback: CATE mean=%.4f, std=%.4f", cate.mean(), cate.std())
    return cate


def build_physiology_covariates(
    continuous: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    horizon: int = 24,
) -> np.ndarray:
    """
    Build patient-level covariate matrix from raw physiology for causal analysis.

    For each important feature, computes mean/min/max over `horizon` hours.
    Also includes observation density as a covariate.

    Returns
    -------
    X: (N, D) covariate matrix
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    N = continuous.shape[0]
    T_use = min(horizon, continuous.shape[1])

    important_features = [
        "heart_rate", "map", "resp_rate", "spo2",
        "creatinine", "lactate", "bilirubin", "wbc",
        "platelet", "gcs", "temperature",
    ]

    covariates = []
    for name in important_features:
        if name not in idx:
            continue
        fi = idx[name]
        vals = continuous[:, :T_use, fi]
        m = masks[:, :T_use, fi]

        # Masked statistics
        masked_vals = np.where(m > 0.5, vals, np.nan)
        with np.errstate(all="ignore"):
            covariates.append(np.nanmean(masked_vals, axis=1))
            covariates.append(np.nanmin(masked_vals, axis=1))
            covariates.append(np.nanmax(masked_vals, axis=1))

    # Overall observation density
    covariates.append(masks[:, :T_use, :].mean(axis=(1, 2)))

    X = np.column_stack(covariates)
    X = np.nan_to_num(X, nan=0.0)
    return X


def run_causal_phenotyping_pipeline(
    s0_dir: Path,
    s2_dir: Path,
    output_dir: Path,
    splits_path: Path | None = None,
    causal_method: str = "t_learner",
    treatment_horizon: int = 24,
    organ_horizon: int = 24,
    phenotype_config: dict | None = None,
) -> dict:
    """
    End-to-end causal phenotyping pipeline.

    Loads existing S0 data and S2 clustering results, runs causal analysis,
    and produces mechanism-based phenotype assignments.

    Parameters
    ----------
    s0_dir: Path to S0 data directory
    s2_dir: Path to S2 data directory (with window_labels.npy)
    output_dir: Where to save results
    splits_path: Path to splits.json
    causal_method: 't_learner' or 'x_learner'
    treatment_horizon: hours to aggregate treatment exposure
    organ_horizon: hours to aggregate organ dysfunction and covariates
    phenotype_config: threshold overrides for naming rules

    Returns
    -------
    report: dict with all results and paths
    """
    s0_dir = Path(s0_dir)
    s2_dir = Path(s2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting causal phenotyping pipeline")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    # processed/ for ML covariates (z-score standardized)
    continuous_processed = np.load(s0_dir / "processed" / "continuous.npy", mmap_mode="r")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
    # raw_aligned/ for SOFA organ scores (original clinical units)
    raw_continuous = np.load(s0_dir / "raw_aligned" / "continuous.npy", mmap_mode="r")
    raw_masks = np.load(s0_dir / "raw_aligned" / "masks_continuous.npy", mmap_mode="r")
    static = pd.read_csv(s0_dir / "static.csv")
    window_labels = np.load(s2_dir / "window_labels.npy")

    N = continuous_processed.shape[0]
    logger.info("  Loaded: N=%d patients, T=%d hours, F=%d features",
                N, continuous_processed.shape[1], continuous_processed.shape[2])
    logger.info("  raw_aligned loaded for SOFA: shape=%s", raw_continuous.shape)

    # Load splits
    if splits_path is None:
        splits_path = s0_dir / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    # Step 2: Compute organ dysfunction scores FROM RAW DATA
    logger.info("Step 2: Computing organ dysfunction scores (SOFA from RAW units)...")
    organ_scores_df = compute_organ_scores(
        continuous=np.array(raw_continuous),
        masks=np.array(raw_masks),
        feature_names=CONTINUOUS_NAMES,
        horizon=organ_horizon,
    )
    logger.info("  SOFA total: mean=%.2f, max=%d",
                organ_scores_df["sofa_total"].mean(),
                organ_scores_df["sofa_total"].max())
    logger.info("  Dominant organs:\n%s",
                organ_scores_df["dominant_organ"].value_counts().to_string())

    # Step 3: Build covariates for causal analysis (from PROCESSED/standardized data)
    logger.info("Step 3: Building physiology covariates (from processed z-score data)...")
    X_covariates = build_physiology_covariates(
        continuous=np.array(continuous_processed),
        masks=np.array(masks),
        feature_names=CONTINUOUS_NAMES,
        horizon=organ_horizon,
    )
    logger.info("  Covariates shape: %s", X_covariates.shape)

    # Step 4: Extract proxy treatment indicator
    # Use vasopressor_proxy from S0 proxy indicators (MAP < 65)
    logger.info("Step 4: Extracting proxy treatment indicator...")
    proxy_path = s0_dir / "processed" / "proxy_indicators.npy"
    if proxy_path.exists():
        proxy = np.load(proxy_path, mmap_mode="r")
        # Channel 0 = vasopressor_proxy (MAP < 65)
        treatment_indicator = (proxy[:, :treatment_horizon, 0] > 0.5).any(axis=1).astype(int)
        logger.info("  Treatment exposure (proxy vasopressor): %d/%d (%.1f%%)",
                    treatment_indicator.sum(), N, 100 * treatment_indicator.mean())
    else:
        # Fallback: use MAP < 65 directly
        map_idx = CONTINUOUS_NAMES.index("map")
        map_vals = continuous[:, :treatment_horizon, map_idx]
        treatment_indicator = (map_vals < 65).any(axis=1).astype(int)
        logger.info("  Treatment (MAP<65 fallback): %d/%d", treatment_indicator.sum(), N)

    # Step 5: Get outcome
    outcome_col = "mortality_inhospital"
    outcome = static[outcome_col].fillna(0).values.astype(float)
    logger.info("  Outcome: %s, rate=%.3f", outcome_col, outcome.mean())

    # Step 6: Estimate CATE
    logger.info("Step 5: Estimating CATE via %s...", causal_method)
    cate_scores = estimate_cate_with_causalml(
        X=X_covariates,
        treatment=treatment_indicator,
        outcome=outcome,
        method=causal_method,
    )

    # Align cate_scores length (in case of NaN filtering)
    if len(cate_scores) < N:
        full_cate = np.zeros(N, dtype=np.float32)
        valid_mask = ~np.isnan(outcome)
        full_cate[valid_mask] = cate_scores
        cate_scores = full_cate

    # Step 7: Compute cluster mortality ordering
    logger.info("Step 6: Computing cluster mortality ordering...")
    cluster_mortality_order = {}
    dominant_clusters = []
    for i in range(N):
        vals, counts = np.unique(window_labels[i], return_counts=True)
        dominant_clusters.append(int(vals[np.argmax(counts)]))

    for c in range(int(window_labels.max()) + 1):
        mask = np.array(dominant_clusters) == c
        if mask.sum() > 0:
            cluster_mortality_order[c] = float(outcome[mask].mean())
            logger.info("  Cluster %d: n=%d, mortality=%.3f",
                        c, mask.sum(), cluster_mortality_order[c])

    # Step 8: Mortality risk estimates (simple logistic regression)
    logger.info("Step 7: Computing mortality risk estimates...")
    risk_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    train_idx = splits["train"]
    risk_model.fit(X_covariates[train_idx], outcome[train_idx])
    mortality_risks = risk_model.predict_proba(X_covariates)[:, 1]

    # Step 9: Assign phenotypes
    logger.info("Step 8: Assigning mechanism-based phenotype names...")
    phenotype_df = assign_all_phenotypes(
        window_labels=window_labels,
        organ_scores_df=organ_scores_df,
        cate_scores=cate_scores,
        mortality_risks=mortality_risks,
        cluster_mortality_order=cluster_mortality_order,
        thresholds=phenotype_config,
    )

    # Step 10: Validation — mortality by phenotype
    logger.info("Step 9: Validating phenotype assignments...")
    phenotype_df["mortality_actual"] = outcome
    phenotype_df["center_id"] = static["center_id"].values if "center_id" in static.columns else "unknown"

    validation = {}
    for key in phenotype_df["phenotype_key"].unique():
        mask = phenotype_df["phenotype_key"] == key
        sub = phenotype_df[mask]
        validation[key] = {
            "n": int(mask.sum()),
            "fraction": round(mask.mean(), 4),
            "mortality_rate": round(float(sub["mortality_actual"].mean()), 4),
            "mean_sofa": round(float(sub["sofa_total"].mean()), 2),
            "mean_cate": round(float(sub["cate_score"].mean()), 4),
        }
    logger.info("  Phenotype validation:\n%s",
                json.dumps(validation, indent=2))

    # Step 11: Cross-center consistency check
    if "center_id" in static.columns:
        center_validation = {}
        for center in phenotype_df["center_id"].unique():
            center_mask = phenotype_df["center_id"] == center
            center_sub = phenotype_df[center_mask]
            center_validation[str(center)] = {
                "n": int(center_mask.sum()),
                "phenotype_distribution": center_sub["phenotype_key"].value_counts().to_dict(),
                "mortality_by_phenotype": {
                    k: round(float(center_sub[center_sub["phenotype_key"] == k]["mortality_actual"].mean()), 4)
                    for k in center_sub["phenotype_key"].unique()
                    if (center_sub["phenotype_key"] == k).sum() > 0
                },
            }
    else:
        center_validation = {}

    # Save results
    phenotype_df.to_csv(output_dir / "phenotype_assignments.csv", index=False)
    organ_scores_df.to_csv(output_dir / "organ_scores.csv", index=False)
    np.save(output_dir / "cate_scores.npy", cate_scores.astype(np.float32))

    report = {
        "pipeline": "causal_phenotyping",
        "n_patients": N,
        "causal_method": causal_method,
        "treatment_horizon": int(treatment_horizon),
        "organ_horizon": int(organ_horizon),
        "phenotype_config": phenotype_config or {},
        "treatment_exposure_rate": round(float(treatment_indicator.mean()), 4),
        "outcome_rate": round(float(outcome.mean()), 4),
        "cate_summary": {
            "mean": round(float(cate_scores.mean()), 4),
            "std": round(float(cate_scores.std()), 4),
            "q10": round(float(np.quantile(cate_scores, 0.10)), 4),
            "q50": round(float(np.quantile(cate_scores, 0.50)), 4),
            "q90": round(float(np.quantile(cate_scores, 0.90)), 4),
        },
        "phenotype_validation": validation,
        "center_validation": center_validation,
        "cluster_mortality_order": {str(k): round(v, 4) for k, v in cluster_mortality_order.items()},
        "artifacts": {
            "phenotype_assignments": str(output_dir / "phenotype_assignments.csv"),
            "organ_scores": str(output_dir / "organ_scores.csv"),
            "cate_scores": str(output_dir / "cate_scores.npy"),
        },
    }

    report_path = output_dir / "causal_phenotyping_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)

    logger.info("=" * 60)
    logger.info("Causal phenotyping pipeline complete. Report: %s", report_path)
    logger.info("=" * 60)

    return report
