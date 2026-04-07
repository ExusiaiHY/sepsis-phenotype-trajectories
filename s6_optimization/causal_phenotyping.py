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
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s0.schema import CONTINUOUS_NAMES
from s6_optimization.domain_adaptation import align_covariates_by_group
from s6_optimization.dowhy_validation import run_dowhy_validation
from s6_optimization.phenotype_naming import (
    assign_all_phenotypes,
    compute_organ_scores,
    PHENOTYPE_NAMES,
)
from s6_optimization.saits_imputation import run_saits_imputation
from s6_optimization.timesfm_features import run_timesfm_feature_extraction

logger = logging.getLogger("s6.causal_phenotyping")


def _prepare_causal_inputs(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return X_clean, treatment, outcome, valid


def _estimate_causalml_on_clean_inputs(
    X_clean: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    method: str = "t_learner",
) -> np.ndarray:
    if method == "t_learner":
        from causalml.inference.meta import BaseTRegressor
        learner = BaseTRegressor(
            learner=HistGradientBoostingRegressor(
                max_depth=5, max_iter=200, random_state=42,
            ),
            control_name=0,
        )
    else:
        from causalml.inference.meta import BaseXRegressor
        learner = BaseXRegressor(
            learner=HistGradientBoostingRegressor(
                max_depth=5, max_iter=200, random_state=42,
            ),
            control_name=0,
        )

    cate = learner.fit_predict(
        X=X_clean,
        treatment=treatment,
        y=outcome,
    )
    if cate.ndim == 2:
        cate = cate[:, 0]
    return np.asarray(cate, dtype=np.float64)


def summarize_cate(cate: np.ndarray) -> dict:
    cate = np.asarray(cate, dtype=np.float64)
    q10 = float(np.quantile(cate, 0.10))
    q50 = float(np.quantile(cate, 0.50))
    q90 = float(np.quantile(cate, 0.90))
    return {
        "mean": round(float(cate.mean()), 4),
        "std": round(float(cate.std()), 4),
        "q10": round(q10, 4),
        "q50": round(q50, 4),
        "q90": round(q90, 4),
        "abs_q90": round(float(max(abs(q10), abs(q90))), 4),
    }


def _evaluate_stability_gate(summary: dict, config: dict | None = None) -> tuple[bool, str | None]:
    cfg = {
        "enabled": True,
        "max_std": 0.12,
        "max_abs_q90": 0.15,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})
    if not cfg.get("enabled", True):
        return False, None

    reasons = []
    if float(summary["std"]) > float(cfg["max_std"]):
        reasons.append(f"std={summary['std']:.4f} > max_std={float(cfg['max_std']):.4f}")
    if float(summary["abs_q90"]) > float(cfg["max_abs_q90"]):
        reasons.append(
            f"abs_q90={summary['abs_q90']:.4f} > max_abs_q90={float(cfg['max_abs_q90']):.4f}"
        )
    if reasons:
        return True, "; ".join(reasons)
    return False, None


def estimate_cate_with_causalml(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    method: str = "t_learner",
    stability_gate: dict | None = None,
) -> dict:
    """
    Estimate Conditional Average Treatment Effect using CausalML, with an
    optional stability gate that falls back to cross-fitted DML.
    """
    X_clean, treatment_clean, outcome_clean, valid_mask = _prepare_causal_inputs(
        X,
        treatment,
        outcome,
    )

    try:
        causalml_cate = _estimate_causalml_on_clean_inputs(
            X_clean=X_clean,
            treatment=treatment_clean,
            outcome=outcome_clean,
            method=method,
        )
        candidate_summary = summarize_cate(causalml_cate)
        logger.info(
            "CausalML %s candidate: mean=%.4f, std=%.4f, abs_q90=%.4f",
            method,
            candidate_summary["mean"],
            candidate_summary["std"],
            candidate_summary["abs_q90"],
        )
        should_fallback, fallback_reason = _evaluate_stability_gate(
            candidate_summary,
            stability_gate,
        )
        if should_fallback:
            logger.warning(
                "CausalML %s rejected by stability gate; using cross-fitted DML (%s)",
                method,
                fallback_reason,
            )
            selected_cate = _fallback_dml_cate(X_clean, treatment_clean, outcome_clean)
            estimator_selected = "cross_fitted_dml"
        else:
            selected_cate = causalml_cate
            estimator_selected = f"causalml_{method}"
    except Exception as exc:
        logger.warning("CausalML failed (%s), falling back to DML: %s", method, exc)
        candidate_summary = None
        fallback_reason = f"causalml_exception: {exc}"
        selected_cate = _fallback_dml_cate(X_clean, treatment_clean, outcome_clean)
        estimator_selected = "cross_fitted_dml"

    full_cate = np.zeros(len(X), dtype=np.float32)
    full_cate[valid_mask] = np.asarray(selected_cate, dtype=np.float32)
    return {
        "cate": full_cate,
        "valid_mask": valid_mask,
        "estimator_selected": estimator_selected,
        "candidate_summary": candidate_summary,
        "selected_summary": summarize_cate(selected_cate),
        "fallback_reason": fallback_reason if estimator_selected == "cross_fitted_dml" else None,
    }


def _fallback_dml_cate(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Sklearn-based doubly-robust DML with honest 5-fold cross-fitting.

    Unlike in-sample fitting, each sample's CATE is predicted by a model
    that never saw that sample during training.
    """
    from sklearn.model_selection import StratifiedKFold

    N = len(X)
    cate = np.zeros(N, dtype=np.float64)
    if len(np.unique(treatment)) < 2:
        logger.warning("DML cross-fitting skipped: treatment has only one class")
        return cate
    class_counts = np.bincount(treatment.astype(int))
    min_class_count = int(class_counts[class_counts > 0].min())
    if min_class_count < 2:
        logger.warning(
            "DML cross-fitting skipped: smallest treatment class has only %d sample(s)",
            min_class_count,
        )
        return cate

    effective_folds = min(n_folds, min_class_count)
    splitter = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=random_state,
    )

    for train_idx, test_idx in splitter.split(X, treatment):
        X_tr, X_te = X[train_idx], X[test_idx]
        w_tr, w_te = treatment[train_idx], treatment[test_idx]
        y_tr = outcome[train_idx]

        # Propensity model
        ps_model = LogisticRegression(max_iter=2000, class_weight="balanced")
        ps_model.fit(X_tr, w_tr)
        ps_te = np.clip(ps_model.predict_proba(X_te)[:, 1], 0.05, 0.95)

        # Outcome model
        X_aug_tr = np.column_stack([X_tr, w_tr])
        outcome_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, random_state=random_state, n_jobs=-1,
        )
        outcome_model.fit(X_aug_tr, y_tr)
        X_aug_te = np.column_stack([X_te, w_te])
        mu_te = outcome_model.predict(X_aug_te)
        mu1_te = outcome_model.predict(np.column_stack([X_te, np.ones(len(X_te))]))
        mu0_te = outcome_model.predict(np.column_stack([X_te, np.zeros(len(X_te))]))

        # CATE model: fit on train pseudo-outcomes, predict on test
        X_aug_tr_full = np.column_stack([X_tr, w_tr])
        mu_tr = outcome_model.predict(X_aug_tr_full)
        ps_tr = np.clip(ps_model.predict_proba(X_tr)[:, 1], 0.05, 0.95)
        mu1_tr = outcome_model.predict(np.column_stack([X_tr, np.ones(len(X_tr))]))
        mu0_tr = outcome_model.predict(np.column_stack([X_tr, np.zeros(len(X_tr))]))
        pseudo_tr = ((w_tr - ps_tr) / (ps_tr * (1.0 - ps_tr) + 1e-8)) * (y_tr - mu_tr) + (mu1_tr - mu0_tr)

        tau_model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, random_state=random_state, n_jobs=-1,
        )
        tau_model.fit(X_tr, pseudo_tr)
        cate[test_idx] = tau_model.predict(X_te)

    logger.info(
        "DML cross-fitted (%d-fold): CATE mean=%.4f, std=%.4f",
        effective_folds,
        cate.mean(),
        cate.std(),
    )
    return cate


def build_physiology_covariates(
    continuous: np.ndarray,
    masks: np.ndarray,
    feature_names: list[str],
    horizon: int = 24,
    imputed_continuous: np.ndarray | None = None,
    extra_features: pd.DataFrame | None = None,
) -> np.ndarray:
    """
    Build patient-level covariate matrix from processed physiology for causal analysis.

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
        if imputed_continuous is not None:
            vals = imputed_continuous[:, :T_use, fi]
            covariates.append(np.mean(vals, axis=1))
            covariates.append(np.min(vals, axis=1))
            covariates.append(np.max(vals, axis=1))
        else:
            vals = continuous[:, :T_use, fi]
            m = masks[:, :T_use, fi]

            masked_vals = np.where(m > 0.5, vals, np.nan)
            with np.errstate(all="ignore"):
                covariates.append(np.nanmean(masked_vals, axis=1))
                covariates.append(np.nanmin(masked_vals, axis=1))
                covariates.append(np.nanmax(masked_vals, axis=1))

    # Overall observation density
    covariates.append(masks[:, :T_use, :].mean(axis=(1, 2)))
    if extra_features is not None and not extra_features.empty:
        covariates.append(extra_features.to_numpy(dtype=np.float32))

    X = np.column_stack(covariates)
    X = np.nan_to_num(X, nan=0.0)
    return X


def run_causal_phenotyping_pipeline(
    s0_dir: Path,
    s2_dir: Path,
    output_dir: Path,
    splits_path: Path | None = None,
    missingness_features: pd.DataFrame | None = None,
    missingness_feature_summary: dict | None = None,
    causal_method: str = "t_learner",
    causal_config: dict | None = None,
    treatment_horizon: int = 24,
    organ_horizon: int = 24,
    phenotype_config: dict | None = None,
    imputation_config: dict | None = None,
    dowhy_config: dict | None = None,
    timesfm_config: dict | None = None,
    domain_adaptation_config: dict | None = None,
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
    causal_cfg = dict(causal_config or {})

    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    # processed/ for ML covariates (z-score standardized)
    processed_continuous_path = s0_dir / "processed" / "continuous.npy"
    processed_masks_path = s0_dir / "processed" / "masks_continuous.npy"
    raw_continuous_path = s0_dir / "raw_aligned" / "continuous.npy"
    raw_masks_path = s0_dir / "raw_aligned" / "masks_continuous.npy"
    missing_inputs = [
        str(path) for path in [
            processed_continuous_path,
            processed_masks_path,
            raw_continuous_path,
            raw_masks_path,
        ] if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "S6 causal phenotyping is missing required inputs: "
            + ", ".join(missing_inputs)
        )

    continuous_processed = np.load(processed_continuous_path, mmap_mode="r")
    masks = np.load(processed_masks_path, mmap_mode="r")
    # raw_aligned/ for SOFA organ scores (original clinical units)
    raw_continuous = np.load(raw_continuous_path, mmap_mode="r")
    raw_masks = np.load(raw_masks_path, mmap_mode="r")
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

    # Step 3: Optional SAITS-based imputation on processed covariates
    saits_bundle = run_saits_imputation(
        continuous=np.array(continuous_processed),
        masks=np.array(masks),
        output_dir=output_dir,
        config=imputation_config,
    )

    # Step 4: Optional TimesFM dynamic features
    logger.info("Step 3: Extracting optional TimesFM dynamic features...")
    timesfm_bundle = run_timesfm_feature_extraction(
        continuous=np.array(continuous_processed),
        masks=np.array(masks),
        feature_names=CONTINUOUS_NAMES,
        output_dir=output_dir,
        config=timesfm_config,
        imputed_continuous=saits_bundle["imputed"] if saits_bundle.get("enabled") else None,
    )

    # Step 5: Build covariates for causal analysis (from processed z-score data)
    logger.info("Step 4: Building physiology covariates (from processed z-score data)...")
    extra_features = []
    if missingness_features is not None and not missingness_features.empty:
        logger.info(
            "  Appending patient-level missingness covariates: shape=%s",
            missingness_features.shape,
        )
        extra_features.append(missingness_features)
    if saits_bundle.get("enabled") and not saits_bundle["features_df"].empty:
        extra_features.append(saits_bundle["features_df"])
    if timesfm_bundle.get("enabled") and not timesfm_bundle["features_df"].empty:
        extra_features.append(timesfm_bundle["features_df"])
    merged_extra_features = pd.concat(extra_features, axis=1) if extra_features else None
    X_covariates = build_physiology_covariates(
        continuous=np.array(continuous_processed),
        masks=np.array(masks),
        feature_names=CONTINUOUS_NAMES,
        horizon=organ_horizon,
        imputed_continuous=saits_bundle["imputed"] if saits_bundle.get("enabled") else None,
        extra_features=merged_extra_features,
    )
    logger.info("  Covariates shape: %s", X_covariates.shape)

    logger.info("Step 4b: Applying optional domain adaptation on covariates...")
    if "center_id" in static.columns:
        domain_bundle = align_covariates_by_group(
            X_covariates,
            static["center_id"].astype(str).values,
            output_dir=output_dir,
            config=domain_adaptation_config,
        )
        X_covariates = domain_bundle["X_aligned"]
    else:
        domain_bundle = {
            "X_aligned": X_covariates,
            "summary": {"enabled": False, "reason": "center_id_unavailable"},
        }

    # Step 6: Extract proxy treatment indicator
    # Use vasopressor_proxy from S0 proxy indicators (MAP < 65)
    logger.info("Step 5: Extracting proxy treatment indicator...")
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
        map_vals = raw_continuous[:, :treatment_horizon, map_idx]
        treatment_indicator = (map_vals < 65).any(axis=1).astype(int)
        logger.info("  Treatment (MAP<65 fallback): %d/%d", treatment_indicator.sum(), N)

    # Step 7: Get outcome
    outcome_col = causal_cfg.get("outcome_col", "mortality_inhospital")
    outcome = static[outcome_col].fillna(0).values.astype(float)
    logger.info("  Outcome: %s, rate=%.3f", outcome_col, outcome.mean())

    # Step 8: Estimate CATE
    logger.info("Step 6: Estimating CATE via %s...", causal_method)
    cate_bundle = estimate_cate_with_causalml(
        X=X_covariates,
        treatment=treatment_indicator,
        outcome=outcome,
        method=causal_method,
        stability_gate=causal_cfg.get("stability_gate"),
    )
    cate_scores = cate_bundle["cate"]

    # Step 9: Compute cluster mortality ordering
    logger.info("Step 7: Computing cluster mortality ordering...")
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

    # Step 10: Mortality risk estimates (simple logistic regression)
    logger.info("Step 8: Computing mortality risk estimates...")
    risk_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    train_idx = splits["train"]
    risk_model.fit(X_covariates[train_idx], outcome[train_idx])
    mortality_risks = risk_model.predict_proba(X_covariates)[:, 1]

    # Step 11: Assign phenotypes
    logger.info("Step 9: Assigning mechanism-based phenotype names...")
    phenotype_df = assign_all_phenotypes(
        window_labels=window_labels,
        organ_scores_df=organ_scores_df,
        cate_scores=cate_scores,
        mortality_risks=mortality_risks,
        cluster_mortality_order=cluster_mortality_order,
        thresholds=phenotype_config,
    )

    # Step 12: Validation — mortality by phenotype
    logger.info("Step 10: Validating phenotype assignments...")
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

    # Step 13: Cross-center consistency check
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

    # Step 14: DoWhy refutation report
    dowhy_report = run_dowhy_validation(
        X_covariates=X_covariates,
        treatment=treatment_indicator,
        outcome=outcome,
        output_dir=output_dir,
        config=dowhy_config,
    )

    # Save results
    phenotype_df.to_csv(output_dir / "phenotype_assignments.csv", index=False)
    organ_scores_df.to_csv(output_dir / "organ_scores.csv", index=False)
    np.save(output_dir / "cate_scores.npy", cate_scores.astype(np.float32))

    report = {
        "pipeline": "causal_phenotyping",
        "n_patients": N,
        "causal_method": causal_method,
        "cate_estimator_selected": cate_bundle["estimator_selected"],
        "causal_stability_gate": causal_cfg.get("stability_gate", {}),
        "causalml_candidate_summary": cate_bundle["candidate_summary"],
        "cate_fallback_reason": cate_bundle["fallback_reason"],
        "data_sources": {
            "covariates": str(processed_continuous_path),
            "organ_scores": str(raw_continuous_path),
        },
        "missingness_covariate_summary": missingness_feature_summary or {"enabled": False},
        "imputation_summary": saits_bundle.get("summary", {"enabled": False}),
        "timesfm_summary": timesfm_bundle.get("summary", {"enabled": False}),
        "domain_adaptation_summary": domain_bundle.get("summary", {"enabled": False}),
        "dowhy_validation": dowhy_report,
        "treatment_horizon": int(treatment_horizon),
        "organ_horizon": int(organ_horizon),
        "phenotype_config": phenotype_config or {},
        "treatment_exposure_rate": round(float(treatment_indicator.mean()), 4),
        "outcome_rate": round(float(outcome.mean()), 4),
        "cate_summary": summarize_cate(cate_scores),
        "phenotype_validation": validation,
        "center_validation": center_validation,
        "cluster_mortality_order": {str(k): round(v, 4) for k, v in cluster_mortality_order.items()},
        "artifacts": {
            "phenotype_assignments": str(output_dir / "phenotype_assignments.csv"),
            "organ_scores": str(output_dir / "organ_scores.csv"),
            "cate_scores": str(output_dir / "cate_scores.npy"),
            "timesfm_features": timesfm_bundle["summary"]["artifacts"]["patient_features"],
            "domain_adaptation_summary": domain_bundle["summary"].get("report_path"),
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
