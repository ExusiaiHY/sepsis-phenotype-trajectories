"""
causal_analysis.py - Stage 4 causal evaluation for treatment-aware phenotypes.

The estimators here are deliberately pragmatic and dependency-light:
  - Propensity score matching via logistic regression + nearest-neighbor match
  - Causal-forest-style DML using random forests on doubly robust pseudo-outcomes
  - Local linear regression discontinuity around clinically meaningful thresholds

All outputs are summarized as JSON-friendly dictionaries so they can be stored
as experiment artifacts alongside the rest of the repo.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from s0.schema import CONTINUOUS_NAMES

logger = logging.getLogger("s4.causal")


def build_causal_frame(
    *,
    cohort_static: pd.DataFrame,
    treatments: np.ndarray,
    treatment_names: list[str],
    continuous: np.ndarray | None = None,
    masks_continuous: np.ndarray | None = None,
    continuous_names: list[str] | None = None,
    embeddings: np.ndarray | None = None,
    phenotype_labels: np.ndarray | None = None,
    note_embeddings: np.ndarray | None = None,
    horizons: tuple[int, ...] = (6, 24),
    patient_id_col: str = "patient_id",
    outcome_col: str | None = None,
    max_embedding_dims: int = 16,
) -> pd.DataFrame:
    """
    Assemble a patient-level causal analysis table from aligned tensors.

    The frame includes:
      - outcome / baseline covariates from `cohort_static`
      - early treatment summaries over multiple horizons
      - physiology summaries from the original S0 continuous tensor
      - compressed representation covariates from embeddings / notes
      - optional phenotype labels
    """
    frame = cohort_static.copy().reset_index(drop=True)
    if patient_id_col not in frame.columns:
        frame[patient_id_col] = np.arange(len(frame)).astype(str)

    if outcome_col is None:
        for candidate in ("mortality_inhospital", "mortality_28d", "hospital_expire_flag"):
            if candidate in frame.columns:
                outcome_col = candidate
                break
    if outcome_col is not None and outcome_col in frame.columns:
        frame[outcome_col] = pd.to_numeric(frame[outcome_col], errors="coerce")

    if phenotype_labels is not None:
        phenotype_arr = np.asarray(phenotype_labels)
        if phenotype_arr.ndim == 2:
            stable = np.all(phenotype_arr == phenotype_arr[:, :1], axis=1)
            dominant = []
            for row in phenotype_arr:
                values, counts = np.unique(row, return_counts=True)
                dominant.append(int(values[np.argmax(counts)]))
            frame["phenotype_dominant"] = dominant
            frame["phenotype_stable"] = stable.astype(int)
            frame["phenotype_first_window"] = phenotype_arr[:, 0].astype(int)
        else:
            frame["phenotype_dominant"] = phenotype_arr.astype(int)

    for horizon in horizons:
        horizon = int(horizon)
        sl = slice(0, horizon)
        for feature_idx, feature_name in enumerate(treatment_names):
            arr = np.asarray(treatments[:, sl, feature_idx], dtype=np.float32)
            if feature_name.endswith("_ml") or feature_name.endswith("_rate"):
                any_exp = (arr > 0).any(axis=1).astype(int)
                total = arr.sum(axis=1)
                peak = arr.max(axis=1)
                frame[f"{feature_name}_any_{horizon}h"] = any_exp
                frame[f"{feature_name}_sum_{horizon}h"] = total
                frame[f"{feature_name}_max_{horizon}h"] = peak
            else:
                active = arr >= 0.5
                frame[f"{feature_name}_any_{horizon}h"] = active.any(axis=1).astype(int)
                frame[f"{feature_name}_hours_{horizon}h"] = active.sum(axis=1)

    if continuous is not None:
        names = continuous_names or CONTINUOUS_NAMES
        idx_map = {name: idx for idx, name in enumerate(names)}
        important = [
            "map",
            "heart_rate",
            "resp_rate",
            "spo2",
            "creatinine",
            "lactate",
            "bilirubin",
            "wbc",
        ]
        for horizon in horizons:
            sl = slice(0, int(horizon))
            for name in important:
                if name not in idx_map:
                    continue
                arr = np.asarray(continuous[:, sl, idx_map[name]], dtype=np.float32)
                frame[f"{name}_mean_{horizon}h"] = arr.mean(axis=1)
                frame[f"{name}_min_{horizon}h"] = arr.min(axis=1)
                frame[f"{name}_max_{horizon}h"] = arr.max(axis=1)
        if masks_continuous is not None:
            for horizon in horizons:
                frame[f"obs_density_{horizon}h"] = masks_continuous[:, : int(horizon), :].mean(axis=(1, 2))

    _append_low_dim_covariates(frame, embeddings, prefix="emb", max_dims=max_embedding_dims)
    _append_low_dim_covariates(frame, note_embeddings, prefix="note", max_dims=min(max_embedding_dims, 8))
    return frame


def estimate_propensity_score_matching(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str],
    caliper: float | None = 0.2,
) -> dict:
    """
    Estimate an average treatment effect via propensity score matching.

    Matching is 1:1 nearest neighbor with replacement on the scalar propensity
    score, optionally filtered by a standard caliper.
    """
    clean, x, y, w = _prepare_causal_inputs(
        df=df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
    )
    if len(np.unique(w)) < 2:
        return {
            "method": "psm",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "n_samples": int(len(clean)),
            "n_treated": int(w.sum()),
            "n_matched": 0,
            "ate": None,
            "note": "single treatment class",
        }

    ps_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    ps_model.fit(x, w)
    ps = np.clip(ps_model.predict_proba(x)[:, 1], 1e-3, 1.0 - 1e-3)
    clean = clean.copy()
    clean["_ps"] = ps

    treated = clean[clean[treatment_col] == 1].copy()
    control = clean[clean[treatment_col] == 0].copy()
    if treated.empty or control.empty:
        return {
            "method": "psm",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "n_samples": int(len(clean)),
            "n_treated": int(w.sum()),
            "n_matched": 0,
            "ate": None,
            "note": "treated or control group missing",
        }

    matcher = NearestNeighbors(n_neighbors=1)
    matcher.fit(control[["_ps"]].to_numpy())
    dist, nn_idx = matcher.kneighbors(treated[["_ps"]].to_numpy())

    if caliper is not None:
        ps_logit = np.log(ps / (1.0 - ps))
        caliper_value = float(caliper) * float(np.nanstd(ps_logit))
        keep = np.abs(treated["_ps"].to_numpy() - control.iloc[nn_idx[:, 0]]["_ps"].to_numpy()) <= caliper_value
    else:
        keep = np.ones(len(treated), dtype=bool)

    matched_treated = treated.loc[keep].reset_index(drop=True)
    matched_control = control.iloc[nn_idx[:, 0][keep]].reset_index(drop=True)
    if matched_treated.empty:
        return {
            "method": "psm",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "n_samples": int(len(clean)),
            "n_treated": int(w.sum()),
            "n_matched": 0,
            "ate": None,
            "note": "no matched pairs within caliper",
        }

    ate = float((matched_treated[outcome_col] - matched_control[outcome_col]).mean())
    return {
        "method": "psm",
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "n_samples": int(len(clean)),
        "n_treated": int(w.sum()),
        "n_control": int((1 - w).sum()),
        "n_matched": int(len(matched_treated)),
        "ate": round(ate, 4),
        "treated_outcome_rate_matched": round(float(matched_treated[outcome_col].mean()), 4),
        "control_outcome_rate_matched": round(float(matched_control[outcome_col].mean()), 4),
        "propensity_mean_treated": round(float(matched_treated["_ps"].mean()), 4),
        "propensity_mean_control": round(float(matched_control["_ps"].mean()), 4),
    }


def estimate_causal_forest_dml(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str],
    effect_modifier_cols: list[str] | None = None,
    phenotype_col: str | None = "phenotype_dominant",
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Estimate heterogeneous treatment effects with a causal-forest-style DML.

    This is not a textbook honest causal forest implementation; it is a
    dependency-light approximation that fits:
      1. a propensity model
      2. an outcome model
      3. a random forest over doubly robust pseudo-outcomes
    """
    effect_modifier_cols = effect_modifier_cols or covariate_cols
    clean, x_cov, y, w = _prepare_causal_inputs(
        df=df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
    )
    if len(np.unique(w)) < 2:
        return {
            "method": "causal_forest_dml",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "n_samples": int(len(clean)),
            "cate_mean": None,
            "note": "single treatment class",
        }

    x_eff = _encode_covariates(clean[effect_modifier_cols])

    ps_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    ps_model.fit(x_cov, w)
    ps = np.clip(ps_model.predict_proba(x_cov)[:, 1], 0.05, 0.95)

    outcome_model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=1,
    )
    x_aug = np.column_stack([x_cov, w])
    outcome_model.fit(x_aug, y)
    mu = outcome_model.predict(x_aug)
    mu1 = outcome_model.predict(np.column_stack([x_cov, np.ones(len(clean))]))
    mu0 = outcome_model.predict(np.column_stack([x_cov, np.zeros(len(clean))]))

    pseudo = ((w - ps) / (ps * (1.0 - ps))) * (y - mu) + (mu1 - mu0)
    tau_model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=1,
    )
    tau_model.fit(x_eff, pseudo)
    cate = tau_model.predict(x_eff)

    result = {
        "method": "causal_forest_dml",
        "note": "random-forest DML proxy for causal forest",
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "n_samples": int(len(clean)),
        "cate_mean": round(float(np.mean(cate)), 4),
        "cate_std": round(float(np.std(cate)), 4),
        "cate_q10": round(float(np.quantile(cate, 0.10)), 4),
        "cate_q50": round(float(np.quantile(cate, 0.50)), 4),
        "cate_q90": round(float(np.quantile(cate, 0.90)), 4),
        "feature_importance_top": _top_feature_importance(tau_model.feature_importances_, list(x_eff.columns)),
    }
    if phenotype_col and phenotype_col in clean.columns:
        tmp = clean[[phenotype_col]].copy()
        tmp["cate"] = cate
        pheno = (
            tmp.groupby(phenotype_col)["cate"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "n", "mean": "cate_mean", "std": "cate_std"})
        )
        result["phenotype_summary"] = pheno.round(4).to_dict(orient="records")
    return result


def estimate_regression_discontinuity(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    running_col: str,
    threshold: float,
    covariate_cols: list[str] | None = None,
    bandwidth: float | None = None,
    treated_when: str = "above",
) -> dict:
    """
    Local linear regression discontinuity around a clinical decision threshold.

    Parameters
    ----------
    treated_when:
        `above` or `below`, indicating on which side of the threshold the
        treatment should become more likely.
    """
    cols = [treatment_col, outcome_col, running_col] + list(covariate_cols or [])
    clean = df[cols].copy()
    clean[outcome_col] = pd.to_numeric(clean[outcome_col], errors="coerce")
    clean[running_col] = pd.to_numeric(clean[running_col], errors="coerce")
    clean[treatment_col] = pd.to_numeric(clean[treatment_col], errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return {
            "method": "rdd",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "running_col": running_col,
            "threshold": threshold,
            "n_samples": 0,
            "local_effect": None,
            "note": "no complete cases",
        }

    x_running = clean[running_col].to_numpy(dtype=float)
    centered = x_running - float(threshold)
    if bandwidth is None:
        bandwidth = max(5.0, 0.5 * float(np.nanstd(centered)))
    keep = np.abs(centered) <= float(bandwidth)
    local = clean.loc[keep].copy()
    if len(local) < 30:
        return {
            "method": "rdd",
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "running_col": running_col,
            "threshold": threshold,
            "bandwidth": round(float(bandwidth), 4),
            "n_samples": int(len(local)),
            "local_effect": None,
            "note": "too few local samples",
        }

    centered_local = pd.to_numeric(local[running_col], errors="coerce").to_numpy(dtype=float) - float(threshold)
    if treated_when == "below":
        d = (centered_local <= 0).astype(float)
    elif treated_when == "above":
        d = (centered_local >= 0).astype(float)
    else:
        raise ValueError(f"Unsupported treated_when: {treated_when}")

    design_cols = [
        np.ones(len(local), dtype=float),
        d,
        centered_local,
        d * centered_local,
    ]
    if covariate_cols:
        cov = _encode_covariates(local[covariate_cols])
        design_cols.append(cov.to_numpy(dtype=float))
    x = np.column_stack(design_cols)
    y = pd.to_numeric(local[outcome_col], errors="coerce").to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    first_stage = float(np.mean(pd.to_numeric(local[treatment_col], errors="coerce").to_numpy(dtype=float)[d == 1])
                        - np.mean(pd.to_numeric(local[treatment_col], errors="coerce").to_numpy(dtype=float)[d == 0]))
    return {
        "method": "rdd",
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "running_col": running_col,
        "threshold": threshold,
        "treated_when": treated_when,
        "bandwidth": round(float(bandwidth), 4),
        "n_samples": int(len(local)),
        "threshold_indicator_effect": round(float(beta[1]), 4),
        "first_stage_jump": round(first_stage, 4),
        "local_effect": round(float(beta[1]), 4),
    }


def run_causal_suite(
    df: pd.DataFrame,
    *,
    treatment_cols: list[str],
    outcome_col: str,
    covariate_cols: list[str],
    effect_modifier_cols: list[str] | None = None,
    phenotype_col: str | None = "phenotype_dominant",
    rdd_specs: dict[str, dict] | None = None,
) -> dict:
    """Run the three causal estimators across multiple treatments."""
    results = {"outcome_col": outcome_col, "treatments": {}}
    for treatment_col in treatment_cols:
        logger.info("Running causal suite for %s", treatment_col)
        treatment_result = {
            "psm": estimate_propensity_score_matching(
                df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
            ),
            "causal_forest_dml": estimate_causal_forest_dml(
                df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariate_cols=covariate_cols,
                effect_modifier_cols=effect_modifier_cols,
                phenotype_col=phenotype_col,
            ),
        }
        if rdd_specs and treatment_col in rdd_specs:
            treatment_result["rdd"] = estimate_regression_discontinuity(
                df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                **rdd_specs[treatment_col],
            )
        results["treatments"][treatment_col] = treatment_result

    results["recommendations"] = generate_precision_treatment_recommendations(results)
    return results


def generate_precision_treatment_recommendations(results: dict) -> list[dict]:
    """
    Turn multi-method causal estimates into conservative treatment suggestions.

    Rules are intentionally strict:
      - require at least two methods to agree on direction
      - do not call something beneficial when all effects are small
    """
    recommendations = []
    for treatment_col, bundle in results.get("treatments", {}).items():
        scores = []
        sources = {}

        psm = bundle.get("psm", {})
        if psm.get("ate") is not None:
            scores.append(-float(psm["ate"]))
            sources["psm"] = float(psm["ate"])

        cf = bundle.get("causal_forest_dml", {})
        if cf.get("cate_mean") is not None:
            scores.append(-float(cf["cate_mean"]))
            sources["causal_forest_dml"] = float(cf["cate_mean"])

        rdd = bundle.get("rdd", {})
        if rdd.get("local_effect") is not None:
            scores.append(-float(rdd["local_effect"]))
            sources["rdd"] = float(rdd["local_effect"])

        if len(scores) < 2:
            direction = "uncertain"
        else:
            beneficial = sum(score > 0.01 for score in scores)
            harmful = sum(score < -0.01 for score in scores)
            if beneficial >= 2:
                direction = "candidate_beneficial"
            elif harmful >= 2:
                direction = "candidate_harmful"
            else:
                direction = "uncertain"

        recommendations.append(
            {
                "treatment_col": treatment_col,
                "direction": direction,
                "evidence": {k: round(v, 4) for k, v in sources.items()},
                "clinical_note": _recommendation_note(direction, treatment_col),
            }
        )
    return recommendations


def save_causal_results(results: dict, output_path: Path) -> None:
    """Persist a causal analysis bundle to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _append_low_dim_covariates(
    frame: pd.DataFrame,
    arr: np.ndarray | None,
    *,
    prefix: str,
    max_dims: int,
) -> None:
    if arr is None:
        return
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 3:
        values = values.mean(axis=1)
    if values.ndim != 2 or values.shape[0] != len(frame):
        return
    n_components = min(max_dims, values.shape[1], values.shape[0] - 1 if values.shape[0] > 1 else 1)
    if n_components <= 0:
        return
    if values.shape[1] > n_components:
        proj = PCA(n_components=n_components, random_state=42).fit_transform(values)
    else:
        proj = values[:, :n_components]
    for i in range(proj.shape[1]):
        frame[f"{prefix}_{i}"] = proj[:, i]


def _prepare_causal_inputs(
    *,
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    clean = df[[treatment_col, outcome_col] + covariate_cols].copy()
    clean[treatment_col] = pd.to_numeric(clean[treatment_col], errors="coerce")
    clean[outcome_col] = pd.to_numeric(clean[outcome_col], errors="coerce")
    clean = clean.dropna(subset=[treatment_col, outcome_col])
    x_df = _encode_covariates(clean[covariate_cols])
    y = clean[outcome_col].to_numpy(dtype=float)
    w = clean[treatment_col].astype(int).to_numpy(dtype=int)
    x = SimpleImputer(strategy="median").fit_transform(x_df)
    return clean.reset_index(drop=True), x, y, w


def _encode_covariates(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(df.copy(), dummy_na=True)
    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    return encoded


def _top_feature_importance(importances: np.ndarray, columns: list[str], top_k: int = 12) -> list[dict]:
    order = np.argsort(importances)[::-1][:top_k]
    return [
        {
            "feature": str(columns[idx]),
            "importance": round(float(importances[idx]), 4),
        }
        for idx in order
    ]


def _recommendation_note(direction: str, treatment_col: str) -> str:
    if direction == "candidate_beneficial":
        return f"{treatment_col} shows directionally consistent benefit and merits phenotype-stratified validation."
    if direction == "candidate_harmful":
        return f"{treatment_col} shows directionally consistent harm signals; confounding review is required before escalation."
    return f"{treatment_col} remains observationally ambiguous; use for hypothesis generation only."
