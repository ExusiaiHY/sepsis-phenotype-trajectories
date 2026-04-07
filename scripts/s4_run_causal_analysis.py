#!/usr/bin/env python3
"""s4_run_causal_analysis.py - Run Stage 4 causal treatment-effect analysis."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s4.causal_analysis import build_causal_frame, run_causal_suite, save_causal_results
from s4.treatment_features import load_treatment_bundle


def _resolve(path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 4 causal analysis")
    parser.add_argument("--config", default="config/s4_config.yaml")
    parser.add_argument("--s0-dir", default=None)
    parser.add_argument("--treatment-dir", default=None)
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--phenotype-labels", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    with open(_resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    s0_dir = _resolve(args.s0_dir or cfg["paths"]["s0_dir"])
    treatment_dir = _resolve(args.treatment_dir or cfg["paths"]["treatment_dir"])
    output_dir = _resolve(args.output_dir or cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_treatment_bundle(treatment_dir)
    continuous = np.load(s0_dir / "processed" / "continuous.npy", mmap_mode="r")
    masks_cont = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
    embeddings = np.load(_resolve(args.embeddings), mmap_mode="r") if args.embeddings else None
    phenotype_labels = np.load(_resolve(args.phenotype_labels), mmap_mode="r") if args.phenotype_labels else None

    s0_static = pd.read_csv(s0_dir / "static.csv")
    s0_static["patient_id"] = s0_static["patient_id"].astype(str)
    cohort_static = bundle["cohort_static"].copy()
    cohort_static["patient_id"] = cohort_static["patient_id"].astype(str)
    s0_index = {pid: idx for idx, pid in enumerate(s0_static["patient_id"].tolist())}
    keep_pairs = [(i, s0_index[pid]) for i, pid in enumerate(cohort_static["patient_id"].tolist()) if pid in s0_index]
    if not keep_pairs:
        raise ValueError("No overlapping patient_id values between treatment cohort and S0 cohort")
    treat_keep = np.asarray([p[0] for p in keep_pairs], dtype=int)
    s0_keep = np.asarray([p[1] for p in keep_pairs], dtype=int)
    cohort_static = cohort_static.iloc[treat_keep].reset_index(drop=True)
    treatments = np.asarray(bundle["treatments"][treat_keep])
    if embeddings is not None:
        embeddings = np.asarray(embeddings[s0_keep])
    if phenotype_labels is not None:
        phenotype_labels = np.asarray(phenotype_labels[s0_keep])

    horizons = tuple(int(x) for x in cfg.get("causal", {}).get("horizons", [6, 24]))
    outcome_col = cfg.get("causal", {}).get("outcome_col", "mortality_inhospital")
    causal_frame = build_causal_frame(
        cohort_static=cohort_static,
        treatments=treatments,
        treatment_names=bundle["feature_names"],
        continuous=np.asarray(continuous[s0_keep]),
        masks_continuous=np.asarray(masks_cont[s0_keep]),
        embeddings=embeddings,
        phenotype_labels=phenotype_labels,
        horizons=horizons,
        outcome_col=outcome_col,
        max_embedding_dims=int(cfg.get("causal", {}).get("max_embedding_dims", 16)),
    )
    outcome_col = _resolve_outcome_col(causal_frame, outcome_col)
    causal_frame_path = output_dir / "causal_frame.csv"
    causal_frame.to_csv(causal_frame_path, index=False)

    treatment_cols = _auto_treatment_cols(causal_frame)
    covariate_cols = _auto_covariates(causal_frame, treatment_cols, outcome_col)
    rdd_specs = _default_rdd_specs(causal_frame)

    results = run_causal_suite(
        causal_frame,
        treatment_cols=treatment_cols,
        outcome_col=outcome_col,
        covariate_cols=covariate_cols,
        effect_modifier_cols=_auto_effect_modifiers(causal_frame, covariate_cols),
        phenotype_col="phenotype_dominant" if "phenotype_dominant" in causal_frame.columns else None,
        rdd_specs=rdd_specs,
    )
    results["causal_frame_path"] = str(causal_frame_path)
    results["treatment_cols"] = treatment_cols
    results["covariate_cols"] = covariate_cols
    save_causal_results(results, output_dir / "causal_analysis_report.json")
    logging.getLogger("s4.causal").info("Saved causal analysis to %s", output_dir / "causal_analysis_report.json")


def _resolve_outcome_col(frame: pd.DataFrame, preferred: str | None) -> str:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["mortality_inhospital", "mortality_28d", "hospital_expire_flag"])
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        "No supported outcome column found in causal frame. "
        f"Tried {candidates}, available columns include: {frame.columns.tolist()}"
    )


def _auto_treatment_cols(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "vasopressor_on_any_6h",
        "antibiotic_on_any_6h",
        "mechanical_vent_on_any_6h",
        "fluid_bolus_ml_any_6h",
        "crystalloid_fluid_ml_any_6h",
        "rrt_on_any_24h",
    ]
    return [col for col in candidates if col in frame.columns]


def _auto_covariates(frame: pd.DataFrame, treatment_cols: list[str], outcome_col: str) -> list[str]:
    covariates = []
    whitelist_prefixes = (
        "age",
        "gender",
        "sex",
        "race",
        "charlson",
        "sirs",
        "first_day_sofa",
        "obs_density_",
        "map_",
        "heart_rate_",
        "resp_rate_",
        "spo2_",
        "creatinine_",
        "lactate_",
        "bilirubin_",
        "wbc_",
        "phenotype_",
        "emb_",
    )
    blocked = set(treatment_cols + [outcome_col])
    for col in frame.columns:
        if col in blocked:
            continue
        if col.startswith(whitelist_prefixes):
            covariates.append(col)
    return covariates


def _auto_effect_modifiers(frame: pd.DataFrame, covariates: list[str]) -> list[str]:
    preferred = [col for col in covariates if col.startswith(("phenotype_", "emb_", "lactate_", "map_", "creatinine_"))]
    return preferred or covariates


def _default_rdd_specs(frame: pd.DataFrame) -> dict[str, dict]:
    specs = {}
    if "vasopressor_on_any_6h" in frame.columns and "map_mean_6h" in frame.columns:
        specs["vasopressor_on_any_6h"] = {
            "running_col": "map_mean_6h",
            "threshold": 65.0,
            "treated_when": "below",
            "covariate_cols": [col for col in ("age", "charlson_comorbidity_index", "phenotype_dominant") if col in frame.columns],
        }
    if "rrt_on_any_24h" in frame.columns and "creatinine_mean_6h" in frame.columns:
        specs["rrt_on_any_24h"] = {
            "running_col": "creatinine_mean_6h",
            "threshold": 2.0,
            "treated_when": "above",
            "covariate_cols": [col for col in ("age", "phenotype_dominant") if col in frame.columns],
        }
    if "mechanical_vent_on_any_6h" in frame.columns and "spo2_min_6h" in frame.columns:
        specs["mechanical_vent_on_any_6h"] = {
            "running_col": "spo2_min_6h",
            "threshold": 92.0,
            "treated_when": "below",
            "covariate_cols": [col for col in ("age", "phenotype_dominant") if col in frame.columns],
        }
    return specs


if __name__ == "__main__":
    main()
