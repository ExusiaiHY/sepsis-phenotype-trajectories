"""
s6_run_external_generalization.py - Run S6 optimization on external temporal bundles.

This makes cross-domain temporal generalization a first-class runnable stage by
reusing prepared external temporal bundles (for example MIMIC and eICU) and
executing the same S6 optimization pipeline on each source.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s0.schema import CONTINUOUS_NAMES
from s6_optimization.baseline_comparison import generate_baseline_comparison_report
from s6_optimization.causal_phenotyping import run_causal_phenotyping_pipeline
from s6_optimization.missingness_encoder import run_missingness_stage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("s6.external_generalization")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S6 optimization on external temporal bundles.")
    parser.add_argument(
        "--config",
        default="config/s6_config.yaml",
        help="Config path relative to project root",
    )
    parser.add_argument(
        "--external-root",
        default="data/external_temporal",
        help="External temporal bundle root relative to project root",
    )
    parser.add_argument(
        "--output-root",
        default="data/s6_external_generalization",
        help="Output root relative to project root",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["mimic", "eicu"],
        help="External sources to run",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum phenotype size for baseline comparison on external sources",
    )
    parser.add_argument(
        "--domain-alpha",
        type=float,
        default=None,
        help="Optional override for domain adaptation alpha blending strength",
    )
    parser.add_argument(
        "--missingness-selected-features",
        nargs="+",
        default=None,
        help="Optional override for patient-level missingness feature subset",
    )
    parser.add_argument(
        "--disable-missingness-patient-features",
        action="store_true",
        help="Disable appending patient-level missingness covariates into Stage 2",
    )
    return parser.parse_args()

def run_external_generalization() -> dict:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if args.domain_alpha is not None:
        cfg.setdefault("domain_adaptation", {})
        cfg["domain_adaptation"]["alpha"] = float(args.domain_alpha)
    if args.missingness_selected_features is not None:
        cfg.setdefault("missingness", {})
        cfg["missingness"]["selected_features"] = list(args.missingness_selected_features)
    if args.disable_missingness_patient_features:
        cfg.setdefault("missingness", {})
        cfg["missingness"]["append_patient_features"] = False

    external_root = PROJECT_ROOT / args.external_root
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    source_reports = {}

    for source in args.sources:
        source_root = external_root / source
        s0_dir = source_root / "s0"
        s2_dir = source_root / "s2"
        if not s0_dir.exists() or not s2_dir.exists():
            raise FileNotFoundError(f"Missing external bundle for {source}: {source_root}")

        output_dir = output_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("External S6 run: source=%s", source)
        logger.info("=" * 60)

        t0 = time.time()
        masks = np.load(s0_dir / "processed" / "masks_continuous.npy", mmap_mode="r")
        missingness_stage = run_missingness_stage(
            masks=np.array(masks),
            output_dir=output_dir,
            config=cfg.get("missingness"),
            feature_names=CONTINUOUS_NAMES,
        )
        causal_report = run_causal_phenotyping_pipeline(
            s0_dir=s0_dir,
            s2_dir=s2_dir,
            output_dir=output_dir,
            splits_path=s0_dir / "splits.json",
            missingness_features=missingness_stage["features_df"],
            missingness_feature_summary=missingness_stage["feature_summary"],
            causal_method=cfg["causal"]["method"],
            causal_config=cfg.get("causal"),
            treatment_horizon=cfg["causal"]["treatment_horizon"],
            organ_horizon=cfg["organ_scoring"]["horizon"],
            phenotype_config=cfg["phenotype_naming"],
            imputation_config=cfg.get("imputation"),
            dowhy_config=cfg.get("dowhy"),
            timesfm_config=cfg.get("timesfm"),
            domain_adaptation_config=cfg.get("domain_adaptation"),
        )
        comparison_report = generate_baseline_comparison_report(
            static_path=s0_dir / "static.csv",
            window_labels_path=s2_dir / "window_labels.npy",
            phenotype_assignments_path=output_dir / "phenotype_assignments.csv",
            output_path=output_dir / "baseline_comparison.json",
            min_group_size=args.min_group_size,
        )
        source_reports[source] = {
            "source_root": str(source_root),
            "output_dir": str(output_dir),
            "duration_sec": round(time.time() - t0, 1),
            "missingness": {
                "summary": missingness_stage["summary"],
                "artifacts": missingness_stage["artifacts"],
                "covariate_summary": missingness_stage["feature_summary"],
            },
            "causal_report": {
                "report_path": causal_report["report_path"],
                "n_patients": causal_report["n_patients"],
                "treatment_exposure_rate": causal_report["treatment_exposure_rate"],
                "outcome_rate": causal_report["outcome_rate"],
                "cate_summary": causal_report["cate_summary"],
                "cate_estimator_selected": causal_report.get("cate_estimator_selected"),
                "timesfm_summary": causal_report.get("timesfm_summary", {}),
                "domain_adaptation_summary": causal_report.get("domain_adaptation_summary", {}),
                "imputation_summary": causal_report.get("imputation_summary", {}),
                "dowhy_validation": causal_report.get("dowhy_validation", {}),
            },
            "baseline_comparison": {
                "supported_group_count": comparison_report["optimized"]["supported_group_count"],
                "supported_mortality_range": comparison_report["optimized"]["supported_mortality_range"],
                "weighted_mortality_std": comparison_report["optimized"]["weighted_mortality_std"],
                "center_distribution_l1": comparison_report["optimized"]["center_distribution_l1"],
                "center_mortality_deviation": comparison_report["optimized"]["center_mortality_deviation"],
                "report_path": str(output_dir / "baseline_comparison.json"),
            },
        }

    report = {
        "pipeline": "s6_external_generalization",
        "config": cfg,
        "external_root": str(external_root),
        "output_root": str(output_root),
        "sources": source_reports,
        "total_duration_sec": round(time.time() - started_at, 1),
    }
    report_path = output_root / "external_generalization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("External generalization report: %s", report_path)
    return report


if __name__ == "__main__":
    run_external_generalization()
