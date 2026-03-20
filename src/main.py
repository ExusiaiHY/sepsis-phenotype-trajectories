"""
main.py - Main entry point

Full pipeline orchestration:
  Data loading -> Missing analysis -> Preprocessing -> Feature extraction ->
  (Representation learning) -> Dimensionality reduction ->
  Clustering (optimal K search) -> Evaluation -> Visualization -> Report generation

Supports command-line argument overrides.
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Ensure src directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config, set_global_seed, setup_logger, resolve_path, ensure_dir, timer
from data_loader import load_data, save_processed_data, get_feature_names
from preprocess import preprocess_pipeline, analyze_missing_pattern
from feature_engineering import extract_features
from representation_model import get_representation_model
from clustering import search_optimal_k, run_final_clustering, reduce_dimensions, compare_methods
from evaluation import evaluate_all
from visualization import (
    plot_cluster_scatter,
    plot_k_selection,
    plot_subtype_heatmap,
    plot_survival_curves,
    plot_missing_pattern,
    plot_trajectory_comparison,
    plot_summary_dashboard,
)


# ============================================================
# Command-Line Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ICU Sepsis Dynamic Subtype Discovery System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use default config
  python main.py --n-patients 1000            # Generate 1000 simulated patients
  python main.py --method gmm --k 5          # Use GMM clustering, K=5
  python main.py --config path/to/config.yaml # Use custom config file
  python main.py --source mimic               # Use MIMIC-IV data source
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Configuration file path")
    parser.add_argument("--n-patients", type=int, default=None,
                        help="Number of simulated patients (overrides config)")
    parser.add_argument("--source", type=str, default=None,
                        choices=["simulated", "mimic", "eicu", "physionet2012", "sepsis2019"],
                        help="Data source (overrides config)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["kmeans", "gmm", "hierarchical", "spectral"],
                        help="Clustering method")
    parser.add_argument("--k", type=int, default=None,
                        help="Specify cluster count (skip auto-search)")
    parser.add_argument("--reduction", type=str, default=None,
                        choices=["umap", "tsne", "pca"],
                        help="Dimensionality reduction method")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--skip-vis", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--compare-methods", action="store_true",
                        help="Run multi-method clustering comparison")
    return parser.parse_args()


# ============================================================
# Main Pipeline
# ============================================================

@timer
def main():
    args = parse_args()

    # --- 1. Load configuration ---
    config = load_config(args.config)

    # Command-line overrides
    if args.n_patients:
        config["data"]["simulated"]["n_patients"] = args.n_patients
    if args.source:
        config["data"]["source"] = args.source
    if args.method:
        config["clustering"]["method"] = args.method
    if args.reduction:
        config["reduction"]["method"] = args.reduction
    if args.seed:
        config["runtime"]["random_seed"] = args.seed
        config["data"]["simulated"]["random_seed"] = args.seed
        config["clustering"]["random_seed"] = args.seed

    set_global_seed(config["runtime"]["random_seed"])
    logger = setup_logger("sepsis_subtype", level=config["runtime"]["log_level"])

    logger.info("=" * 60)
    logger.info("ICU Sepsis Dynamic Subtype Discovery System")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # --- 2. Data loading ---
    logger.info("[Step 1/7] Data loading")
    time_series_3d, patient_info = load_data(config)
    feature_names = get_feature_names(config)
    logger.info(f"  Data shape: {time_series_3d.shape}")
    logger.info(f"  Patients: {len(patient_info)}, Features: {len(feature_names)}")

    # --- 3. Missing pattern analysis ---
    logger.info("[Step 2/7] Missing pattern analysis")
    missing_stats = analyze_missing_pattern(time_series_3d, feature_names)
    logger.info(f"\n{missing_stats[['variable', 'missing_rate']].to_string(index=False)}")

    if not args.skip_vis:
        plot_missing_pattern(missing_stats, config)

    # --- 4. Preprocessing ---
    logger.info("[Step 3/7] Data preprocessing")
    processed_3d, scaler_params = preprocess_pipeline(time_series_3d, config, feature_names=feature_names)

    save_processed_data(processed_3d, patient_info, config, tag="preprocessed")

    # --- 5. Feature extraction ---
    logger.info("[Step 4/7] Feature extraction")
    feature_df = extract_features(processed_3d, config, feature_names)
    logger.info(f"  Feature matrix: {feature_df.shape}")

    # --- 6. Representation learning (MVP: PCA reduction) ---
    logger.info("[Step 5/7] Representation learning")
    rep_model = get_representation_model(config, mode="statistical")
    X = rep_model.fit_transform(feature_df.values)
    logger.info(f"  Representation dims: {X.shape}")

    # --- 7. Dimensionality reduction (for visualization) ---
    coords_2d = reduce_dimensions(X, config)

    # --- 8. Clustering ---
    logger.info("[Step 6/7] Clustering analysis")
    if args.k:
        labels = run_final_clustering(X, config, n_clusters=args.k)
        k_search_result = None
    else:
        k_search_result = search_optimal_k(X, config)
        labels = run_final_clustering(X, config, n_clusters=k_search_result["optimal_k"])

    if args.compare_methods:
        n_k = len(np.unique(labels))
        comparison = compare_methods(X, n_clusters=n_k)
        logger.info(f"Clustering method comparison:\n{comparison.to_string(index=False)}")

    # --- 9. Evaluation ---
    logger.info("[Step 7/7] Result evaluation")
    report = evaluate_all(X, labels, patient_info, config, feature_df)

    # --- 10. Visualization ---
    if not args.skip_vis:
        logger.info("Generating visualizations...")

        true_labels = patient_info["subtype_true"].values if "subtype_true" in patient_info.columns else None

        plot_cluster_scatter(coords_2d, labels, config, true_labels=true_labels)

        if k_search_result is not None:
            plot_k_selection(k_search_result["k_scores"], k_search_result["optimal_k"], config)

        plot_subtype_heatmap(feature_df, labels, config)
        plot_survival_curves(labels, patient_info, config)
        plot_trajectory_comparison(processed_3d, labels, feature_names, config)

        if k_search_result is not None:
            plot_summary_dashboard(
                coords_2d, labels,
                k_search_result["k_scores"],
                k_search_result["optimal_k"],
                patient_info, config,
            )

    # --- 11. Save report ---
    _save_report(report, config)

    logger.info("=" * 60)
    logger.info("All steps completed!")
    logger.info(f"Figures output: {resolve_path(config['paths']['output_figures'])}")
    logger.info(f"Reports output: {resolve_path(config['paths']['output_reports'])}")
    logger.info("=" * 60)


# ============================================================
# Report Saving
# ============================================================

def _save_report(report: dict, config: dict) -> None:
    """Save evaluation report as JSON and text formats."""
    out_dir = ensure_dir(resolve_path(config["paths"]["output_reports"]))

    serializable = _make_serializable(report)

    # JSON format
    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Text summary
    txt_path = out_dir / "evaluation_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ICU Sepsis Dynamic Subtype Discovery - Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if "cluster_metrics" in report:
            f.write("1. Clustering Quality Metrics\n")
            for k, v in report["cluster_metrics"].items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        if "external_metrics" in report:
            f.write("2. External Validation Metrics\n")
            ext = report["external_metrics"]
            f.write(f"  ARI: {ext.get('adjusted_rand_index', 'N/A')}\n")
            f.write(f"  NMI: {ext.get('normalized_mutual_info', 'N/A')}\n")
            f.write("\n")

        if "survival" in report:
            f.write("3. Survival Stratification Analysis\n")
            for k, info in report["survival"].get("per_cluster", {}).items():
                f.write(f"  Subtype {k}: mortality={info.get('mortality_rate')}, "
                        f"ICU LOS mean={info.get('mean_icu_los')}h\n")
            f.write("\n")

        if "subtype_profiles" in report:
            f.write("4. Subtype Clinical Profiles\n")
            profiles = report["subtype_profiles"]
            if isinstance(profiles, pd.DataFrame):
                f.write(profiles.to_string(index=False))
            f.write("\n")

    import logging
    logging.getLogger("sepsis_subtype").info(f"Report saved: {json_path}, {txt_path}")


def _make_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
