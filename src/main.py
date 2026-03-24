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

import duckdb
import numpy as np
import pandas as pd

# Ensure src directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    load_config,
    set_global_seed,
    setup_logger,
    resolve_path,
    ensure_dir,
    timer,
    sanitize_tag,
    build_output_name,
)
from data_loader import load_data, save_processed_data, get_feature_names
from preprocess import preprocess_pipeline, analyze_missing_pattern
from feature_engineering import extract_features
from representation_model import get_representation_model
from clustering import search_optimal_k, run_final_clustering, reduce_dimensions, compare_methods
from evaluation import evaluate_all
from build_analysis_table import build_analysis_tables
from eicu_loader import prepare_eicu_demo_artifacts
from import_to_duckdb import CREATE_SQL_PATH, create_tables, load_csv_files, validate_import
from run_concepts import run_all_concepts
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
  python main.py --source mimic --data-dir archive/mimic-iv-mock --overwrite-db --tag mimic_mock
  python main.py --source eicu --data-dir data/external/eicu_demo --tag eicu_demo
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
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override raw/demo data directory for the selected source")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Override prepared artifact directory for the selected source")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Override DuckDB path for MIMIC source")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional run tag used to isolate output artifacts and tagged cache files")
    parser.add_argument("--output-reports-dir", type=str, default=None,
                        help="Optional reports directory override")
    parser.add_argument("--output-figures-dir", type=str, default=None,
                        help="Optional figures directory override")
    parser.add_argument("--hours", type=int, default=None,
                        help="Override extracted hours / timesteps for MIMIC or eICU")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Optional eICU patient cap during preparation/loading")
    parser.add_argument("--overwrite-db", action="store_true",
                        help="Rebuild the MIMIC DuckDB database before preparing analysis tables")
    parser.add_argument("--skip-vis", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--compare-methods", action="store_true",
                        help="Run multi-method clustering comparison")
    return parser.parse_args()


def _apply_cli_overrides(config: dict, args) -> None:
    """Apply command-line overrides directly to the loaded config."""
    tag = sanitize_tag(args.tag)

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

    config.setdefault("runtime", {})["output_tag"] = tag

    if args.output_reports_dir:
        config["paths"]["output_reports"] = args.output_reports_dir
    elif tag:
        config["paths"]["output_reports"] = str(Path(config["paths"]["output_reports"]) / tag)

    if args.output_figures_dir:
        config["paths"]["output_figures"] = args.output_figures_dir
    elif tag:
        config["paths"]["output_figures"] = str(Path(config["paths"]["output_figures"]) / tag)

    source = config["data"]["source"]
    if source == "mimic":
        mimic_cfg = config["data"].setdefault("mimic", {})
        if args.data_dir:
            mimic_cfg["raw_data_dir"] = args.data_dir
        if args.processed_dir:
            mimic_cfg["processed_dir"] = args.processed_dir
        if args.db_path:
            mimic_cfg["db_path"] = args.db_path
        if args.hours is not None:
            mimic_cfg["hours"] = args.hours
    elif source == "eicu":
        eicu_cfg = config["data"].setdefault("eicu", {})
        if args.data_dir:
            eicu_cfg["data_dir"] = args.data_dir
        if args.processed_dir:
            eicu_cfg["processed_dir"] = args.processed_dir
        if tag:
            eicu_cfg["tag"] = tag
        if args.hours is not None:
            eicu_cfg["n_timesteps"] = args.hours
        if args.max_patients is not None:
            eicu_cfg["max_patients"] = args.max_patients


def _prepare_external_inputs(config: dict, args, logger) -> None:
    """Prepare external-source artifacts so `load_data` can use a single entry point."""
    source = config["data"]["source"]
    if source == "mimic":
        _prepare_mimic_source(config, args, logger)
    elif source == "eicu":
        _prepare_eicu_source(config, args, logger)


def _mimic_analysis_tables_exist(processed_dir: Path) -> bool:
    """Return whether MIMIC patient_static + patient_timeseries already exist."""
    for suffix in (".parquet", ".csv"):
        static_path = processed_dir / f"patient_static{suffix}"
        ts_path = processed_dir / f"patient_timeseries{suffix}"
        if static_path.exists() and ts_path.exists():
            return True
    return False


def _mimic_db_ready(db_path: Path) -> bool:
    """Check whether the DuckDB file already contains the derived MIMIC tables we need."""
    if not db_path.exists():
        return False

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        required_tables = [
            "mimiciv_hosp.patients",
            "mimiciv_icu.icustays",
            "mimiciv_derived.icustay_detail",
            "mimiciv_derived.sepsis3",
        ]
        for table in required_tables:
            conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _rebuild_mimic_database(raw_dir: Path, db_path: Path) -> None:
    """Import raw MIMIC CSVs into DuckDB and execute concept SQL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn, CREATE_SQL_PATH)
        load_csv_files(conn, raw_dir)
        import_ok = validate_import(conn)
    finally:
        conn.close()

    if not import_ok:
        raise RuntimeError(
            "MIMIC import finished with empty critical tables. "
            "Check that the raw directory contains the expected hosp/ and icu/ CSV files."
        )

    concept_results = run_all_concepts(db_path=db_path, dry_run=False, stop_on_error=False)
    if concept_results["failed"]:
        failed = ", ".join(concept_results["failed"][:5])
        raise RuntimeError(f"MIMIC concept generation failed: {failed}")


def _prepare_mimic_source(config: dict, args, logger) -> None:
    """Build MIMIC analysis tables on demand so main.py becomes the only user entry point."""
    mimic_cfg = config["data"].setdefault("mimic", {})
    raw_dir = resolve_path(mimic_cfg.get("raw_data_dir", "data/external/mimic_iv_demo"))
    processed_dir = resolve_path(mimic_cfg.get("processed_dir") or config["paths"]["processed_data"])
    db_path = resolve_path(mimic_cfg.get("db_path", "db/mimic4_demo.db"))
    n_hours = int(mimic_cfg.get("hours", 48))
    output_format = mimic_cfg.get("output_format", "parquet")

    analysis_ready = _mimic_analysis_tables_exist(processed_dir)
    rebuild_requested = bool(args.overwrite_db)

    if analysis_ready and not rebuild_requested:
        logger.info("Using existing MIMIC analysis tables: %s", processed_dir)
        return

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"MIMIC raw data directory not found: {raw_dir}\n"
            "Download/extract the CSV dump so `hosp/` and `icu/` exist, then re-run with --data-dir."
        )

    if rebuild_requested:
        logger.info("Rebuilding MIMIC DuckDB from raw data: %s", raw_dir)
        _rebuild_mimic_database(raw_dir, db_path)
    elif not _mimic_db_ready(db_path):
        logger.info("Preparing fresh MIMIC DuckDB from raw data: %s", raw_dir)
        _rebuild_mimic_database(raw_dir, db_path)
    else:
        logger.info("Reusing existing MIMIC DuckDB: %s", db_path)

    logger.info("Exporting MIMIC analysis tables to %s", processed_dir)
    build_analysis_tables(
        db_path=db_path,
        output_dir=processed_dir,
        n_hours=n_hours,
        output_format=output_format,
    )


def _eicu_cache_paths(processed_dir: Path, tag: str) -> tuple[Path, Path]:
    """Return the expected eICU cache file paths."""
    return processed_dir / f"time_series_{tag}.npy", processed_dir / f"patient_info_{tag}.csv"


def _prepare_eicu_source(config: dict, args, logger) -> None:
    """Build eICU cache artifacts on demand so main.py can be the single public entry point."""
    eicu_cfg = config["data"].setdefault("eicu", {})
    data_dir = resolve_path(eicu_cfg.get("data_dir") or eicu_cfg.get("demo_dir") or "data/external/eicu_demo")
    processed_dir = resolve_path(eicu_cfg.get("processed_dir", "data/processed_eicu_demo"))
    tag = eicu_cfg.get("tag", "eicu_demo")

    ts_path, info_path = _eicu_cache_paths(processed_dir, tag)
    cache_exists = ts_path.exists() and info_path.exists()
    refresh_requested = bool(args.hours is not None or args.max_patients is not None)

    if cache_exists and not refresh_requested:
        logger.info("Using existing eICU cached artifacts: %s", processed_dir)
        return

    if not data_dir.exists():
        raise FileNotFoundError(
            f"eICU raw data directory not found: {data_dir}\n"
            "Place the official CSV files there, or pass --data-dir to a local eICU extract."
        )

    logger.info("Preparing eICU cached artifacts from raw data: %s", data_dir)
    report = prepare_eicu_demo_artifacts(config, output_dir=processed_dir, tag=tag)
    logger.info(
        "eICU cache ready: patients=%s, report=%s",
        report["n_patients"],
        report["report_path"],
    )


# ============================================================
# Main Pipeline
# ============================================================

@timer
def main():
    args = parse_args()

    # --- 1. Load configuration ---
    config = load_config(args.config)

    _apply_cli_overrides(config, args)

    set_global_seed(config["runtime"]["random_seed"])
    logger = setup_logger("sepsis_subtype", level=config["runtime"]["log_level"])

    logger.info("=" * 60)
    logger.info("ICU Sepsis Dynamic Subtype Discovery System")
    logger.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if config["runtime"].get("output_tag"):
        logger.info("Run tag: %s", config["runtime"]["output_tag"])
    logger.info("=" * 60)

    _prepare_external_inputs(config, args, logger)

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

    save_processed_data(
        processed_3d,
        patient_info,
        config,
        tag=build_output_name("preprocessed", config["runtime"].get("output_tag")),
    )

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
    tag = config.get("runtime", {}).get("output_tag")

    serializable = _make_serializable(report)

    # JSON format
    json_path = out_dir / f"{build_output_name('evaluation_report', tag)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Text summary
    txt_path = out_dir / f"{build_output_name('evaluation_summary', tag)}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ICU Sepsis Dynamic Subtype Discovery - Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if tag:
            f.write(f"Run tag: {tag}\n")
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
