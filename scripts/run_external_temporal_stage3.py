#!/usr/bin/env python3
"""
run_external_temporal_stage3.py - Unified external S1.5 + Stage 3 runner.

How to run:
  cd project
  ./.venv/bin/python scripts/run_external_temporal_stage3.py --source all
  ./.venv/bin/python scripts/run_external_temporal_stage3.py --source mimic --max-patients 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from s0.external_temporal_builder import prepare_external_temporal_s0


def parse_args():
    parser = argparse.ArgumentParser(description="Run external temporal transfer workflow")
    parser.add_argument("--source", choices=["mimic", "eicu", "all"], default="all")
    parser.add_argument("--output-root", default="data/external_temporal")
    parser.add_argument("--reference-stats", default="data/s0/processed/preprocess_stats.json")
    parser.add_argument("--s15-checkpoint", default="data/s15/checkpoints/pretrain_best.pt")
    parser.add_argument("--n-hours", type=int, default=48)
    parser.add_argument("--split-method", choices=["random", "cross_center"], default="random")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--skip-source-prep", action="store_true")
    parser.add_argument("--skip-s0", action="store_true")
    parser.add_argument("--skip-s15", action="store_true")
    parser.add_argument("--skip-s2", action="store_true")

    parser.add_argument("--mimic-raw-dir", default="mimic-iv-3.1")
    parser.add_argument("--mimic-processed-dir", default="data/processed_mimic_real")
    parser.add_argument("--mimic-db-path", default="db/mimic4_external_temporal.db")
    parser.add_argument("--eicu-raw-dir", default="EICU 2.0数据")
    parser.add_argument("--eicu-processed-dir", default="data/processed_eicu_real")
    parser.add_argument("--eicu-tag", default="eicu_demo")

    parser.add_argument("--fit-sample-size", type=int, default=400000)
    parser.add_argument("--silhouette-sample-size", type=int, default=20000)
    parser.add_argument("--overall-silhouette-sample-size", type=int, default=50000)
    parser.add_argument("--predict-batch-size", type=int, default=200000)
    return parser.parse_args()


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def to_project_string(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("external_temporal")

    sources = ["mimic", "eicu"] if args.source == "all" else [args.source]
    output_root = resolve_project_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary_path = output_root / "external_temporal_runs.json"
    if all_summary_path.exists():
        summaries = json.loads(all_summary_path.read_text(encoding="utf-8"))
    else:
        summaries = {}
    for source in sources:
        logger.info("=" * 72)
        logger.info("External temporal workflow: %s", source)
        logger.info("=" * 72)

        prepared_dir = _ensure_source_artifacts(source, args)
        source_root = output_root / source
        source_root.mkdir(parents=True, exist_ok=True)
        s0_dir = source_root / "s0"
        s15_dir = source_root / "s15"
        s2_dir = source_root / "s2"
        config_dir = source_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        s0_summary = None
        if not args.skip_s0:
            s0_summary = prepare_external_temporal_s0(
                source=source,
                output_dir=s0_dir,
                processed_dir=prepared_dir,
                reference_stats_path=resolve_project_path(args.reference_stats),
                n_hours=args.n_hours,
                split_method=args.split_method,
                max_patients=args.max_patients,
            )
            logger.info("S0 external bundle ready: %s", s0_dir)

        s15_config_path = _write_s15_config(
            config_dir=config_dir,
            s0_dir=s0_dir,
            s15_dir=s15_dir,
            checkpoint_path=resolve_project_path(args.s15_checkpoint),
            device=args.device,
            batch_size=args.batch_size,
        )
        s2_config_path = _write_s2_config(
            config_dir=config_dir,
            s0_dir=s0_dir,
            s2_dir=s2_dir,
            checkpoint_path=resolve_project_path(args.s15_checkpoint),
            device=args.device,
            batch_size=args.batch_size,
            fit_sample_size=args.fit_sample_size,
            silhouette_sample_size=args.silhouette_sample_size,
            overall_silhouette_sample_size=args.overall_silhouette_sample_size,
            predict_batch_size=args.predict_batch_size,
        )

        if not args.skip_s15:
            _run_python(
                [
                    "scripts/s15_extract.py",
                    "--config",
                    str(s15_config_path.relative_to(PROJECT_ROOT)),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ]
            )

        if not args.skip_s2:
            _run_python(
                [
                    "scripts/s2_extract_rolling.py",
                    "--config",
                    str(s2_config_path.relative_to(PROJECT_ROOT)),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
            _run_python(
                [
                    "scripts/s2_cluster_and_analyze.py",
                    "--config",
                    str(s2_config_path.relative_to(PROJECT_ROOT)),
                ]
            )

        summary = {
            "source": source,
            "prepared_source_dir": str(prepared_dir),
            "source_root": str(source_root),
            "s0_dir": str(s0_dir),
            "s15_dir": str(s15_dir),
            "s2_dir": str(s2_dir),
            "s15_config": str(s15_config_path),
            "s2_config": str(s2_config_path),
            "s0_summary": s0_summary,
            "device": args.device,
            "batch_size": args.batch_size,
            "fit_sample_size": args.fit_sample_size,
            "silhouette_sample_size": args.silhouette_sample_size,
            "overall_silhouette_sample_size": args.overall_silhouette_sample_size,
            "predict_batch_size": args.predict_batch_size,
        }
        summary_path = source_root / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries[source] = summary
        logger.info("Run summary saved: %s", summary_path)

    all_summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Combined summary saved: %s", all_summary_path)


def _ensure_source_artifacts(source: str, args) -> Path:
    logger = logging.getLogger("external_temporal.prep")
    if source == "mimic":
        prepared_dir = resolve_project_path(args.mimic_processed_dir)
        static_candidates = [prepared_dir / "patient_static.parquet", prepared_dir / "patient_static.csv"]
        ts_candidates = [prepared_dir / "patient_timeseries.parquet", prepared_dir / "patient_timeseries.csv"]
        if any(path.exists() for path in static_candidates) and any(path.exists() for path in ts_candidates):
            return prepared_dir
        if args.skip_source_prep:
            raise FileNotFoundError(
                f"MIMIC prepared artifacts not found under {prepared_dir}. "
                "Either generate them first or drop --skip-source-prep."
            )
        logger.info("Prepared MIMIC artifacts missing. Running prepare_mimic_demo.py ...")
        _run_python(
            [
                "scripts/prepare_mimic_demo.py",
                "--data-dir",
                args.mimic_raw_dir,
                "--output-dir",
                args.mimic_processed_dir,
                "--db-path",
                args.mimic_db_path,
                "--hours",
                str(args.n_hours),
                "--format",
                "parquet",
                "--overwrite-db",
            ]
        )
        return prepared_dir

    if source == "eicu":
        prepared_dir = resolve_project_path(args.eicu_processed_dir)
        tensor_path = prepared_dir / f"time_series_{args.eicu_tag}.npy"
        info_path = prepared_dir / f"patient_info_{args.eicu_tag}.csv"
        feat_path = prepared_dir / f"feature_names_{args.eicu_tag}.json"
        if tensor_path.exists() and info_path.exists() and feat_path.exists():
            return prepared_dir
        if args.skip_source_prep:
            raise FileNotFoundError(
                f"eICU prepared artifacts not found under {prepared_dir}. "
                "Either generate them first or drop --skip-source-prep."
            )
        logger.info("Prepared eICU artifacts missing. Running prepare_eicu_demo.py ...")
        _run_python(
            [
                "scripts/prepare_eicu_demo.py",
                "--data-dir",
                args.eicu_raw_dir,
                "--output-dir",
                args.eicu_processed_dir,
                "--hours",
                str(args.n_hours),
                "--tag",
                args.eicu_tag,
            ]
        )
        return prepared_dir

    raise ValueError(f"Unsupported source: {source}")


def _write_s15_config(
    config_dir: Path,
    s0_dir: Path,
    s15_dir: Path,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
) -> Path:
    with open(PROJECT_ROOT / "config" / "s15_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["s0_dir"] = to_project_string(s0_dir)
    cfg["paths"]["s15_dir"] = to_project_string(s15_dir)
    cfg["paths"]["s15_checkpoint"] = to_project_string(checkpoint_path)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = device
    cfg["runtime"]["batch_size"] = batch_size
    out_path = config_dir / "s15_external.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _write_s2_config(
    config_dir: Path,
    s0_dir: Path,
    s2_dir: Path,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    fit_sample_size: int | None,
    silhouette_sample_size: int | None,
    overall_silhouette_sample_size: int | None,
    predict_batch_size: int | None,
) -> Path:
    with open(PROJECT_ROOT / "config" / "s2_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["s0_dir"] = to_project_string(s0_dir)
    cfg["paths"]["s2_dir"] = to_project_string(s2_dir)
    cfg["paths"]["s15_encoder"] = to_project_string(checkpoint_path)
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = device
    cfg["runtime"]["batch_size"] = batch_size
    cfg.setdefault("clustering", {})
    cfg["clustering"]["fit_sample_size"] = fit_sample_size
    cfg["clustering"]["silhouette_sample_size"] = silhouette_sample_size
    cfg["clustering"]["overall_silhouette_sample_size"] = overall_silhouette_sample_size
    cfg["clustering"]["predict_batch_size"] = predict_batch_size
    out_path = config_dir / "s2_external.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def _run_python(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    logging.getLogger("external_temporal.exec").info("Running: %s", " ".join(str(part) for part in cmd))
    cache_root = PROJECT_ROOT / ".cache"
    mpl_cache = cache_root / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
