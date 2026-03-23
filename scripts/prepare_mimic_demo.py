#!/usr/bin/env python3
"""
prepare_mimic_demo.py - Build MIMIC demo/raw analysis tables for the legacy V1 pipeline.

How to run:
  cd project
  ./.venv/bin/python scripts/prepare_mimic_demo.py
  ./.venv/bin/python scripts/prepare_mimic_demo.py --data-dir archive/mimic-iv-mock --overwrite-db
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_analysis_table import build_analysis_tables
from import_to_duckdb import CREATE_SQL_PATH, create_tables, load_csv_files, validate_import
from run_concepts import run_all_concepts


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MIMIC demo/raw analysis tables")
    parser.add_argument("--data-dir", default="data/external/mimic_iv_demo")
    parser.add_argument("--output-dir", default="data/processed_mimic_demo")
    parser.add_argument("--db-path", default="db/mimic4_demo.db")
    parser.add_argument("--hours", type=int, default=48)
    parser.add_argument("--format", default="parquet", choices=["csv", "parquet"])
    parser.add_argument("--overwrite-db", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    db_path = PROJECT_ROOT / args.db_path

    if not data_dir.exists():
        raise FileNotFoundError(
            f"MIMIC raw/demo directory not found: {data_dir}\n"
            "Place the demo/full files under hosp/ and icu/ before running this script."
        )
    if db_path.exists() and not args.overwrite_db:
        raise FileExistsError(
            f"Database already exists: {db_path}\n"
            "Re-run with --overwrite-db if you want this script to rebuild it."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn, CREATE_SQL_PATH)
        load_csv_files(conn, data_dir)
        import_ok = validate_import(conn)
    finally:
        conn.close()

    concept_results = run_all_concepts(db_path=db_path, dry_run=False, stop_on_error=False)
    patient_static, patient_ts = build_analysis_tables(
        db_path=db_path,
        output_dir=output_dir,
        n_hours=args.hours,
        output_format=args.format,
    )

    report = {
        "source": "mimic",
        "raw_data_dir": str(data_dir),
        "db_path": str(db_path),
        "output_dir": str(output_dir),
        "import_ok": bool(import_ok),
        "concept_success_count": len(concept_results["success"]),
        "concept_failure_count": len(concept_results["failed"]),
        "n_patients": int(patient_static["stay_id"].nunique()),
        "n_timeseries_rows": int(len(patient_ts)),
        "n_features": int(len([c for c in patient_ts.columns if c not in ("stay_id", "subject_id", "hr", "grid_time")])),
        "sepsis_rate": round(float(patient_static["is_sepsis3"].mean()), 4),
        "mortality_rate": round(float(patient_static["mortality_28d"].mean()), 4),
        "artifacts": {
            "patient_static": str(output_dir / f"patient_static.{args.format}"),
            "patient_timeseries": str(output_dir / f"patient_timeseries.{args.format}"),
        },
    }
    report_path = output_dir / "mimic_demo_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
