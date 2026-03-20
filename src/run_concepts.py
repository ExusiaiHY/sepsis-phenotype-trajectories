"""
run_concepts.py - Execute mimic-code concepts SQL on DuckDB

Parses concepts_duckdb/duckdb.sql for execution order,
runs each SQL file sequentially, and reports success/failure.

Usage:
  python run_concepts.py                              # Default db/mimic4.db
  python run_concepts.py --db-path ../db/mimic4.db    # Specify database
  python run_concepts.py --dry-run                    # Parse only, don't execute
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "mimic4.db"
CONCEPTS_DIR = PROJECT_ROOT / "mimic-code-main" / "mimic-iv" / "concepts_duckdb"
ENTRY_SQL = CONCEPTS_DIR / "duckdb.sql"


def parse_execution_order(entry_sql: Path) -> list[Path]:
    """Parse duckdb.sql entry file to extract dependency-ordered SQL file list."""
    sql_files = []
    with open(entry_sql, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(".read "):
                rel_path = line[6:].strip()
                full_path = entry_sql.parent / rel_path
                if full_path.exists():
                    sql_files.append(full_path)
                else:
                    print(f"  Warning: file not found {rel_path}")
    return sql_files


def execute_concept_sql(conn: duckdb.DuckDBPyConnection, sql_path: Path) -> tuple[bool, str, str]:
    """Execute a single concept SQL file. Returns (success, table_name, info)."""
    sql_text = sql_path.read_text(encoding="utf-8")

    match = re.search(r'CREATE TABLE (mimiciv_derived\.\w+)', sql_text)
    table_name = match.group(1) if match else sql_path.stem

    try:
        conn.execute(sql_text)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        return True, table_name, f"{row_count} rows"
    except Exception as e:
        error_msg = str(e).split("\n")[0][:150]
        return False, table_name, error_msg


def run_all_concepts(
    db_path: Path,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> dict:
    """Execute all concept SQL files and return results summary."""
    print("=" * 65)
    print("Executing MIMIC-IV Concepts (DuckDB)")
    print(f"Database: {db_path}")
    print(f"Concepts directory: {CONCEPTS_DIR}")
    print("=" * 65)

    sql_files = parse_execution_order(ENTRY_SQL)
    print(f"\n{len(sql_files)} concept SQL files to execute\n")

    if dry_run:
        print("[DRY RUN] Listing execution order only:")
        for i, f in enumerate(sql_files, 1):
            print(f"  {i:2d}. {f.relative_to(CONCEPTS_DIR)}")
        return {"success": [], "failed": [], "skipped": sql_files}

    conn = duckdb.connect(str(db_path))
    results = {"success": [], "failed": [], "details": {}}

    conn.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived;")

    start_total = time.time()
    for i, sql_path in enumerate(sql_files, 1):
        rel_name = str(sql_path.relative_to(CONCEPTS_DIR))
        start = time.time()

        success, table_name, info = execute_concept_sql(conn, sql_path)
        elapsed = time.time() - start

        status = "OK" if success else "FAIL"
        print(f"  [{i:2d}/{len(sql_files)}] {rel_name:50s} {status:4s}  {info:30s}  ({elapsed:.2f}s)")

        results["details"][table_name] = {
            "success": success,
            "info": info,
            "elapsed": round(elapsed, 3),
            "sql_file": rel_name,
        }

        if success:
            results["success"].append(table_name)
        else:
            results["failed"].append(table_name)
            if stop_on_error:
                print(f"\nStopping (--stop-on-error)")
                break

    total_time = time.time() - start_total
    conn.close()

    print("\n" + "=" * 65)
    print(f"Execution complete! Elapsed: {total_time:.1f}s")
    print(f"  Succeeded: {len(results['success'])} / {len(sql_files)}")
    print(f"  Failed: {len(results['failed'])} / {len(sql_files)}")

    if results["failed"]:
        print(f"\nFailed concepts:")
        for t in results["failed"]:
            d = results["details"][t]
            print(f"  {t}: {d['info']}")

    if results["success"]:
        print(f"\nKey derived table status:")
        key_tables = [
            "mimiciv_derived.vitalsign",
            "mimiciv_derived.bg",
            "mimiciv_derived.chemistry",
            "mimiciv_derived.complete_blood_count",
            "mimiciv_derived.sofa",
            "mimiciv_derived.sirs",
            "mimiciv_derived.suspicion_of_infection",
            "mimiciv_derived.sepsis3",
            "mimiciv_derived.ventilation",
            "mimiciv_derived.norepinephrine",
            "mimiciv_derived.icustay_detail",
            "mimiciv_derived.antibiotic",
        ]
        for t in key_tables:
            if t in results["details"]:
                d = results["details"][t]
                mark = "+" if d["success"] else "x"
                print(f"  [{mark}] {t}: {d['info']}")
            else:
                print(f"  [?] {t}: not executed")

    print("=" * 65)
    return results


def main():
    parser = argparse.ArgumentParser(description="Execute MIMIC-IV concepts SQL")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't execute")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    results = run_all_concepts(
        db_path=Path(args.db_path),
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )

    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
