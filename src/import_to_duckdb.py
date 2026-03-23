"""
import_to_duckdb.py - Import MIMIC-IV CSV data into DuckDB

Cross-platform Python equivalent of import_duckdb.sh:
1. Read create.sql to build table schema
2. Scan data directory for CSV.gz files
3. COPY into DuckDB tables
4. Validate critical tables

Usage:
  python import_to_duckdb.py                              # Use mock data
  python import_to_duckdb.py --data-dir ../mimic-iv-data  # Use real data
  python import_to_duckdb.py --db-path ../db/mimic4.db    # Specify output path
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "db" / "mimic4.db"


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MOCK_DIR = _first_existing(
    [
        PROJECT_ROOT / "mimic-iv-mock",
        PROJECT_ROOT / "archive" / "mimic-iv-mock",
    ]
)
CREATE_SQL_PATH = _first_existing(
    [
        PROJECT_ROOT / "mimic-code-main" / "mimic-iv" / "buildmimic" / "postgres" / "create.sql",
        PROJECT_ROOT / "archive" / "mimic-code-main" / "mimic-iv" / "buildmimic" / "postgres" / "create.sql",
    ]
)


def adapt_create_sql(sql_text: str) -> str:
    """
    Adapt PostgreSQL create.sql to DuckDB-compatible syntax.
    Equivalent to the sed transformations in import_duckdb.sh.
    """
    import re
    sql_text = re.sub(r'TIMESTAMP\(\d+\)', 'TIMESTAMP', sql_text)
    sql_text = re.sub(r'(spec_type_desc.+?)NOT NULL', r'\1', sql_text)
    sql_text = re.sub(r'(drug\s+VARCHAR.+?)NOT NULL', r'\1', sql_text)
    return sql_text


def create_tables(conn: duckdb.DuckDBPyConnection, create_sql_path: Path) -> None:
    """Read create.sql and build all tables."""
    print(f"[1/3] Reading table DDL: {create_sql_path}")
    sql_text = create_sql_path.read_text(encoding="utf-8")
    sql_text = adapt_create_sql(sql_text)

    statements = [s.strip() for s in sql_text.split(";") if s.strip()]
    created = 0
    for stmt in statements:
        if stmt.upper().startswith(("DROP", "CREATE")):
            try:
                conn.execute(stmt)
                if stmt.upper().startswith("CREATE TABLE"):
                    created += 1
            except Exception as e:
                print(f"  Warning: {str(e)[:100]}")

    print(f"  Tables created: {created}")


def load_csv_files(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Scan data directory and import CSV files into corresponding tables."""
    print(f"[2/3] Importing data: {data_dir}")

    schema_map = {"hosp": "mimiciv_hosp", "icu": "mimiciv_icu"}
    total_loaded = 0
    total_rows = 0

    for subdir_name, schema_name in schema_map.items():
        subdir = data_dir / subdir_name
        if not subdir.exists():
            print(f"  Skipping: {subdir} does not exist")
            continue

        csv_files = sorted(list(subdir.glob("*.csv.gz")) + list(subdir.glob("*.csv")))
        for csv_path in csv_files:
            table_name = csv_path.name.split(".")[0]
            full_table = f"{schema_name}.{table_name}"

            try:
                result = conn.execute(
                    f"COPY {full_table} FROM '{csv_path}' (HEADER, DELIM ',', QUOTE '\"', ESCAPE '\"');"
                )
                row_count = conn.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]
                total_loaded += 1
                total_rows += row_count
                print(f"  {full_table}: {row_count} rows")
            except duckdb.CatalogException:
                print(f"  Skipped: table {full_table} not found in schema")
            except Exception as e:
                print(f"  Error: {full_table} - {str(e)[:120]}")

    print(f"  Import complete: {total_loaded} tables, {total_rows:,} rows")


def validate_import(conn: duckdb.DuckDBPyConnection) -> bool:
    """Validate that critical tables have been imported with data."""
    print("[3/3] Validating import")
    critical_tables = [
        "mimiciv_hosp.patients",
        "mimiciv_hosp.admissions",
        "mimiciv_hosp.labevents",
        "mimiciv_icu.icustays",
        "mimiciv_icu.chartevents",
        "mimiciv_icu.d_items",
    ]
    all_ok = True
    for table in critical_tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            status = "OK" if count > 0 else "EMPTY!"
            if count == 0:
                all_ok = False
            print(f"  {table}: {count} rows [{status}]")
        except Exception as e:
            print(f"  {table}: Error - {e}")
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Import MIMIC-IV CSV into DuckDB")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_MOCK_DIR),
                        help="MIMIC-IV data directory (with hosp/ and icu/ subdirs)")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH),
                        help="Output DuckDB file path")
    parser.add_argument("--create-sql", type=str, default=str(CREATE_SQL_PATH),
                        help="Path to create.sql")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    db_path = Path(args.db_path)
    create_sql = Path(args.create_sql)

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Removing existing database: {db_path}")
        db_path.unlink()

    print("=" * 60)
    print("MIMIC-IV -> DuckDB Import")
    print(f"Data directory: {data_dir}")
    print(f"Database file: {db_path}")
    print("=" * 60)

    start = time.time()

    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn, create_sql)
        load_csv_files(conn, data_dir)
        ok = validate_import(conn)

        elapsed = time.time() - start
        print()
        if ok:
            print(f"Import successful! Elapsed: {elapsed:.1f}s")
            print(f"Database file: {db_path} ({db_path.stat().st_size / 1024:.1f} KB)")
        else:
            print("Import completed with some empty tables, please check data directory")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
