"""
mimic_db_profile.py - Structured profiling for the local DuckDB MIMIC database.

Purpose:
  Provide a reproducible schema and cohort summary for the local MIMIC-IV
  DuckDB database used by the legacy extraction pipeline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_CANDIDATES = [
    PROJECT_ROOT / "archive" / "db" / "mimic4.db",
    PROJECT_ROOT / "db" / "mimic4.db",
]

KEY_TABLES = {
    "mimiciv_hosp": ["patients", "admissions"],
    "mimiciv_icu": ["icustays"],
    "mimiciv_derived": [
        "icustay_detail",
        "sepsis3",
        "sirs",
        "sofa",
        "vitalsign",
        "chemistry",
        "complete_blood_count",
        "first_day_vitalsign",
        "first_day_lab",
    ],
}


def resolve_db_path(db_path: Path | None = None) -> Path:
    if db_path is not None:
        resolved = Path(db_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Database not found: {resolved}")
        return resolved

    for candidate in DEFAULT_DB_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find mimic4.db in archive/db/ or db/")


def build_mimic_profile(db_path: Path | None = None) -> dict:
    resolved = resolve_db_path(db_path)
    conn = duckdb.connect(str(resolved))
    try:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": str(resolved),
            "schemas": _fetch_list(
                conn,
                "SELECT DISTINCT schema_name FROM information_schema.schemata ORDER BY schema_name",
            ),
            "table_inventory": _table_inventory(conn),
            "key_table_counts": _key_table_counts(conn),
            "analysis_table_readiness": _analysis_table_readiness(conn),
            "cohort_summary": _cohort_summary(conn),
            "first_day_missingness": _first_day_missingness(conn),
        }
    finally:
        conn.close()
    return report


def save_profile(report: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mimic_duckdb_profile.json"
    md_path = output_dir / "mimic_duckdb_profile.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(render_profile_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_profile_markdown(report: dict) -> str:
    lines = [
        "# MIMIC DuckDB Profile",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Database: `{report['db_path']}`",
        f"- Schemas: `{', '.join(report['schemas'])}`",
        "",
        "## Key Table Counts",
        "",
        "| Table | Rows |",
        "|------|-----:|",
    ]
    for table_name, count in report["key_table_counts"].items():
        lines.append(f"| `{table_name}` | {count} |")

    cohort = report["cohort_summary"]
    lines.extend(
        [
            "",
            "## Cohort Summary",
            "",
            f"- ICU stays in `mimiciv_derived.icustay_detail`: `{cohort['icu_stays']}`",
            f"- Distinct patients: `{cohort['patients']}`",
            f"- Mean age: `{cohort['mean_age']}`",
            f"- Male rate: `{cohort['male_rate']}`",
            f"- Hospital mortality: `{cohort['hospital_expire_rate']}`",
            f"- Sepsis-3 prevalence: `{cohort['sepsis3_prevalence']}`",
            "",
            "## First-Day Missingness",
            "",
            "| Variable | Missing Rate |",
            "|----------|-------------:|",
        ]
    )

    for name, value in report["first_day_missingness"].items():
        lines.append(f"| `{name}` | {value} |")

    readiness = report["analysis_table_readiness"]
    lines.extend(
        [
            "",
            "## Analysis Table Readiness",
            "",
            f"- Required tables present: `{readiness['all_required_present']}`",
            f"- Present tables: `{readiness['present_count']}` / `{readiness['required_count']}`",
        ]
    )

    return "\n".join(lines) + "\n"


def _fetch_list(conn: duckdb.DuckDBPyConnection, query: str) -> list[str]:
    return [str(row[0]) for row in conn.execute(query).fetchall()]


def _table_inventory(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema LIKE 'mimiciv_%'
        ORDER BY table_schema, table_name
        """
    ).fetchall()

    inventory: dict[str, list[str]] = {}
    for schema_name, table_name in rows:
        inventory.setdefault(str(schema_name), []).append(str(table_name))
    return inventory


def _key_table_counts(conn: duckdb.DuckDBPyConnection) -> dict[str, int | None]:
    counts = {}
    for schema_name, table_names in KEY_TABLES.items():
        for table_name in table_names:
            full_name = f"{schema_name}.{table_name}"
            try:
                counts[full_name] = int(conn.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0])
            except Exception:
                counts[full_name] = None
    return counts


def _analysis_table_readiness(conn: duckdb.DuckDBPyConnection) -> dict:
    required = [
        "mimiciv_derived.icustay_detail",
        "mimiciv_derived.sepsis3",
        "mimiciv_derived.first_day_sofa",
        "mimiciv_derived.first_day_vitalsign",
        "mimiciv_derived.first_day_lab",
        "mimiciv_derived.charlson",
        "mimiciv_derived.norepinephrine",
        "mimiciv_derived.sirs",
        "mimiciv_hosp.patients",
    ]

    present = {}
    for full_name in required:
        schema_name, table_name = full_name.split(".", 1)
        exists = conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = ? AND table_name = ?
            """,
            [schema_name, table_name],
        ).fetchone()[0]
        present[full_name] = bool(exists)

    present_count = sum(present.values())
    return {
        "required_count": len(required),
        "present_count": int(present_count),
        "all_required_present": present_count == len(required),
        "required_tables": present,
    }


def _cohort_summary(conn: duckdb.DuckDBPyConnection) -> dict[str, float | int | None]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS icu_stays,
            COUNT(DISTINCT subject_id) AS patients,
            AVG(admission_age) AS mean_age,
            AVG(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) AS male_rate,
            AVG(CASE WHEN hospital_expire_flag = 1 THEN 1 ELSE 0 END) AS hospital_expire_rate
        FROM mimiciv_derived.icustay_detail
        """
    ).fetchone()

    sepsis_prevalence = conn.execute(
        """
        SELECT AVG(CASE WHEN s.stay_id IS NOT NULL THEN 1 ELSE 0 END)
        FROM mimiciv_derived.icustay_detail d
        LEFT JOIN mimiciv_derived.sepsis3 s USING (stay_id)
        """
    ).fetchone()[0]

    return {
        "icu_stays": int(row[0]),
        "patients": int(row[1]),
        "mean_age": round(float(row[2]), 2) if row[2] is not None else None,
        "male_rate": round(float(row[3]), 4) if row[3] is not None else None,
        "hospital_expire_rate": round(float(row[4]), 4) if row[4] is not None else None,
        "sepsis3_prevalence": round(float(sepsis_prevalence), 4) if sepsis_prevalence is not None else None,
    }


def _first_day_missingness(conn: duckdb.DuckDBPyConnection) -> dict[str, float | None]:
    row = conn.execute(
        """
        SELECT
            AVG(CASE WHEN heart_rate_min IS NULL THEN 1 ELSE 0 END) AS fd_heart_rate_missing,
            AVG(CASE WHEN sbp_min IS NULL THEN 1 ELSE 0 END) AS fd_sbp_missing,
            AVG(CASE WHEN resp_rate_min IS NULL THEN 1 ELSE 0 END) AS fd_resp_rate_missing,
            AVG(CASE WHEN spo2_min IS NULL THEN 1 ELSE 0 END) AS fd_spo2_missing
        FROM mimiciv_derived.first_day_vitalsign
        """
    ).fetchone()

    lab_row = conn.execute(
        """
        SELECT
            AVG(CASE WHEN creatinine_max IS NULL THEN 1 ELSE 0 END) AS fd_creatinine_missing,
            AVG(CASE WHEN bilirubin_total_max IS NULL THEN 1 ELSE 0 END) AS fd_bilirubin_missing,
            AVG(CASE WHEN platelets_min IS NULL THEN 1 ELSE 0 END) AS fd_platelets_missing,
            AVG(CASE WHEN wbc_max IS NULL THEN 1 ELSE 0 END) AS fd_wbc_missing
        FROM mimiciv_derived.first_day_lab
        """
    ).fetchone()

    names = [
        "fd_heart_rate_missing",
        "fd_sbp_missing",
        "fd_resp_rate_missing",
        "fd_spo2_missing",
        "fd_creatinine_missing",
        "fd_bilirubin_missing",
        "fd_platelets_missing",
        "fd_wbc_missing",
    ]
    values = list(row) + list(lab_row)
    return {
        name: round(float(value), 4) if value is not None else None
        for name, value in zip(names, values)
    }
