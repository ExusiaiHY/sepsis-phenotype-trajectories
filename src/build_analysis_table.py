"""
build_analysis_table.py - Build patient-level sepsis analysis tables

Extracts and merges data from DuckDB mimiciv_derived.* tables to produce:

1. patient_static: Patient-level static info (one row per patient)
   - Demographics, ICU stay info, Sepsis-3 diagnosis, outcomes

2. patient_timeseries: Patient-level time-series (N_HOURS rows per patient)
   - Vital signs, labs, SOFA scores, urine output, vasopressors

Time window strategy:
  - Sepsis-3 patients: anchor at sofa_time, -24h to +24h (48h total)
  - Non-Sepsis-3 ICU patients: anchor at ICU admission, first 48h
  - Time resolution: 1 hour

Usage:
  python build_analysis_table.py
  python build_analysis_table.py --db-path ../db/mimic4.db --hours 48 --format parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_DB_PATH = _first_existing(
    [
        PROJECT_ROOT / "db" / "mimic4.db",
        PROJECT_ROOT / "archive" / "db" / "mimic4.db",
    ]
)


# ============================================================
# 1. Patient Static Info SQL
# ============================================================

STATIC_SQL = """
WITH sepsis_flag AS (
    SELECT
        stay_id,
        1 AS is_sepsis3,
        sofa_time AS sepsis_onset_time,
        sofa_score AS sepsis_sofa_score,
        respiration, coagulation, liver,
        cardiovascular, cns, renal
    FROM mimiciv_derived.sepsis3
),
mortality AS (
    SELECT
        subject_id,
        CASE WHEN dod IS NOT NULL THEN 1 ELSE 0 END AS mortality_28d,
        dod
    FROM mimiciv_hosp.patients
)
SELECT
    d.subject_id,
    d.hadm_id,
    d.stay_id,
    d.gender,
    d.admission_age AS age,
    d.race,
    d.icu_intime,
    d.icu_outtime,
    CAST(d.los_icu AS DOUBLE) AS los_icu_days,
    d.los_hospital AS los_hospital_days,
    d.hospital_expire_flag,
    COALESCE(m.mortality_28d, 0) AS mortality_28d,
    m.dod,
    COALESCE(sf.is_sepsis3, 0) AS is_sepsis3,
    sf.sepsis_onset_time,
    sf.sepsis_sofa_score,
    COALESCE(sf.sepsis_onset_time, d.icu_intime) AS anchor_time,
    fds.sofa AS first_day_sofa,
    fdv.heart_rate_min AS fd_hr_min,
    fdv.heart_rate_max AS fd_hr_max,
    fdv.sbp_min AS fd_sbp_min,
    fdv.sbp_max AS fd_sbp_max,
    fdv.resp_rate_min AS fd_rr_min,
    fdv.temperature_min AS fd_temp_min,
    fdv.temperature_max AS fd_temp_max,
    fdv.spo2_min AS fd_spo2_min,
    fdl.creatinine_max AS fd_creatinine_max,
    fdl.bilirubin_total_max AS fd_bilirubin_max,
    fdl.platelets_min AS fd_platelet_min,
    fdl.wbc_max AS fd_wbc_max,
    ch.charlson_comorbidity_index,
    CASE WHEN ne.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor_use,
    sirs.sirs
FROM mimiciv_derived.icustay_detail d
LEFT JOIN mortality m ON d.subject_id = m.subject_id
LEFT JOIN sepsis_flag sf ON d.stay_id = sf.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa fds ON d.stay_id = fds.stay_id
LEFT JOIN mimiciv_derived.first_day_vitalsign fdv ON d.stay_id = fdv.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fdl ON d.stay_id = fdl.stay_id
LEFT JOIN mimiciv_derived.charlson ch ON d.hadm_id = ch.hadm_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.norepinephrine) ne ON d.stay_id = ne.stay_id
LEFT JOIN mimiciv_derived.sirs sirs ON d.stay_id = sirs.stay_id
ORDER BY d.stay_id
"""


# ============================================================
# 2. Time-Series Extraction SQL
# ============================================================

TIMESERIES_SQL = """
WITH anchor AS (
    SELECT
        d.stay_id,
        d.subject_id,
        d.hadm_id,
        COALESCE(s3.sofa_time, d.icu_intime) AS anchor_time,
        d.icu_intime,
        d.icu_outtime
    FROM mimiciv_derived.icustay_detail d
    LEFT JOIN mimiciv_derived.sepsis3 s3 ON d.stay_id = s3.stay_id
),
hours AS (
    SELECT UNNEST(GENERATE_SERIES(0, {n_hours} - 1)) AS hr
),
time_grid AS (
    SELECT
        a.stay_id,
        a.subject_id,
        a.hadm_id,
        h.hr,
        a.anchor_time + INTERVAL (h.hr) HOUR AS grid_time,
        a.anchor_time
    FROM anchor a
    CROSS JOIN hours h
),
vs_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        AVG(vs.heart_rate) AS heart_rate,
        AVG(vs.sbp) AS sbp,
        AVG(vs.dbp) AS dbp,
        AVG(vs.mbp) AS map,
        AVG(vs.resp_rate) AS resp_rate,
        AVG(vs.spo2) AS spo2,
        AVG(CAST(vs.temperature AS DOUBLE)) AS temperature
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.vitalsign vs
        ON tg.stay_id = vs.stay_id
        AND vs.charttime >= tg.grid_time
        AND vs.charttime < tg.grid_time + INTERVAL '1' HOUR
    GROUP BY tg.stay_id, tg.hr
),
sofa_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        sf.sofa_24hours AS sofa_total,
        sf.respiration_24hours AS sofa_resp,
        sf.coagulation_24hours AS sofa_coag,
        sf.liver_24hours AS sofa_liver,
        sf.cardiovascular_24hours AS sofa_cardio,
        sf.cns_24hours AS sofa_cns,
        sf.renal_24hours AS sofa_renal,
        sf.meanbp_min,
        sf.uo_24hr AS urine_output_24hr,
        sf.bilirubin_max,
        sf.creatinine_max,
        sf.platelet_min,
        sf.rate_norepinephrine AS norepi_rate
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.sofa sf
        ON tg.stay_id = sf.stay_id
        AND sf.starttime >= tg.grid_time
        AND sf.starttime < tg.grid_time + INTERVAL '1' HOUR
),
gcs_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(g.gcs ORDER BY g.charttime DESC) AS gcs,
        FIRST(g.gcs_motor ORDER BY g.charttime DESC) AS gcs_motor,
        FIRST(g.gcs_verbal ORDER BY g.charttime DESC) AS gcs_verbal,
        FIRST(g.gcs_eyes ORDER BY g.charttime DESC) AS gcs_eyes
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.gcs g
        ON tg.stay_id = g.stay_id
        AND g.charttime >= tg.grid_time - INTERVAL '4' HOUR
        AND g.charttime <= tg.grid_time + INTERVAL '1' HOUR
    GROUP BY tg.stay_id, tg.hr
),
labs_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(ch.creatinine ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS creatinine,
        FIRST(ch.bun ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS bun,
        FIRST(ch.sodium ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS sodium,
        FIRST(ch.potassium ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS potassium,
        FIRST(ch.bicarbonate ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS bicarbonate,
        FIRST(ch.glucose ORDER BY ABS(EPOCH(ch.charttime - tg.grid_time))) AS glucose
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.chemistry ch
        ON tg.hadm_id = ch.hadm_id
        AND ch.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND ch.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
cbc_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(cb.wbc ORDER BY ABS(EPOCH(cb.charttime - tg.grid_time))) AS wbc,
        FIRST(cb.platelet ORDER BY ABS(EPOCH(cb.charttime - tg.grid_time))) AS platelet,
        FIRST(cb.hemoglobin ORDER BY ABS(EPOCH(cb.charttime - tg.grid_time))) AS hemoglobin
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.complete_blood_count cb
        ON tg.hadm_id = cb.hadm_id
        AND cb.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND cb.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
coag_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(co.inr ORDER BY ABS(EPOCH(co.charttime - tg.grid_time))) AS inr
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.coagulation co
        ON tg.hadm_id = co.hadm_id
        AND co.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND co.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
)
SELECT
    tg.stay_id,
    tg.subject_id,
    tg.hr,
    tg.grid_time,
    vs.heart_rate,
    vs.sbp,
    vs.dbp,
    vs.map,
    vs.resp_rate,
    vs.spo2,
    vs.temperature,
    gc.gcs,
    gc.gcs_motor,
    gc.gcs_verbal,
    gc.gcs_eyes,
    lb.creatinine,
    lb.bun,
    lb.sodium,
    lb.potassium,
    lb.bicarbonate,
    lb.glucose,
    cbc.wbc,
    cbc.platelet,
    cbc.hemoglobin,
    cg.inr,
    sf.sofa_total,
    sf.sofa_resp,
    sf.sofa_coag,
    sf.sofa_liver,
    sf.sofa_cardio,
    sf.sofa_cns,
    sf.sofa_renal,
    sf.norepi_rate,
    sf.urine_output_24hr,
    sf.meanbp_min
FROM time_grid tg
LEFT JOIN vs_hourly vs ON tg.stay_id = vs.stay_id AND tg.hr = vs.hr
LEFT JOIN sofa_hourly sf ON tg.stay_id = sf.stay_id AND tg.hr = sf.hr
LEFT JOIN gcs_hourly gc ON tg.stay_id = gc.stay_id AND tg.hr = gc.hr
LEFT JOIN labs_hourly lb ON tg.stay_id = lb.stay_id AND tg.hr = lb.hr
LEFT JOIN cbc_hourly cbc ON tg.stay_id = cbc.stay_id AND tg.hr = cbc.hr
LEFT JOIN coag_hourly cg ON tg.stay_id = cg.stay_id AND tg.hr = cg.hr
ORDER BY tg.stay_id, tg.hr
"""


# ============================================================
# Main Pipeline
# ============================================================

def build_analysis_tables(
    db_path: Path,
    output_dir: Path,
    n_hours: int = 48,
    output_format: str = "csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and export patient-level analysis tables."""
    print("=" * 65)
    print("Building Patient-Level Sepsis Analysis Tables")
    print(f"Database: {db_path}")
    print(f"Time window: {n_hours} hours")
    print(f"Output format: {output_format}")
    print("=" * 65)

    conn = duckdb.connect(str(db_path), read_only=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # --- 1. Static table ---
    print("\n[1/4] Building static info table...")
    patient_static = conn.execute(STATIC_SQL).fetchdf()
    print(f"  Shape: {patient_static.shape}")
    print(f"  Sepsis-3 patients: {patient_static['is_sepsis3'].sum()} / {len(patient_static)}")
    print(f"  28-day mortality: {patient_static['mortality_28d'].mean():.1%}")

    # --- 2. Time-series table ---
    print("\n[2/4] Building time-series table...")
    ts_sql = TIMESERIES_SQL.replace("{n_hours}", str(n_hours))
    patient_ts = conn.execute(ts_sql).fetchdf()
    n_patients = patient_ts["stay_id"].nunique()
    n_features = len([c for c in patient_ts.columns if c not in ("stay_id", "subject_id", "hr", "grid_time")])
    print(f"  Shape: {patient_ts.shape}")
    print(f"  Patients: {n_patients}, Timesteps: {n_hours}, Features: {n_features}")

    # Missing rate analysis
    print("\n[3/4] Missing rate analysis:")
    feature_cols = [c for c in patient_ts.columns if c not in ("stay_id", "subject_id", "hr", "grid_time")]
    for col in feature_cols:
        miss = patient_ts[col].isna().mean()
        bar = "#" * int(miss * 30) + "." * (30 - int(miss * 30))
        print(f"  {col:22s} {miss:5.1%} |{bar}|")

    # --- 3. Export ---
    print(f"\n[4/4] Exporting data...")
    if output_format == "parquet":
        static_path = output_dir / "patient_static.parquet"
        ts_path = output_dir / "patient_timeseries.parquet"
        patient_static.to_parquet(static_path, index=False)
        patient_ts.to_parquet(ts_path, index=False)
    else:
        static_path = output_dir / "patient_static.csv"
        ts_path = output_dir / "patient_timeseries.csv"
        patient_static.to_csv(static_path, index=False)
        patient_ts.to_csv(ts_path, index=False)

    conn.close()

    elapsed = time.time() - start
    print(f"  Static table: {static_path} ({static_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Time-series table: {ts_path} ({ts_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nDone! Elapsed: {elapsed:.1f}s")

    print("\n" + "=" * 65)
    print("Integration with main.py pipeline:")
    print(f"  Time-series table can be reshaped to 3D tensor:")
    print(f"    shape = ({n_patients}, {n_hours}, {n_features})")
    print(f"  Static table corresponds to patient_info DataFrame")
    print(f"  Set data.source: 'mimic' in config.yaml")
    print(f"  Then run: python main.py --source mimic")
    print("=" * 65)

    return patient_static, patient_ts


def main():
    parser = argparse.ArgumentParser(description="Build sepsis analysis tables")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hours", type=int, default=48, help="Time window length (hours)")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"])
    args = parser.parse_args()

    build_analysis_tables(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
        n_hours=args.hours,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
