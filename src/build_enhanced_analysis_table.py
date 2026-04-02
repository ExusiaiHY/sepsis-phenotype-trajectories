"""
build_enhanced_analysis_table.py - Build ENHANCED patient-level sepsis analysis tables

Extends build_analysis_table.py with additional data dimensions required for
subtype diagnosis and treatment recommendation:
  1. blood_differential (lymphocytes, monocytes, etc.)
  2. inflammation (CRP)
  3. enzyme (ALT, AST, total bilirubin)
  4. enhanced labs from labevents (ferritin, D-dimer, fibrinogen)
  5. microbiology events (blood/resp/urine culture positivity)
  6. ventilation status (mechanical ventilation)

Outputs:
  - patient_static_enhanced.parquet / .csv
  - patient_timeseries_enhanced.parquet / .csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

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
        PROJECT_ROOT / "db" / "mimic4_real.db",
        PROJECT_ROOT / "archive" / "db" / "mimic4.db",
    ]
)

# ============================================================
# 1. Enhanced Patient Static Info SQL
# ============================================================

ENHANCED_STATIC_SQL = """
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
),
-- Microbiology summaries
micro_summary AS (
    SELECT
        m.hadm_id,
        MAX(CASE WHEN lower(m.spec_type_desc) LIKE '%blood%' AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS blood_culture_positive,
        MAX(CASE WHEN (lower(m.spec_type_desc) LIKE '%sputum%' OR lower(m.spec_type_desc) LIKE '%resp%'
                       OR lower(m.spec_type_desc) LIKE '%broncho%') AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS resp_culture_positive,
        MAX(CASE WHEN lower(m.spec_type_desc) LIKE '%urine%' AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS urine_culture_positive,
        COUNT(DISTINCT CASE WHEN m.org_name IS NOT NULL THEN m.micro_specimen_id END) AS n_positive_cultures,
        MAX(CASE WHEN m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS any_culture_positive
    FROM mimiciv_hosp.microbiologyevents m
    GROUP BY m.hadm_id
),
-- Ventilation in first 24h
vent_first24 AS (
    SELECT
        d.stay_id,
        MAX(CASE WHEN lower(v.ventilation_status) IN ('invasive', 'mechanical', 'tracheostomy') THEN 1 ELSE 0 END) AS mech_vent_first24h
    FROM mimiciv_derived.icustay_detail d
    LEFT JOIN mimiciv_derived.ventilation v
        ON d.stay_id = v.stay_id
        AND v.starttime <= d.icu_intime + INTERVAL '24' HOUR
    GROUP BY d.stay_id
),
-- Enhanced first-day labs
fd_bd AS (
    SELECT
        d.stay_id,
        MIN(b.lymphocytes_abs) AS fd_lymphocytes_min,
        MIN(b.lymphocytes) AS fd_lymphocytes_pct_min
    FROM mimiciv_derived.icustay_detail d
    LEFT JOIN mimiciv_derived.blood_differential b
        ON d.hadm_id = b.hadm_id
        AND b.charttime >= d.icu_intime
        AND b.charttime < d.icu_intime + INTERVAL '24' HOUR
    GROUP BY d.stay_id
),
fd_infl AS (
    SELECT
        d.stay_id,
        MAX(i.crp) AS fd_crp_max
    FROM mimiciv_derived.icustay_detail d
    LEFT JOIN mimiciv_derived.inflammation i
        ON d.hadm_id = i.hadm_id
        AND i.charttime >= d.icu_intime
        AND i.charttime < d.icu_intime + INTERVAL '24' HOUR
    GROUP BY d.stay_id
),
fd_enz AS (
    SELECT
        d.stay_id,
        MAX(e.alt) AS fd_alt_max,
        MAX(e.ast) AS fd_ast_max,
        MAX(e.bilirubin_total) AS fd_bilirubin_max
    FROM mimiciv_derived.icustay_detail d
    LEFT JOIN mimiciv_derived.enzyme e
        ON d.hadm_id = e.hadm_id
        AND e.charttime >= d.icu_intime
        AND e.charttime < d.icu_intime + INTERVAL '24' HOUR
    GROUP BY d.stay_id
),
-- Ferritin and D-dimer from raw labevents (itemids: Ferritin=50924, D-Dimer=50915,51196,52551)
fd_bg AS (
    SELECT
        stay_id,
        MAX(lactate_max) AS fd_lactate_max
    FROM mimiciv_derived.first_day_bg
    GROUP BY stay_id
),
fd_enhanced_labs AS (
    SELECT
        l.hadm_id,
        MAX(CASE WHEN l.itemid = 50924 THEN CAST(l.valuenum AS DOUBLE) END) AS fd_ferritin_max,
        MAX(CASE WHEN l.itemid IN (50915, 51196, 52551) THEN CAST(l.valuenum AS DOUBLE) END) AS fd_ddimer_max,
        MAX(CASE WHEN l.itemid IN (51214, 51623, 52116, 52117) THEN CAST(l.valuenum AS DOUBLE) END) AS fd_fibrinogen_max
    FROM mimiciv_hosp.labevents l
    WHERE l.itemid IN (50924, 50915, 51196, 52551, 51214, 51623, 52116, 52117)
      AND l.valuenum IS NOT NULL
    GROUP BY l.hadm_id
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
    fds.respiration AS sofa_resp,
    fds.coagulation AS sofa_coag,
    fds.liver AS sofa_liver,
    fds.cardiovascular AS sofa_cardio,
    fds.cns AS sofa_cns,
    fds.renal AS sofa_renal,
    fdv.heart_rate_min AS fd_hr_min,
    fdv.heart_rate_max AS fd_hr_max,
    fdv.sbp_min AS fd_sbp_min,
    fdv.sbp_max AS fd_sbp_max,
    fdv.mbp_min AS fd_mbp_min,
    fdv.resp_rate_min AS fd_rr_min,
    fdv.temperature_min AS fd_temp_min,
    fdv.temperature_max AS fd_temp_max,
    fdv.spo2_min AS fd_spo2_min,
    fdl.creatinine_max AS fd_creatinine_max,
    fdl.bilirubin_total_max AS fd_bilirubin_max,
    fdl.platelets_min AS fd_platelet_min,
    fdl.wbc_max AS fd_wbc_max,
    fdbg.fd_lactate_max,
    ch.charlson_comorbidity_index,
    CASE WHEN ne.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor_use,
    sirs.sirs,
    -- enhanced microbiology
    COALESCE(ms.blood_culture_positive, 0) AS blood_culture_positive,
    COALESCE(ms.resp_culture_positive, 0) AS resp_culture_positive,
    COALESCE(ms.urine_culture_positive, 0) AS urine_culture_positive,
    COALESCE(ms.n_positive_cultures, 0) AS n_positive_cultures,
    COALESCE(ms.any_culture_positive, 0) AS any_culture_positive,
    -- enhanced ventilation
    COALESCE(v24.mech_vent_first24h, 0) AS mech_vent_first24h,
    -- enhanced labs
    fdbd.fd_lymphocytes_min,
    fdbd.fd_lymphocytes_pct_min,
    fdin.fd_crp_max,
    fde.fd_alt_max,
    fde.fd_ast_max,
    fel.fd_ferritin_max,
    fel.fd_ddimer_max,
    fel.fd_fibrinogen_max
FROM mimiciv_derived.icustay_detail d
LEFT JOIN mortality m ON d.subject_id = m.subject_id
LEFT JOIN sepsis_flag sf ON d.stay_id = sf.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa fds ON d.stay_id = fds.stay_id
LEFT JOIN mimiciv_derived.first_day_vitalsign fdv ON d.stay_id = fdv.stay_id
LEFT JOIN mimiciv_derived.first_day_lab fdl ON d.stay_id = fdl.stay_id
LEFT JOIN mimiciv_derived.charlson ch ON d.hadm_id = ch.hadm_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.norepinephrine) ne ON d.stay_id = ne.stay_id
LEFT JOIN mimiciv_derived.sirs sirs ON d.stay_id = sirs.stay_id
LEFT JOIN micro_summary ms ON d.hadm_id = ms.hadm_id
LEFT JOIN vent_first24 v24 ON d.stay_id = v24.stay_id
LEFT JOIN fd_bd fdbd ON d.stay_id = fdbd.stay_id
LEFT JOIN fd_infl fdin ON d.stay_id = fdin.stay_id
LEFT JOIN fd_enz fde ON d.stay_id = fde.stay_id
LEFT JOIN fd_bg fdbg ON d.stay_id = fdbg.stay_id
LEFT JOIN fd_enhanced_labs fel ON d.hadm_id = fel.hadm_id
ORDER BY d.stay_id
"""


# ============================================================
# 2. Enhanced Time-Series Extraction SQL
# ============================================================

ENHANCED_TIMESERIES_SQL = """
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
),
-- NEW: blood differential hourly
bd_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(b.lymphocytes_abs ORDER BY ABS(EPOCH(b.charttime - tg.grid_time))) AS lymphocytes_abs,
        FIRST(b.lymphocytes ORDER BY ABS(EPOCH(b.charttime - tg.grid_time))) AS lymphocytes_pct,
        FIRST(b.monocytes_abs ORDER BY ABS(EPOCH(b.charttime - tg.grid_time))) AS monocytes_abs,
        FIRST(b.neutrophils_abs ORDER BY ABS(EPOCH(b.charttime - tg.grid_time))) AS neutrophils_abs
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.blood_differential b
        ON tg.hadm_id = b.hadm_id
        AND b.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND b.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
-- NEW: inflammation hourly
infl_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(i.crp ORDER BY ABS(EPOCH(i.charttime - tg.grid_time))) AS crp
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.inflammation i
        ON tg.hadm_id = i.hadm_id
        AND i.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND i.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
-- NEW: enzyme hourly
enz_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        FIRST(e.alt ORDER BY ABS(EPOCH(e.charttime - tg.grid_time))) AS alt,
        FIRST(e.ast ORDER BY ABS(EPOCH(e.charttime - tg.grid_time))) AS ast,
        FIRST(e.bilirubin_total ORDER BY ABS(EPOCH(e.charttime - tg.grid_time))) AS bilirubin_total
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.enzyme e
        ON tg.hadm_id = e.hadm_id
        AND e.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND e.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
-- NEW: enhanced labs from labevents (ferritin, D-dimer, fibrinogen)
enhanced_labs_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        MAX(CASE WHEN l.itemid = 50924 THEN CAST(l.valuenum AS DOUBLE) END) AS ferritin,
        MAX(CASE WHEN l.itemid IN (50915, 51196, 52551) THEN CAST(l.valuenum AS DOUBLE) END) AS ddimer,
        MAX(CASE WHEN l.itemid IN (51214, 51623, 52116, 52117) THEN CAST(l.valuenum AS DOUBLE) END) AS fibrinogen
    FROM time_grid tg
    LEFT JOIN mimiciv_hosp.labevents l
        ON tg.hadm_id = l.hadm_id
        AND l.itemid IN (50924, 50915, 51196, 52551, 51214, 51623, 52116, 52117)
        AND l.charttime >= tg.grid_time - INTERVAL '6' HOUR
        AND l.charttime < tg.grid_time + INTERVAL '6' HOUR
    GROUP BY tg.stay_id, tg.hr
),
-- NEW: ventilation hourly
vent_hourly AS (
    SELECT
        tg.stay_id,
        tg.hr,
        MAX(CASE WHEN lower(v.ventilation_status) IN ('invasive', 'mechanical', 'tracheostomy') THEN 1 ELSE 0 END) AS mech_vent
    FROM time_grid tg
    LEFT JOIN mimiciv_derived.ventilation v
        ON tg.stay_id = v.stay_id
        AND tg.grid_time >= v.starttime
        AND tg.grid_time < v.endtime
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
    sf.meanbp_min,
    -- enhanced
    bd.lymphocytes_abs,
    bd.lymphocytes_pct,
    bd.monocytes_abs,
    bd.neutrophils_abs,
    ih.crp,
    eh.alt,
    eh.ast,
    eh.bilirubin_total,
    el.ferritin,
    el.ddimer,
    el.fibrinogen,
    vh.mech_vent
FROM time_grid tg
LEFT JOIN vs_hourly vs ON tg.stay_id = vs.stay_id AND tg.hr = vs.hr
LEFT JOIN sofa_hourly sf ON tg.stay_id = sf.stay_id AND tg.hr = sf.hr
LEFT JOIN gcs_hourly gc ON tg.stay_id = gc.stay_id AND tg.hr = gc.hr
LEFT JOIN labs_hourly lb ON tg.stay_id = lb.stay_id AND tg.hr = lb.hr
LEFT JOIN cbc_hourly cbc ON tg.stay_id = cbc.stay_id AND tg.hr = cbc.hr
LEFT JOIN coag_hourly cg ON tg.stay_id = cg.stay_id AND tg.hr = cg.hr
LEFT JOIN bd_hourly bd ON tg.stay_id = bd.stay_id AND tg.hr = bd.hr
LEFT JOIN infl_hourly ih ON tg.stay_id = ih.stay_id AND tg.hr = ih.hr
LEFT JOIN enz_hourly eh ON tg.stay_id = eh.stay_id AND tg.hr = eh.hr
LEFT JOIN enhanced_labs_hourly el ON tg.stay_id = el.stay_id AND tg.hr = el.hr
LEFT JOIN vent_hourly vh ON tg.stay_id = vh.stay_id AND tg.hr = vh.hr
ORDER BY tg.stay_id, tg.hr
"""


# ============================================================
# Main Pipeline
# ============================================================

def build_enhanced_analysis_tables(
    db_path: Path,
    output_dir: Path,
    n_hours: int = 48,
    output_format: str = "parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and export ENHANCED patient-level analysis tables."""
    print("=" * 65)
    print("Building ENHANCED Patient-Level Sepsis Analysis Tables")
    print(f"Database: {db_path}")
    print(f"Time window: {n_hours} hours")
    print(f"Output format: {output_format}")
    print("=" * 65)

    conn = duckdb.connect(str(db_path), read_only=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # --- 1. Static table ---
    print("\n[1/4] Building enhanced static info table...")
    patient_static = conn.execute(ENHANCED_STATIC_SQL).fetchdf()
    print(f"  Shape: {patient_static.shape}")
    print(f"  Sepsis-3 patients: {patient_static['is_sepsis3'].sum()} / {len(patient_static)}")
    print(f"  28-day mortality: {patient_static['mortality_28d'].mean():.1%}")
    print(f"  Blood culture positive: {patient_static['blood_culture_positive'].sum()} / {len(patient_static)}")
    print(f"  Resp culture positive: {patient_static['resp_culture_positive'].sum()} / {len(patient_static)}")
    print(f"  Mech vent first 24h: {patient_static['mech_vent_first24h'].sum()} / {len(patient_static)}")

    # --- 2. Time-series table ---
    print("\n[2/4] Building enhanced time-series table...")
    ts_sql = ENHANCED_TIMESERIES_SQL.replace("{n_hours}", str(n_hours))
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
        static_path = output_dir / "patient_static_enhanced.parquet"
        ts_path = output_dir / "patient_timeseries_enhanced.parquet"
        patient_static.to_parquet(static_path, index=False)
        patient_ts.to_parquet(ts_path, index=False)
    else:
        static_path = output_dir / "patient_static_enhanced.csv"
        ts_path = output_dir / "patient_timeseries_enhanced.csv"
        patient_static.to_csv(static_path, index=False)
        patient_ts.to_csv(ts_path, index=False)

    conn.close()

    elapsed = time.time() - start
    print(f"  Static table: {static_path} ({static_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Time-series table: {ts_path} ({ts_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nDone! Elapsed: {elapsed:.1f}s")

    print("\n" + "=" * 65)
    print("Next step: run subtype_label_engine.py to generate proxy labels")
    print("=" * 65)

    return patient_static, patient_ts


def main():
    parser = argparse.ArgumentParser(description="Build enhanced sepsis analysis tables")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--hours", type=int, default=48, help="Time window length (hours)")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"])
    args = parser.parse_args()

    build_enhanced_analysis_tables(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
        n_hours=args.hours,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
