"""
treatment_features.py - Build Stage 4 treatment-aware hourly feature bundles.

This module keeps the original S0/S1.5 continuous physiology path intact and
adds a parallel treatment tensor that can be fused downstream. The output bundle
is source-agnostic and is intentionally small:

  - treatments.npy            (N, T, K)
  - masks_treatments.npy      (N, T, K)
  - cohort_static.csv         aligned patient-level metadata
  - patient_level_summary.csv compact exposure summary
  - treatment_feature_names.json
  - treatment_report.json
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("s4.treatment_features")

TREATMENT_FEATURES = [
    "vasopressor_on",
    "vasopressor_rate",
    "antibiotic_on",
    "crystalloid_fluid_ml",
    "fluid_bolus_ml",
    "mechanical_vent_on",
    "rrt_on",
]
TREATMENT_INDEX = {name: idx for idx, name in enumerate(TREATMENT_FEATURES)}

MIMIC_FLUID_CATEGORY_CRYSTALLOID = {
    "02-fluids (crystalloids)",
    "03-iv fluid bolus",
}
MIMIC_FLUID_CATEGORY_BOLUS = {"03-iv fluid bolus"}
MIMIC_ANTIBIOTIC_PATTERN = (
    r"vancomycin|cefepime|meropenem|piperacillin|tazobactam|zosyn|ceftriaxone|"
    r"linezolid|levofloxacin|ciprofloxacin|azithromycin|ertapenem|imipenem|"
    r"metronidazole|amikacin|gentamicin"
)
MIMIC_VENT_PATTERN = r"^invasive ventilation$|^mechanically ventilated$"
MIMIC_RRT_PATTERN = r"^hemodialysis$|^dialysis - crrt$|^dialysis - cvvhd$|^dialysis - cvvhdf$|^dialysis - scuf$|^peritoneal dialysis$"

EICU_ANTIBIOTIC_PATTERN = MIMIC_ANTIBIOTIC_PATTERN
EICU_CRYSTALLOID_PATTERN = r"crystalloid|saline|ringer|lactated|ivpb"


def build_treatment_feature_bundle(
    *,
    source: str,
    prepared_dir: Path,
    output_dir: Path,
    raw_dir: Path | None = None,
    n_hours: int = 48,
    max_patients: int | None = None,
    tag: str = "eicu_demo",
    sepsis3_only: bool = True,
) -> dict:
    """
    Build a unified treatment bundle for MIMIC-IV or eICU.

    Parameters
    ----------
    source:
        `mimic` or `eicu`.
    prepared_dir:
        Directory with patient-level prepared tables.
    output_dir:
        Where the bundle will be written.
    raw_dir:
        Raw database directory containing event tables. Required for treatment
        channels not already present in the prepared tables.
    """
    source = source.lower()
    prepared_dir = Path(prepared_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source == "mimic":
        if raw_dir is None:
            raise ValueError("raw_dir is required for MIMIC treatment feature extraction")
        return _build_mimic_bundle(
            prepared_dir=prepared_dir,
            raw_dir=Path(raw_dir),
            output_dir=output_dir,
            n_hours=n_hours,
            max_patients=max_patients,
            sepsis3_only=sepsis3_only,
        )
    if source == "eicu":
        if raw_dir is None:
            raise ValueError("raw_dir is required for eICU treatment feature extraction")
        return _build_eicu_bundle(
            prepared_dir=prepared_dir,
            raw_dir=Path(raw_dir),
            output_dir=output_dir,
            n_hours=n_hours,
            max_patients=max_patients,
            tag=tag,
        )
    raise ValueError(f"Unsupported source: {source}")


def load_treatment_bundle(bundle_dir: Path) -> dict:
    """Load an existing treatment bundle from disk."""
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "treatment_feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(bundle_dir / "treatment_report.json", encoding="utf-8") as f:
        report = json.load(f)
    return {
        "treatments": np.load(bundle_dir / "treatments.npy", mmap_mode="r"),
        "masks_treatments": np.load(bundle_dir / "masks_treatments.npy", mmap_mode="r"),
        "feature_names": feature_names,
        "cohort_static": pd.read_csv(bundle_dir / "cohort_static.csv"),
        "patient_level_summary": pd.read_csv(bundle_dir / "patient_level_summary.csv"),
        "report": report,
    }


def _build_mimic_bundle(
    *,
    prepared_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    n_hours: int,
    max_patients: int | None,
    sepsis3_only: bool,
) -> dict:
    static_path = _first_existing(
        prepared_dir / "patient_static.parquet",
        prepared_dir / "patient_static.csv",
    )
    timeseries_path = _first_existing(
        prepared_dir / "patient_timeseries.parquet",
        prepared_dir / "patient_timeseries.csv",
    )

    con = duckdb.connect()
    try:
        static_source = _duckdb_table_expr(static_path)
        where_clauses = []
        if sepsis3_only:
            where_clauses.append("COALESCE(is_sepsis3, 0) = 1")
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        limit_sql = f"LIMIT {int(max_patients)}" if max_patients is not None else ""
        static = con.execute(
            f"""
            SELECT
                stay_id,
                hadm_id,
                subject_id,
                gender,
                age,
                race,
                hospital_expire_flag AS mortality_inhospital,
                mortality_28d,
                is_sepsis3,
                anchor_time,
                first_day_sofa,
                charlson_comorbidity_index,
                sirs
            FROM {static_source}
            {where_sql}
            ORDER BY stay_id
            {limit_sql}
            """
        ).fetchdf()
        if static.empty:
            raise ValueError("No MIMIC stays selected for treatment feature extraction")

        static["stay_id"] = pd.to_numeric(static["stay_id"], errors="coerce").astype("Int64")
        static["hadm_id"] = pd.to_numeric(static["hadm_id"], errors="coerce").astype("Int64")
        static["anchor_time"] = pd.to_datetime(static["anchor_time"], errors="coerce")
        static = static.dropna(subset=["stay_id", "hadm_id", "anchor_time"]).reset_index(drop=True)

        stay_ids = static["stay_id"].astype(np.int64).tolist()
        stay_to_idx = {stay_id: idx for idx, stay_id in enumerate(stay_ids)}

        treatments, masks = _init_treatment_arrays(len(stay_ids), n_hours)

        con.register("target_static", static[["stay_id", "hadm_id", "anchor_time"]].copy())

        timeseries_source = _duckdb_table_expr(timeseries_path)
        ts = con.execute(
            f"""
            SELECT stay_id, hr, norepi_rate
            FROM {timeseries_source}
            WHERE stay_id IN (SELECT stay_id FROM target_static)
              AND hr BETWEEN 0 AND {n_hours - 1}
            """
        ).fetchdf()
        for row in ts.itertuples(index=False):
            patient_idx = stay_to_idx.get(int(row.stay_id))
            hour = int(row.hr)
            rate = float(row.norepi_rate) if pd.notna(row.norepi_rate) else 0.0
            treatments[patient_idx, hour, TREATMENT_INDEX["vasopressor_rate"]] = rate
            treatments[patient_idx, hour, TREATMENT_INDEX["vasopressor_on"]] = 1.0 if rate > 0 else 0.0
            masks[patient_idx, hour, TREATMENT_INDEX["vasopressor_rate"]] = 1.0
            masks[patient_idx, hour, TREATMENT_INDEX["vasopressor_on"]] = 1.0

        d_items_path = raw_dir / "icu" / "d_items.csv.gz"
        inputevents_path = raw_dir / "icu" / "inputevents.csv.gz"
        procedureevents_path = raw_dir / "icu" / "procedureevents.csv.gz"
        prescriptions_path = raw_dir / "hosp" / "prescriptions.csv.gz"

        if inputevents_path.exists():
            fluids = con.execute(
                f"""
                WITH fluid_events AS (
                    SELECT
                        ie.stay_id,
                        CAST(FLOOR(date_diff('minute', ts.anchor_time, CAST(ie.starttime AS TIMESTAMP)) / 60.0) AS BIGINT) AS hr,
                        COALESCE(ie.totalamount, ie.amount, ie.originalamount, 0.0) AS amount_ml,
                        lower(COALESCE(ie.ordercategoryname, '')) AS ordercategoryname
                    FROM read_csv_auto('{_sql_escape(inputevents_path)}', union_by_name=true) ie
                    INNER JOIN target_static ts USING (stay_id)
                )
                SELECT
                    stay_id,
                    hr,
                    SUM(CASE
                        WHEN ordercategoryname IN ({_sql_list(sorted(MIMIC_FLUID_CATEGORY_CRYSTALLOID))})
                        THEN GREATEST(amount_ml, 0.0) ELSE 0.0 END) AS crystalloid_fluid_ml,
                    SUM(CASE
                        WHEN ordercategoryname IN ({_sql_list(sorted(MIMIC_FLUID_CATEGORY_BOLUS))})
                        THEN GREATEST(amount_ml, 0.0) ELSE 0.0 END) AS fluid_bolus_ml
                FROM fluid_events
                WHERE hr BETWEEN 0 AND {n_hours - 1}
                GROUP BY stay_id, hr
                """
            ).fetchdf()
            for row in fluids.itertuples(index=False):
                patient_idx = stay_to_idx.get(int(row.stay_id))
                hour = int(row.hr)
                treatments[patient_idx, hour, TREATMENT_INDEX["crystalloid_fluid_ml"]] = float(
                    row.crystalloid_fluid_ml or 0.0
                )
                treatments[patient_idx, hour, TREATMENT_INDEX["fluid_bolus_ml"]] = float(
                    row.fluid_bolus_ml or 0.0
                )
            masks[:, :, TREATMENT_INDEX["crystalloid_fluid_ml"]] = 1.0
            masks[:, :, TREATMENT_INDEX["fluid_bolus_ml"]] = 1.0

        if prescriptions_path.exists():
            antibiotics = con.execute(
                f"""
                SELECT
                    pr.hadm_id,
                    CAST(FLOOR(date_diff('minute', ts.anchor_time, CAST(pr.starttime AS TIMESTAMP)) / 60.0) AS BIGINT) AS start_hr,
                    CAST(CEIL(date_diff('minute', ts.anchor_time,
                        COALESCE(CAST(pr.stoptime AS TIMESTAMP), CAST(pr.starttime AS TIMESTAMP))) / 60.0) AS BIGINT) AS end_hr
                FROM read_csv_auto('{_sql_escape(prescriptions_path)}', union_by_name=true) pr
                INNER JOIN target_static ts USING (hadm_id)
                WHERE regexp_matches(lower(COALESCE(pr.drug, '')), '{MIMIC_ANTIBIOTIC_PATTERN}')
                """
            ).fetchdf()
            hadm_to_patients = static.groupby("hadm_id")["stay_id"].apply(list).to_dict()
            for row in antibiotics.itertuples(index=False):
                hadm_id = int(row.hadm_id)
                for stay_id in hadm_to_patients.get(hadm_id, []):
                    _mark_interval(
                        treatments=treatments,
                        masks=masks,
                        patient_idx=stay_to_idx[int(stay_id)],
                        feature_idx=TREATMENT_INDEX["antibiotic_on"],
                        start_hr=row.start_hr,
                        end_hr=row.end_hr,
                        n_hours=n_hours,
                        value=1.0,
                    )
            if len(antibiotics):
                masks[:, :, TREATMENT_INDEX["antibiotic_on"]] = 1.0

        if procedureevents_path.exists() and d_items_path.exists():
            procedures = con.execute(
                f"""
                WITH d_items AS (
                    SELECT itemid, lower(label) AS label
                    FROM read_csv_auto('{_sql_escape(d_items_path)}', union_by_name=true)
                )
                SELECT
                    pe.stay_id,
                    di.label,
                    CAST(FLOOR(date_diff('minute', ts.anchor_time, CAST(pe.starttime AS TIMESTAMP)) / 60.0) AS BIGINT) AS start_hr,
                    CAST(CEIL(date_diff('minute', ts.anchor_time,
                        COALESCE(CAST(pe.endtime AS TIMESTAMP), CAST(pe.starttime AS TIMESTAMP))) / 60.0) AS BIGINT) AS end_hr
                FROM read_csv_auto('{_sql_escape(procedureevents_path)}', union_by_name=true) pe
                INNER JOIN target_static ts USING (stay_id)
                LEFT JOIN d_items di USING (itemid)
                WHERE di.label IS NOT NULL
                  AND (
                    regexp_matches(di.label, '{MIMIC_VENT_PATTERN}')
                    OR regexp_matches(di.label, '{MIMIC_RRT_PATTERN}')
                  )
                """
            ).fetchdf()
            for row in procedures.itertuples(index=False):
                label = str(row.label or "")
                if re.search(MIMIC_VENT_PATTERN, label):
                    feature_idx = TREATMENT_INDEX["mechanical_vent_on"]
                elif re.search(MIMIC_RRT_PATTERN, label):
                    feature_idx = TREATMENT_INDEX["rrt_on"]
                else:
                    continue
                _mark_interval(
                    treatments=treatments,
                    masks=masks,
                    patient_idx=stay_to_idx[int(row.stay_id)],
                    feature_idx=feature_idx,
                    start_hr=row.start_hr,
                    end_hr=row.end_hr,
                    n_hours=n_hours,
                    value=1.0,
                )
            if len(procedures):
                if procedures["label"].str.contains(MIMIC_VENT_PATTERN, regex=True, na=False).any():
                    masks[:, :, TREATMENT_INDEX["mechanical_vent_on"]] = 1.0
                if procedures["label"].str.contains(MIMIC_RRT_PATTERN, regex=True, na=False).any():
                    masks[:, :, TREATMENT_INDEX["rrt_on"]] = 1.0
    finally:
        con.close()

    cohort_static = static.copy()
    cohort_static["patient_id"] = cohort_static["stay_id"].astype(str)
    patient_summary = _build_patient_level_summary(cohort_static, treatments, TREATMENT_FEATURES)
    return _write_bundle(
        output_dir=output_dir,
        source="mimic",
        cohort_static=cohort_static,
        treatments=treatments,
        masks=masks,
        patient_summary=patient_summary,
        metadata={
            "source": "mimic",
            "prepared_dir": str(prepared_dir),
            "raw_dir": str(raw_dir),
            "n_hours": n_hours,
            "n_patients": int(len(cohort_static)),
            "sepsis3_only": bool(sepsis3_only),
        },
    )


def _build_eicu_bundle(
    *,
    prepared_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    n_hours: int,
    max_patients: int | None,
    tag: str,
) -> dict:
    patient_info_path = prepared_dir / f"patient_info_{tag}.csv"
    feature_names_path = prepared_dir / f"feature_names_{tag}.json"
    time_series_path = prepared_dir / f"time_series_{tag}.npy"
    if not patient_info_path.exists():
        raise FileNotFoundError(f"Missing eICU patient info: {patient_info_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Missing eICU feature names: {feature_names_path}")
    if not time_series_path.exists():
        raise FileNotFoundError(f"Missing eICU time series tensor: {time_series_path}")

    cohort_static = pd.read_csv(patient_info_path)
    if max_patients is not None:
        cohort_static = cohort_static.head(int(max_patients)).copy()
    cohort_static = cohort_static.reset_index(drop=True)
    with open(feature_names_path, encoding="utf-8") as f:
        feature_names = json.load(f)
    time_series = np.load(time_series_path, mmap_mode="r")

    n_patients = len(cohort_static)
    treatments, masks = _init_treatment_arrays(n_patients, n_hours)

    idx_map = {name: i for i, name in enumerate(feature_names)}
    for src_name, dst_name in (
        ("vasopressor", "vasopressor_on"),
        ("mechanical_vent", "mechanical_vent_on"),
        ("rrt", "rrt_on"),
    ):
        if src_name not in idx_map:
            continue
        src_idx = idx_map[src_name]
        dst_idx = TREATMENT_INDEX[dst_name]
        arr = np.asarray(time_series[:n_patients, :n_hours, src_idx], dtype=np.float32)
        treatments[:, :, dst_idx] = np.where(np.isfinite(arr), np.maximum(arr, 0.0), 0.0)
        masks[:, :, dst_idx] = 1.0

    medication_path = _first_existing_optional(
        raw_dir / "medication.csv.gz",
        raw_dir / "medication.csv",
    )
    intake_path = _first_existing_optional(
        raw_dir / "intakeOutput.csv.gz",
        raw_dir / "intakeOutput.csv",
        raw_dir / "intakeoutput.csv.gz",
        raw_dir / "intakeoutput.csv",
    )
    if medication_path is not None or intake_path is not None:
        con = duckdb.connect()
        try:
            target = cohort_static[["stay_id"]].copy()
            target["stay_id"] = pd.to_numeric(target["stay_id"], errors="coerce").astype("Int64")
            target = target.dropna().astype({"stay_id": np.int64})
            con.register("target_stays", target)
            stay_to_idx = {int(stay_id): idx for idx, stay_id in enumerate(target["stay_id"].tolist())}

            if medication_path is not None:
                antibiotics = con.execute(
                    f"""
                    SELECT
                        patientunitstayid AS stay_id,
                        CAST(FLOOR(COALESCE(drugstartoffset, drugorderoffset, 0) / 60.0) AS BIGINT) AS start_hr,
                        CAST(CEIL(COALESCE(drugstopoffset, drugstartoffset, drugorderoffset, 0) / 60.0) AS BIGINT) AS end_hr
                    FROM read_csv_auto('{_sql_escape(medication_path)}', union_by_name=true)
                    WHERE patientunitstayid IN (SELECT stay_id FROM target_stays)
                      AND regexp_matches(lower(COALESCE(drugname, '')), '{EICU_ANTIBIOTIC_PATTERN}')
                    """
                ).fetchdf()
                for row in antibiotics.itertuples(index=False):
                    _mark_interval(
                        treatments=treatments,
                        masks=masks,
                        patient_idx=stay_to_idx[int(row.stay_id)],
                        feature_idx=TREATMENT_INDEX["antibiotic_on"],
                        start_hr=row.start_hr,
                        end_hr=row.end_hr,
                        n_hours=n_hours,
                        value=1.0,
                    )
                if len(antibiotics):
                    masks[:, :, TREATMENT_INDEX["antibiotic_on"]] = 1.0

            if intake_path is not None:
                fluids = con.execute(
                    f"""
                    SELECT
                        patientunitstayid AS stay_id,
                        CAST(FLOOR(COALESCE(intakeoutputoffset, 0) / 60.0) AS BIGINT) AS hr,
                        SUM(
                            CASE WHEN regexp_matches(lower(COALESCE(cellpath, '') || ' ' || COALESCE(celllabel, '')), '{EICU_CRYSTALLOID_PATTERN}')
                            THEN GREATEST(COALESCE(cellvaluenumeric, 0.0), 0.0) ELSE 0.0 END
                        ) AS crystalloid_fluid_ml,
                        SUM(
                            CASE WHEN regexp_matches(lower(COALESCE(cellpath, '') || ' ' || COALESCE(celllabel, '')), '{EICU_CRYSTALLOID_PATTERN}')
                                  AND COALESCE(cellvaluenumeric, 0.0) >= 250.0
                            THEN GREATEST(COALESCE(cellvaluenumeric, 0.0), 0.0) ELSE 0.0 END
                        ) AS fluid_bolus_ml
                    FROM read_csv_auto('{_sql_escape(intake_path)}', union_by_name=true)
                    WHERE patientunitstayid IN (SELECT stay_id FROM target_stays)
                      AND COALESCE(intakeoutputoffset, 0) >= 0
                    GROUP BY stay_id, hr
                    HAVING hr BETWEEN 0 AND {n_hours - 1}
                    """
                ).fetchdf()
                for row in fluids.itertuples(index=False):
                    patient_idx = stay_to_idx[int(row.stay_id)]
                    hour = int(row.hr)
                    treatments[patient_idx, hour, TREATMENT_INDEX["crystalloid_fluid_ml"]] = float(
                        row.crystalloid_fluid_ml or 0.0
                    )
                    treatments[patient_idx, hour, TREATMENT_INDEX["fluid_bolus_ml"]] = float(
                        row.fluid_bolus_ml or 0.0
                    )
                if len(fluids):
                    masks[:, :, TREATMENT_INDEX["crystalloid_fluid_ml"]] = 1.0
                    masks[:, :, TREATMENT_INDEX["fluid_bolus_ml"]] = 1.0
        finally:
            con.close()

    cohort_static["patient_id"] = cohort_static["stay_id"].astype(str)
    patient_summary = _build_patient_level_summary(cohort_static, treatments, TREATMENT_FEATURES)
    return _write_bundle(
        output_dir=output_dir,
        source="eicu",
        cohort_static=cohort_static,
        treatments=treatments,
        masks=masks,
        patient_summary=patient_summary,
        metadata={
            "source": "eicu",
            "prepared_dir": str(prepared_dir),
            "raw_dir": str(raw_dir),
            "tag": tag,
            "n_hours": n_hours,
            "n_patients": int(len(cohort_static)),
        },
    )


def _write_bundle(
    *,
    output_dir: Path,
    source: str,
    cohort_static: pd.DataFrame,
    treatments: np.ndarray,
    masks: np.ndarray,
    patient_summary: pd.DataFrame,
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "treatments.npy", treatments.astype(np.float32))
    np.save(output_dir / "masks_treatments.npy", masks.astype(np.float32))
    cohort_static.to_csv(output_dir / "cohort_static.csv", index=False)
    patient_summary.to_csv(output_dir / "patient_level_summary.csv", index=False)
    with open(output_dir / "treatment_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(TREATMENT_FEATURES, f, ensure_ascii=False, indent=2)

    report = {
        **metadata,
        "feature_names": TREATMENT_FEATURES,
        "feature_exposure_rate": {},
        "feature_mean_value": {},
        "artifacts": {
            "treatments": str(output_dir / "treatments.npy"),
            "masks_treatments": str(output_dir / "masks_treatments.npy"),
            "cohort_static": str(output_dir / "cohort_static.csv"),
            "patient_level_summary": str(output_dir / "patient_level_summary.csv"),
            "feature_names": str(output_dir / "treatment_feature_names.json"),
        },
    }
    for feature_idx, feature_name in enumerate(TREATMENT_FEATURES):
        arr = treatments[:, :, feature_idx]
        report["feature_mean_value"][feature_name] = round(float(arr.mean()), 4)
        if feature_name.endswith("_ml") or feature_name.endswith("_rate"):
            exposure = arr > 0
        else:
            exposure = arr >= 0.5
        report["feature_exposure_rate"][feature_name] = round(float(exposure.any(axis=1).mean()), 4)

    report_path = output_dir / "treatment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)
    logger.info("Saved treatment bundle for %s: %s", source, output_dir)
    return report


def _build_patient_level_summary(
    cohort_static: pd.DataFrame,
    treatments: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    summary = cohort_static.copy()
    for feature_idx, feature_name in enumerate(feature_names):
        arr = treatments[:, :, feature_idx]
        positive = arr > 0 if (feature_name.endswith("_ml") or feature_name.endswith("_rate")) else arr >= 0.5

        any_name = f"{feature_name}_any_24h"
        first_name = f"{feature_name}_first_hour"
        total_name = f"{feature_name}_total_24h"
        duration_name = f"{feature_name}_hours_on"

        summary[any_name] = positive.any(axis=1).astype(int)
        first = []
        for row in positive:
            hits = np.flatnonzero(row)
            first.append(int(hits[0]) if len(hits) else np.nan)
        summary[first_name] = first
        summary[total_name] = arr.sum(axis=1)
        summary[duration_name] = positive.sum(axis=1)

    return summary


def _init_treatment_arrays(n_patients: int, n_hours: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((n_patients, n_hours, len(TREATMENT_FEATURES)), dtype=np.float32),
        np.zeros((n_patients, n_hours, len(TREATMENT_FEATURES)), dtype=np.float32),
    )


def _mark_interval(
    *,
    treatments: np.ndarray,
    masks: np.ndarray,
    patient_idx: int,
    feature_idx: int,
    start_hr: float | int | None,
    end_hr: float | int | None,
    n_hours: int,
    value: float,
) -> None:
    if start_hr is None or pd.isna(start_hr):
        return
    s = max(int(math.floor(float(start_hr))), 0)
    e_raw = s + 1 if end_hr is None or pd.isna(end_hr) else int(math.ceil(float(end_hr)))
    e = min(max(e_raw, s + 1), n_hours)
    if s >= n_hours or e <= 0:
        return
    treatments[patient_idx, s:e, feature_idx] = value
    masks[patient_idx, s:e, feature_idx] = 1.0


def _duckdb_table_expr(path: Path) -> str:
    path = Path(path)
    if path.suffix == ".parquet":
        return f"read_parquet('{_sql_escape(path)}')"
    return f"read_csv_auto('{_sql_escape(path)}', union_by_name=true)"


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate files exist: {[str(c) for c in candidates]}")


def _first_existing_optional(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _sql_escape(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _sql_list(values: list[str]) -> str:
    escaped = []
    for value in values:
        escaped.append("'" + value.replace("'", "''") + "'")
    return ", ".join(escaped)
