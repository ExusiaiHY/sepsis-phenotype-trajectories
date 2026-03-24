"""
eicu_loader.py - eICU raw/demo dataset loader for the legacy V1 pipeline.

This module converts selected eICU CSV tables into the same standardized
3D tensor + patient_info interface used by the rest of the project.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import resolve_path, setup_logger

logger = setup_logger(__name__)

TABLE_CANDIDATES = {
    "patient": ["patient.csv.gz", "patient.csv"],
    "lab": ["lab.csv.gz", "lab.csv"],
    "vitalperiodic": ["vitalPeriodic.csv.gz", "vitalPeriodic.csv", "vitalperiodic.csv.gz", "vitalperiodic.csv"],
    "vitalaperiodic": ["vitalAperiodic.csv.gz", "vitalAperiodic.csv", "vitalaperiodic.csv.gz", "vitalaperiodic.csv"],
    "infusiondrug": ["infusionDrug.csv.gz", "infusionDrug.csv", "infusiondrug.csv.gz", "infusiondrug.csv"],
    "intakeoutput": ["intakeOutput.csv.gz", "intakeOutput.csv", "intakeoutput.csv.gz", "intakeoutput.csv"],
    "respiratorycare": ["respiratoryCare.csv.gz", "respiratoryCare.csv", "respiratorycare.csv.gz", "respiratorycare.csv"],
    "apacheapsvar": ["apacheApsVar.csv.gz", "apacheApsVar.csv", "apacheapsvar.csv.gz", "apacheapsvar.csv"],
    "apachepatientresult": [
        "apachePatientResult.csv.gz",
        "apachePatientResult.csv",
        "apachepatientresult.csv.gz",
        "apachepatientresult.csv",
    ],
    "apachepredvar": ["apachePredVar.csv.gz", "apachePredVar.csv", "apachepredvar.csv.gz", "apachepredvar.csv"],
}

VASOPRESSOR_PATTERN = r"norepinephrine|epinephrine|vasopressin|phenylephrine|dopamine|dobutamine"

TABLE_REQUIRED_COLUMNS = {
    "patient": {
        "patientunitstayid",
        "uniquepid",
        "gender",
        "age",
        "hospitalid",
        "unittype",
        "hospitaldischargestatus",
        "unitdischargeoffset",
        "apacheadmissiondx",
    },
    "lab": {"patientunitstayid", "labresultoffset", "labname", "labresult"},
    "vitalperiodic": {
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
        "respiration",
        "sao2",
        "temperature",
    },
    "vitalaperiodic": {
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivediastolic",
        "noninvasivemean",
    },
    "infusiondrug": {"patientunitstayid", "infusionoffset", "drugname", "drugrate"},
    "intakeoutput": {"patientunitstayid", "intakeoutputoffset", "dialysistotal", "celllabel", "cellpath"},
    "respiratorycare": {
        "patientunitstayid",
        "respcarestatusoffset",
        "ventstartoffset",
        "ventendoffset",
        "airwaytype",
    },
    "apacheapsvar": {
        "patientunitstayid",
        "vent",
        "dialysis",
        "heartrate",
        "meanbp",
        "temperature",
        "respiratoryrate",
        "creatinine",
        "bilirubin",
        "wbc",
        "pao2",
        "fio2",
    },
    "apachepredvar": {"patientunitstayid", "ventday1"},
}


def load_eicu_from_config(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Load eICU raw/demo files according to config."""
    eicu_cfg = config["data"].get("eicu", {})
    data_dir_value = eicu_cfg.get("data_dir") or eicu_cfg.get("demo_dir") or "data/external/eicu_demo"
    data_dir = resolve_path(data_dir_value)
    n_timesteps = int(eicu_cfg.get("n_timesteps", config["data"]["simulated"]["n_timesteps"]))
    max_patients = eicu_cfg.get("max_patients")
    feature_names = (
        config["variables"]["vitals"]
        + config["variables"]["labs"]
        + config["variables"]["treatments"]
    )
    return load_eicu_dataset(
        data_dir=data_dir,
        feature_names=feature_names,
        n_timesteps=n_timesteps,
        max_patients=max_patients,
    )


def load_eicu_dataset(
    data_dir: Path,
    feature_names: list[str],
    n_timesteps: int = 48,
    max_patients: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Convert selected raw eICU tables into a patient tensor and patient_info table.

    The loader is intentionally tolerant to partially available demo data:
    it requires `patient.csv*`, but other tables are optional and act as
    incremental feature sources or fallbacks.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"eICU data directory not found: {data_dir}\n"
            "Place the demo/full CSV files in this directory and set data.eicu.data_dir accordingly."
        )

    patient = _read_table(data_dir, "patient")
    if patient.empty:
        raise FileNotFoundError(
            f"Could not find eICU patient table under {data_dir}.\n"
            f"Supported filenames: {', '.join(TABLE_CANDIDATES['patient'])}"
        )

    patient = _prepare_patient_table(patient)
    if max_patients is not None:
        patient = patient.head(int(max_patients)).copy()

    stay_ids = patient["patientunitstayid"].astype(int).tolist()
    stay_to_idx = {stay_id: idx for idx, stay_id in enumerate(stay_ids)}
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    tensor = np.full((len(stay_ids), n_timesteps, len(feature_names)), np.nan, dtype=float)
    for treatment in ("vasopressor", "mechanical_vent", "rrt"):
        if treatment in feature_to_idx:
            tensor[:, :, feature_to_idx[treatment]] = 0.0

    _load_periodic_vitals(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps)
    _load_aperiodic_vitals(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps)
    _load_labs(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps)
    aps = _load_apache_fallbacks(data_dir, stay_to_idx, feature_to_idx, tensor)
    shock_binary = _load_treatments(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps, aps)

    patient_info = _build_patient_info(patient, shock_binary)

    logger.info("eICU data loaded: %s", tensor.shape)
    logger.info(
        "  Patients: %d, mortality rate: %.1f%%",
        len(patient_info),
        100.0 * float(patient_info["mortality_28d"].mean()) if len(patient_info) else 0.0,
    )

    return tensor, patient_info


def prepare_eicu_demo_artifacts(config: dict, output_dir: Path, tag: str = "eicu_demo") -> dict:
    """Load eICU raw/demo files and export cached artifacts plus a small readiness report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor, patient_info = load_eicu_from_config(config)
    feature_names = (
        config["variables"]["vitals"]
        + config["variables"]["labs"]
        + config["variables"]["treatments"]
    )
    data_dir = resolve_path(
        config["data"].get("eicu", {}).get("data_dir")
        or config["data"].get("eicu", {}).get("demo_dir")
        or "data/external/eicu_demo"
    )

    tensor_path = output_dir / f"time_series_{tag}.npy"
    patient_path = output_dir / f"patient_info_{tag}.csv"
    features_path = output_dir / f"feature_names_{tag}.json"

    np.save(tensor_path, tensor)
    patient_info.to_csv(patient_path, index=False)
    features_path.write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "source": "eicu",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "n_patients": int(tensor.shape[0]),
        "n_timesteps": int(tensor.shape[1]),
        "n_features": int(tensor.shape[2]),
        "mortality_rate": round(float(patient_info["mortality_28d"].mean()), 4) if len(patient_info) else 0.0,
        "shock_rate": round(float(patient_info["shock_onset"].mean()), 4) if len(patient_info) else 0.0,
        "tables_found": _detect_available_tables(data_dir),
        "feature_missing_rate": {
            feature: round(float(np.isnan(tensor[:, :, idx]).mean()), 4)
            for idx, feature in enumerate(feature_names)
        },
        "artifacts": {
            "time_series": str(tensor_path),
            "patient_info": str(patient_path),
            "feature_names": str(features_path),
        },
    }
    report_path = output_dir / "eicu_demo_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _detect_available_tables(data_dir: Path) -> dict[str, str | None]:
    detected: dict[str, str | None] = {}
    for table_name, candidates in TABLE_CANDIDATES.items():
        matched = None
        for candidate in candidates:
            candidate_path = data_dir / candidate
            if candidate_path.exists():
                matched = candidate_path.name
                break
        detected[table_name] = matched
    return detected


def _read_table(data_dir: Path, table_name: str) -> pd.DataFrame:
    for candidate in TABLE_CANDIDATES[table_name]:
        path = data_dir / candidate
        if path.exists():
            required = TABLE_REQUIRED_COLUMNS.get(table_name)
            usecols = None
            if required:
                usecols = lambda col: str(col).strip().lower() in required
            logger.info("Reading eICU table %s from %s", table_name, path.name)
            df = pd.read_csv(path, low_memory=False, usecols=usecols)
            df.columns = [str(col).strip().lower() for col in df.columns]
            logger.info("Loaded eICU table %s: %s rows x %s cols", table_name, len(df), len(df.columns))
            return df
    return pd.DataFrame()


def _prepare_patient_table(patient: pd.DataFrame) -> pd.DataFrame:
    patient = patient.copy()
    patient["patientunitstayid"] = pd.to_numeric(patient["patientunitstayid"], errors="coerce")
    patient = patient.dropna(subset=["patientunitstayid"])
    patient["patientunitstayid"] = patient["patientunitstayid"].astype(int)
    patient = patient.drop_duplicates(subset=["patientunitstayid"]).sort_values("patientunitstayid").reset_index(drop=True)
    return patient


def _build_patient_info(patient: pd.DataFrame, shock_binary: pd.Series) -> pd.DataFrame:
    age_text = patient.get("age", pd.Series(index=patient.index, dtype=object)).fillna("").astype(str).str.strip()
    age = pd.to_numeric(age_text.str.extract(r"(\d+)")[0], errors="coerce")
    gender_text = patient.get("gender", pd.Series(index=patient.index, dtype=object)).fillna("").astype(str).str.lower()
    gender = np.where(
        gender_text.str.contains("female", na=False),
        "F",
        np.where(gender_text.str.contains("male", na=False), "M", "U"),
    )
    mortality = patient.get("hospitaldischargestatus", pd.Series(index=patient.index, dtype=object)).fillna("").astype(str)
    mortality = mortality.str.lower().str.contains("expired", na=False).astype(int)
    icu_los = pd.to_numeric(patient.get("unitdischargeoffset", np.nan), errors="coerce") / 60.0
    apachescore = pd.to_numeric(patient.get("apacheadmissiondx", np.nan), errors="coerce")

    info = pd.DataFrame(
        {
            "patient_id": patient["patientunitstayid"].astype(str),
            "stay_id": patient["patientunitstayid"].astype(int),
            "uniquepid": patient.get("uniquepid"),
            "hospitalid": pd.to_numeric(patient.get("hospitalid", np.nan), errors="coerce"),
            "unittype": patient.get("unittype"),
            "age": age,
            "gender": gender,
            "mortality_28d": mortality,
            "mortality_source": "hospitaldischargestatus_proxy",
            "icu_los": icu_los,
            "shock_onset": shock_binary.astype(int).reindex(patient["patientunitstayid"].astype(int)).fillna(0).astype(int).values,
            "apache_admission_dx_numeric": apachescore,
            "data_source": "eicu",
        }
    )
    return info


def _prepare_hourly_table(
    df: pd.DataFrame,
    stay_to_idx: dict[int, int],
    offset_col: str,
    n_timesteps: int,
) -> pd.DataFrame:
    if df.empty or offset_col not in df.columns or "patientunitstayid" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["patientunitstayid"] = pd.to_numeric(working["patientunitstayid"], errors="coerce")
    working[offset_col] = pd.to_numeric(working[offset_col], errors="coerce")
    working = working.dropna(subset=["patientunitstayid", offset_col])
    working["patientunitstayid"] = working["patientunitstayid"].astype(int)
    working = working[working["patientunitstayid"].isin(stay_to_idx)]
    working["hour"] = np.floor_divide(working[offset_col].astype(int), 60)
    working = working[(working["hour"] >= 0) & (working["hour"] < n_timesteps)]
    return working


def _write_hourly_feature(
    tensor: np.ndarray,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    df: pd.DataFrame,
    value_col: str,
    feature_name: str,
    fill_only_missing: bool = False,
) -> None:
    if feature_name not in feature_to_idx or df.empty or value_col not in df.columns:
        return
    feature_idx = feature_to_idx[feature_name]
    grouped = df[["patientunitstayid", "hour", value_col]].copy()
    grouped[value_col] = pd.to_numeric(grouped[value_col], errors="coerce")
    grouped = grouped.dropna()
    if grouped.empty:
        return
    grouped = grouped.groupby(["patientunitstayid", "hour"], as_index=False)[value_col].mean()
    for row in grouped.itertuples(index=False):
        stay_idx = stay_to_idx.get(int(row.patientunitstayid))
        if stay_idx is None:
            continue
        if fill_only_missing and not np.isnan(tensor[stay_idx, int(row.hour), feature_idx]):
            continue
        tensor[stay_idx, int(row.hour), feature_idx] = float(getattr(row, value_col))


def _write_baseline_feature(
    tensor: np.ndarray,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    df: pd.DataFrame,
    value_col: str,
    feature_name: str,
) -> None:
    if feature_name not in feature_to_idx or df.empty or value_col not in df.columns:
        return
    feature_idx = feature_to_idx[feature_name]
    baseline = df[["patientunitstayid", value_col]].copy()
    baseline["patientunitstayid"] = pd.to_numeric(baseline["patientunitstayid"], errors="coerce")
    baseline[value_col] = pd.to_numeric(baseline[value_col], errors="coerce")
    baseline = baseline.dropna()
    baseline["patientunitstayid"] = baseline["patientunitstayid"].astype(int)
    baseline = baseline.drop_duplicates(subset=["patientunitstayid"], keep="first")
    for row in baseline.itertuples(index=False):
        stay_idx = stay_to_idx.get(int(row.patientunitstayid))
        if stay_idx is None or not np.isnan(tensor[stay_idx, 0, feature_idx]):
            continue
        tensor[stay_idx, 0, feature_idx] = float(getattr(row, value_col))


def _load_periodic_vitals(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
) -> None:
    vitals = _read_table(data_dir, "vitalperiodic")
    vitals = _prepare_hourly_table(vitals, stay_to_idx, "observationoffset", n_timesteps)
    mapping = {
        "heartrate": "heart_rate",
        "systemicsystolic": "sbp",
        "systemicdiastolic": "dbp",
        "systemicmean": "map",
        "respiration": "resp_rate",
        "sao2": "spo2",
        "temperature": "temperature",
    }
    for source_col, feature_name in mapping.items():
        _write_hourly_feature(tensor, stay_to_idx, feature_to_idx, vitals, source_col, feature_name)


def _load_aperiodic_vitals(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
) -> None:
    vitals = _read_table(data_dir, "vitalaperiodic")
    vitals = _prepare_hourly_table(vitals, stay_to_idx, "observationoffset", n_timesteps)
    mapping = {
        "noninvasivesystolic": "sbp",
        "noninvasivediastolic": "dbp",
        "noninvasivemean": "map",
    }
    for source_col, feature_name in mapping.items():
        _write_hourly_feature(
            tensor,
            stay_to_idx,
            feature_to_idx,
            vitals,
            source_col,
            feature_name,
            fill_only_missing=True,
        )


def _load_labs(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
) -> None:
    labs = _read_table(data_dir, "lab")
    labs = _prepare_hourly_table(labs, stay_to_idx, "labresultoffset", n_timesteps)
    if labs.empty or "labname" not in labs.columns or "labresult" not in labs.columns:
        return
    labs["labname_norm"] = labs["labname"].fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "lactate": "lactate",
        "creatinine": "creatinine",
        "total bilirubin": "bilirubin",
        "bilirubin": "bilirubin",
        "platelets x 1000": "platelet",
        "wbc x 1000": "wbc",
        "pt - inr": "inr",
    }
    for lab_name, feature_name in mapping.items():
        subset = labs.loc[labs["labname_norm"] == lab_name]
        _write_hourly_feature(tensor, stay_to_idx, feature_to_idx, subset, "labresult", feature_name)


def _load_apache_fallbacks(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
) -> pd.DataFrame:
    aps = _read_table(data_dir, "apacheapsvar")
    if aps.empty:
        return pd.DataFrame()
    aps["patientunitstayid"] = pd.to_numeric(aps.get("patientunitstayid"), errors="coerce")
    aps = aps.dropna(subset=["patientunitstayid"])
    aps["patientunitstayid"] = aps["patientunitstayid"].astype(int)
    aps = aps[aps["patientunitstayid"].isin(stay_to_idx)].drop_duplicates(subset=["patientunitstayid"])

    baseline_map = {
        "heartrate": "heart_rate",
        "meanbp": "map",
        "temperature": "temperature",
        "respiratoryrate": "resp_rate",
        "creatinine": "creatinine",
        "bilirubin": "bilirubin",
        "wbc": "wbc",
    }
    for source_col, feature_name in baseline_map.items():
        _write_baseline_feature(tensor, stay_to_idx, feature_to_idx, aps, source_col, feature_name)

    if "pao2_fio2" in feature_to_idx and {"pao2", "fio2"}.issubset(aps.columns):
        ratio = aps[["patientunitstayid", "pao2", "fio2"]].copy()
        ratio["pao2"] = pd.to_numeric(ratio["pao2"], errors="coerce")
        ratio["fio2"] = pd.to_numeric(ratio["fio2"], errors="coerce")
        ratio = ratio.dropna()
        if not ratio.empty:
            fio2_frac = pd.Series(
                np.where(ratio["fio2"] > 1.0, ratio["fio2"] / 100.0, ratio["fio2"]),
                index=ratio.index,
            ).replace(0, np.nan)
            ratio["pao2_fio2"] = ratio["pao2"] / fio2_frac
            _write_baseline_feature(tensor, stay_to_idx, feature_to_idx, ratio, "pao2_fio2", "pao2_fio2")

    return aps


def _load_treatments(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
    aps: pd.DataFrame,
) -> pd.Series:
    shock_binary = pd.Series(0, index=pd.Index(stay_to_idx.keys(), name="patientunitstayid"), dtype=int)
    _load_vasopressors(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps, shock_binary)
    _load_mechanical_vent(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps, aps)
    _load_rrt(data_dir, stay_to_idx, feature_to_idx, tensor, n_timesteps, aps)
    return shock_binary


def _load_vasopressors(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
    shock_binary: pd.Series,
) -> None:
    if "vasopressor" not in feature_to_idx:
        return
    infusions = _read_table(data_dir, "infusiondrug")
    infusions = _prepare_hourly_table(infusions, stay_to_idx, "infusionoffset", n_timesteps)
    if infusions.empty or "drugname" not in infusions.columns:
        return
    infusions["drugname_norm"] = infusions["drugname"].fillna("").astype(str).str.lower()
    vaso = infusions.loc[infusions["drugname_norm"].str.contains(VASOPRESSOR_PATTERN, regex=True, na=False)]
    if vaso.empty:
        return
    earliest = vaso.groupby("patientunitstayid")["hour"].min()
    feature_idx = feature_to_idx["vasopressor"]
    for stay_id, hour in earliest.items():
        stay_idx = stay_to_idx.get(int(stay_id))
        if stay_idx is None:
            continue
        tensor[stay_idx, int(hour):, feature_idx] = 1.0
        shock_binary.loc[int(stay_id)] = 1


def _load_mechanical_vent(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
    aps: pd.DataFrame,
) -> None:
    if "mechanical_vent" not in feature_to_idx:
        return
    feature_idx = feature_to_idx["mechanical_vent"]
    resp = _read_table(data_dir, "respiratorycare")
    if not resp.empty and "patientunitstayid" in resp.columns:
        resp["patientunitstayid"] = pd.to_numeric(resp["patientunitstayid"], errors="coerce")
        resp["ventstartoffset"] = pd.to_numeric(resp.get("ventstartoffset"), errors="coerce")
        resp["ventendoffset"] = pd.to_numeric(resp.get("ventendoffset"), errors="coerce")
        resp["respcarestatusoffset"] = pd.to_numeric(resp.get("respcarestatusoffset"), errors="coerce")
        resp = resp.dropna(subset=["patientunitstayid"])
        resp["patientunitstayid"] = resp["patientunitstayid"].astype(int)
        resp = resp[resp["patientunitstayid"].isin(stay_to_idx)]
        start_offset = resp["ventstartoffset"].fillna(resp["respcarestatusoffset"]).clip(lower=0)
        end_offset = resp["ventendoffset"].fillna(resp["ventstartoffset"]).fillna(resp["respcarestatusoffset"]).clip(lower=0)
        resp["start_hour"] = np.floor_divide(start_offset.astype(int), 60)
        resp["end_hour"] = np.floor_divide(end_offset.astype(int), 60) + 1
        for row in resp.itertuples(index=False):
            stay_idx = stay_to_idx.get(int(row.patientunitstayid))
            if stay_idx is None:
                continue
            start_hour = max(0, int(getattr(row, "start_hour", 0)))
            end_hour = min(n_timesteps, max(start_hour + 1, int(getattr(row, "end_hour", start_hour + 1))))
            tensor[stay_idx, start_hour:end_hour, feature_idx] = 1.0

    pred = _read_table(data_dir, "apachepredvar")
    if not pred.empty and {"patientunitstayid", "ventday1"}.issubset(pred.columns):
        pred["patientunitstayid"] = pd.to_numeric(pred["patientunitstayid"], errors="coerce")
        pred["ventday1"] = pd.to_numeric(pred["ventday1"], errors="coerce")
        pred = pred.dropna(subset=["patientunitstayid", "ventday1"])
        pred["patientunitstayid"] = pred["patientunitstayid"].astype(int)
        for stay_id in pred.loc[pred["ventday1"] > 0, "patientunitstayid"].unique():
            stay_idx = stay_to_idx.get(int(stay_id))
            if stay_idx is None or np.any(tensor[stay_idx, :, feature_idx] == 1.0):
                continue
            tensor[stay_idx, :, feature_idx] = 1.0

    if not aps.empty and {"patientunitstayid", "vent"}.issubset(aps.columns):
        vent_flag = pd.to_numeric(aps["vent"], errors="coerce")
        for stay_id in aps.loc[vent_flag > 0, "patientunitstayid"].astype(int).tolist():
            stay_idx = stay_to_idx.get(stay_id)
            if stay_idx is None or np.any(tensor[stay_idx, :, feature_idx] == 1.0):
                continue
            tensor[stay_idx, :, feature_idx] = 1.0


def _load_rrt(
    data_dir: Path,
    stay_to_idx: dict[int, int],
    feature_to_idx: dict[str, int],
    tensor: np.ndarray,
    n_timesteps: int,
    aps: pd.DataFrame,
) -> None:
    if "rrt" not in feature_to_idx:
        return
    feature_idx = feature_to_idx["rrt"]
    io = _read_table(data_dir, "intakeoutput")
    io = _prepare_hourly_table(io, stay_to_idx, "intakeoutputoffset", n_timesteps)
    if not io.empty:
        dialysistotal = pd.to_numeric(io.get("dialysistotal"), errors="coerce")
        text_mask = (
            io.get("celllabel", pd.Series("", index=io.index)).fillna("").astype(str).str.lower().str.contains("dialysis", na=False)
            | io.get("cellpath", pd.Series("", index=io.index)).fillna("").astype(str).str.lower().str.contains("dialysis", na=False)
        )
        mask = (dialysistotal > 0).fillna(False) | text_mask
        if mask.any():
            earliest = io.loc[mask].groupby("patientunitstayid")["hour"].min()
            for stay_id, hour in earliest.items():
                stay_idx = stay_to_idx.get(int(stay_id))
                if stay_idx is None:
                    continue
                tensor[stay_idx, int(hour):, feature_idx] = 1.0

    if not aps.empty and {"patientunitstayid", "dialysis"}.issubset(aps.columns):
        dialysis_flag = pd.to_numeric(aps["dialysis"], errors="coerce")
        for stay_id in aps.loc[dialysis_flag > 0, "patientunitstayid"].astype(int).tolist():
            stay_idx = stay_to_idx.get(stay_id)
            if stay_idx is None or np.any(tensor[stay_idx, :, feature_idx] == 1.0):
                continue
            tensor[stay_idx, :, feature_idx] = 1.0
