"""
data_loader.py - Data loading module

Responsibilities:
1. Load data from different sources (simulated / MIMIC / eICU) based on config
2. Provide standardized output format:
   - time_series_3d: np.ndarray, shape (n_patients, n_timesteps, n_features)
   - patient_info: pd.DataFrame with patient-level static info and outcomes
3. Simulated data generator: produces clinically plausible ICU sepsis time series
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from utils import load_config, set_global_seed, setup_logger, resolve_path, timer

logger = setup_logger(__name__)

# Sepsis 2019 import
try:
    from load_sepsis2019 import load_sepsis2019, PROJECT_FEATURES as SEPSIS2019_FEATURES
except ImportError:
    pass

try:
    from eicu_loader import load_eicu_from_config
except ImportError:
    pass


# ============================================================
# Clinical Reference Ranges (for simulated data)
# ============================================================

VITAL_RANGES = {
    "heart_rate":   (80, 15),     # bpm
    "sbp":          (120, 20),    # mmHg
    "dbp":          (70, 12),     # mmHg
    "map":          (85, 12),     # mmHg
    "resp_rate":    (16, 4),      # /min
    "spo2":         (96, 2),      # %
    "temperature":  (37.0, 0.5),  # Celsius
}

LAB_RANGES = {
    "lactate":      (1.5, 0.8),   # mmol/L
    "creatinine":   (1.0, 0.4),   # mg/dL
    "bilirubin":    (1.0, 0.6),   # mg/dL
    "platelet":     (250, 80),    # x10^3/uL
    "wbc":          (8.0, 3.0),   # x10^3/uL
    "pao2_fio2":    (350, 80),    # mmHg
    "inr":          (1.1, 0.2),   # ratio
}


# ============================================================
# Sepsis Subtype Definitions (for simulated data)
# ============================================================
# Based on the 4 common sepsis phenotypes from the literature:
# alpha: low inflammation, low organ failure, low mortality
# beta:  elderly, chronic comorbidities, moderate organ failure
# gamma: high inflammation, respiratory failure dominant
# delta: multi-organ failure, high mortality, shock

SUBTYPE_PROFILES = {
    0: {  # alpha - mild
        "name": "alpha",
        "mortality_prob": 0.05,
        "shock_prob": 0.05,
        "icu_los_mean": 48,
        "vital_shift": {"heart_rate": 5, "sbp": -5, "map": -3, "temperature": 0.3},
        "lab_shift": {"lactate": 0.3, "creatinine": 0.1, "wbc": 2},
        "treatment_prob": {"vasopressor": 0.05, "mechanical_vent": 0.10, "rrt": 0.02},
    },
    1: {  # beta - chronic comorbidities
        "name": "beta",
        "mortality_prob": 0.15,
        "shock_prob": 0.15,
        "icu_los_mean": 96,
        "vital_shift": {"heart_rate": 10, "sbp": -15, "map": -8, "temperature": 0.5},
        "lab_shift": {"lactate": 0.8, "creatinine": 0.8, "wbc": 4, "bilirubin": 0.5},
        "treatment_prob": {"vasopressor": 0.20, "mechanical_vent": 0.25, "rrt": 0.10},
    },
    2: {  # gamma - respiratory failure dominant
        "name": "gamma",
        "mortality_prob": 0.25,
        "shock_prob": 0.20,
        "icu_los_mean": 120,
        "vital_shift": {"heart_rate": 15, "resp_rate": 8, "spo2": -5, "temperature": 1.0},
        "lab_shift": {"lactate": 1.5, "pao2_fio2": -120, "wbc": 6},
        "treatment_prob": {"vasopressor": 0.15, "mechanical_vent": 0.60, "rrt": 0.08},
    },
    3: {  # delta - multi-organ failure
        "name": "delta",
        "mortality_prob": 0.45,
        "shock_prob": 0.55,
        "icu_los_mean": 72,
        "vital_shift": {"heart_rate": 25, "sbp": -30, "map": -20, "temperature": 1.2},
        "lab_shift": {"lactate": 3.5, "creatinine": 2.0, "bilirubin": 2.0, "platelet": -100, "wbc": 8, "inr": 0.5},
        "treatment_prob": {"vasopressor": 0.70, "mechanical_vent": 0.55, "rrt": 0.30},
    },
}


# ============================================================
# Simulated Data Generator
# ============================================================

@timer
def generate_simulated_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Generate simulated ICU sepsis patient time-series data.

    Features:
    - Different subtypes have distinct baseline shifts and temporal trends
    - Vital signs recorded hourly (low missing), labs every 4-8h (high missing)
    - Treatment interventions as binary time series
    - AR(1) temporal dynamics (not independent random values)

    Returns
    -------
    time_series_3d : np.ndarray, shape (n_patients, n_timesteps, n_features)
    patient_info : pd.DataFrame
    """
    sim_cfg = config["data"]["simulated"]
    n_patients = sim_cfg["n_patients"]
    n_timesteps = sim_cfg["n_timesteps"]
    n_subtypes = sim_cfg["n_subtypes"]
    missing_rate = sim_cfg["missing_rate"]

    set_global_seed(sim_cfg["random_seed"])

    vitals = config["variables"]["vitals"]
    labs = config["variables"]["labs"]
    treatments = config["variables"]["treatments"]
    all_features = vitals + labs + treatments
    n_features = len(all_features)

    logger.info(f"Generating simulated data: {n_patients} patients, {n_timesteps} timesteps, "
                f"{n_features} features, {n_subtypes} subtypes")

    # Assign subtypes (uneven distribution, more realistic)
    subtype_probs = [0.35, 0.25, 0.25, 0.15]
    subtypes = np.random.choice(n_subtypes, size=n_patients, p=subtype_probs)

    time_series_3d = np.zeros((n_patients, n_timesteps, n_features))
    patient_records = []

    for i in range(n_patients):
        subtype = subtypes[i]
        profile = SUBTYPE_PROFILES[subtype]

        # --- Generate vital signs ---
        for j, var_name in enumerate(vitals):
            base_mean, base_std = VITAL_RANGES[var_name]
            shift = profile["vital_shift"].get(var_name, 0)

            series = _generate_ar1_series(
                n_timesteps, mean=base_mean + shift,
                std=base_std * 0.5, phi=0.85
            )
            if var_name in ("heart_rate", "resp_rate", "temperature"):
                trend = np.linspace(0, shift * 0.3, n_timesteps)
                series += trend
            elif var_name in ("sbp", "map", "spo2"):
                trend = np.linspace(0, shift * 0.2, n_timesteps)
                series += trend

            time_series_3d[i, :, j] = series

        # --- Generate lab values ---
        for j, var_name in enumerate(labs):
            col_idx = len(vitals) + j
            base_mean, base_std = LAB_RANGES[var_name]
            shift = profile["lab_shift"].get(var_name, 0)

            series = _generate_ar1_series(
                n_timesteps, mean=base_mean + shift,
                std=base_std * 0.3, phi=0.90
            )
            # Labs sampled every 4-8 hours; rest are NaN
            sample_interval = np.random.choice([4, 6, 8])
            mask = np.ones(n_timesteps, dtype=bool)
            sampled_points = np.arange(0, n_timesteps, sample_interval)
            mask[sampled_points] = False
            series[mask] = np.nan

            time_series_3d[i, :, col_idx] = series

        # --- Generate treatment interventions ---
        for j, var_name in enumerate(treatments):
            col_idx = len(vitals) + len(labs) + j
            prob = profile["treatment_prob"].get(var_name, 0.1)
            if np.random.random() < prob:
                start_hour = np.random.randint(0, max(1, n_timesteps // 2))
                time_series_3d[i, start_hour:, col_idx] = 1.0
            else:
                time_series_3d[i, :, col_idx] = 0.0

        # --- Apply random missing to vital signs ---
        vital_missing_mask = np.random.random((n_timesteps, len(vitals))) < (missing_rate * 0.3)
        for j in range(len(vitals)):
            time_series_3d[i, vital_missing_mask[:, j], j] = np.nan

        # --- Generate outcome variables ---
        mortality = int(np.random.random() < profile["mortality_prob"])
        shock = int(np.random.random() < profile["shock_prob"])
        icu_los = max(12, np.random.normal(profile["icu_los_mean"], 24))

        patient_records.append({
            "patient_id": f"SIM_{i:04d}",
            "subtype_true": subtype,
            "subtype_name": profile["name"],
            "mortality_28d": mortality,
            "shock_onset": shock,
            "icu_los": round(icu_los, 1),
            "age": int(np.clip(np.random.normal(65, 15), 18, 95)),
            "gender": np.random.choice(["M", "F"]),
        })

    patient_info = pd.DataFrame(patient_records)

    logger.info(f"Simulated data generated. Subtype distribution: {dict(zip(*np.unique(subtypes, return_counts=True)))}")
    logger.info(f"28-day mortality rate: {patient_info['mortality_28d'].mean():.1%}")

    return time_series_3d, patient_info


def _generate_ar1_series(
    n: int, mean: float, std: float, phi: float = 0.8
) -> np.ndarray:
    """
    Generate an AR(1) autoregressive time series.

    AR(1) model: x_t = phi * x_{t-1} + (1-phi) * mean + epsilon
    More realistic than independent random values for physiological signals.
    """
    series = np.zeros(n)
    series[0] = mean + np.random.normal(0, std)
    innovation_std = std * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + (1 - phi) * mean + np.random.normal(0, innovation_std)
    return series


# ============================================================
# MIMIC-IV Data Loader
# ============================================================

def load_mimic_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load MIMIC-IV sepsis cohort from CSV/Parquet exported by build_analysis_table.py.

    Reads from data/processed/:
      - patient_static.csv (or .parquet)
      - patient_timeseries.csv (or .parquet)

    Converts to the same format as simulated data:
      - time_series_3d: (n_patients, n_timesteps, n_features)
      - patient_info: DataFrame

    If files don't exist, prompts user to run build_analysis_table.py first.
    """
    mimic_cfg = config["data"].get("mimic", {})
    processed_dir = resolve_path(mimic_cfg.get("processed_dir") or config["paths"]["processed_data"])

    static_path = processed_dir / "patient_static.parquet"
    ts_path = processed_dir / "patient_timeseries.parquet"
    if not static_path.exists():
        static_path = processed_dir / "patient_static.csv"
        ts_path = processed_dir / "patient_timeseries.csv"
    if not static_path.exists():
        raise FileNotFoundError(
            "MIMIC-IV analysis tables not found. Please run first:\n"
            "  python build_analysis_table.py\n"
            f"Expected location: {processed_dir}/"
        )

    logger.info(f"Loading MIMIC-IV data: {static_path.name}")

    if static_path.suffix == ".parquet":
        patient_info = pd.read_parquet(static_path)
    else:
        patient_info = pd.read_csv(static_path)

    if ts_path.suffix == ".parquet":
        ts_df = pd.read_parquet(ts_path)
    else:
        ts_df = pd.read_csv(ts_path)

    meta_cols = {"stay_id", "subject_id", "hr", "grid_time"}
    feature_cols = [c for c in ts_df.columns if c not in meta_cols]

    stay_ids = sorted(ts_df["stay_id"].unique())
    n_patients = len(stay_ids)
    n_timesteps = ts_df["hr"].nunique()
    n_features = len(feature_cols)

    time_series_3d = np.full((n_patients, n_timesteps, n_features), np.nan)
    for i, sid in enumerate(stay_ids):
        patient_data = ts_df.loc[ts_df["stay_id"] == sid].sort_values("hr")
        vals = patient_data[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(np.nan).values
        time_series_3d[i, :vals.shape[0], :] = np.where(pd.isna(vals), np.nan, vals).astype(float)

    patient_info = patient_info.set_index("stay_id").loc[stay_ids].reset_index()

    rename_map = {
        "is_sepsis3": "subtype_true",
        "mortality_28d": "mortality_28d",
        "los_icu_days": "icu_los",
    }
    for old, new in rename_map.items():
        if old in patient_info.columns and old != new:
            patient_info[new] = patient_info[old]

    if "patient_id" not in patient_info.columns:
        patient_info["patient_id"] = patient_info["stay_id"].astype(str)
    if "shock_onset" not in patient_info.columns:
        patient_info["shock_onset"] = (patient_info.get("vasopressor_use", 0)).astype(int)
    if "icu_los" not in patient_info.columns:
        patient_info["icu_los"] = patient_info.get("los_icu_days", 48)

    logger.info(f"MIMIC-IV data loaded: {time_series_3d.shape}")
    logger.info(f"  Features: {feature_cols}")
    logger.info(f"  Sepsis-3 patients: {patient_info['is_sepsis3'].sum()} / {n_patients}")

    return time_series_3d, patient_info


# ============================================================
# eICU Data Loader (reserved for V2)
# ============================================================



# ============================================================
# Sepsis 2019 Data Loader
# ============================================================
def load_sepsis2019_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load PhysioNet/CinC Challenge 2019 Early Prediction of Sepsis data.

    Uses real ICU data from ~40,000 patients with hourly measurements
    of 37 clinical variables including Lactate, Bilirubin, Creatinine.
    
    Config paths:
      data.sepsis2019.data_dir: Path to directory with setA/ and setB/
      data.sepsis2019.n_hours: Number of hours per patient (default: 48)
    """
    s19_cfg = config["data"].get("sepsis2019", {})
    data_dir = s19_cfg.get("data_dir", "data/external")
    n_hours = s19_cfg.get("n_hours", config["data"]["simulated"]["n_timesteps"])
    
    data_path = resolve_path(data_dir) / "sepsis2019"
    ts_3d, patient_info = load_sepsis2019(data_path, n_hours=n_hours)
    
    return ts_3d, patient_info

def load_eicu_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Load eICU sepsis cohort for external validation."""
    eicu_cfg = config["data"].get("eicu", {})
    processed_dir_value = eicu_cfg.get("processed_dir")
    tag = eicu_cfg.get("tag", "eicu_demo")

    if processed_dir_value:
        processed_dir = resolve_path(processed_dir_value)
        ts_path = processed_dir / f"time_series_{tag}.npy"
        info_path = processed_dir / f"patient_info_{tag}.csv"
        if ts_path.exists() and info_path.exists():
            logger.info("Loading cached eICU data: %s", ts_path.name)
            time_series_3d = np.load(ts_path)
            patient_info = pd.read_csv(info_path)
            logger.info("eICU cached data loaded: %s", time_series_3d.shape)
            return time_series_3d, patient_info

    return load_eicu_from_config(config)


# ============================================================
# PhysioNet 2012 ICU Data Loader
# ============================================================

def load_physionet_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load PhysioNet 2012 Multi-parameter ICU Monitoring data.

    Uses real ICU data from 4 hospitals (set-a, set-b, set-c),
    converted to project standard format via load_physionet2012.py.
    
    Config paths:
      data.physionet.data_dir: Path to directory with set-a/, set-b/, set-c/
    """
    from load_physionet2012 import load_physionet2012, PROJECT_FEATURES
    
    pn_cfg = config["data"].get("physionet", {})
    data_dir = pn_cfg.get("data_dir", "data/external")
    n_hours = pn_cfg.get("n_hours", config["data"]["simulated"]["n_timesteps"])
    
    data_path = resolve_path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"PhysioNet 2012 data not found at {data_path}.\n"
            f"Please download from: https://physionet.org/content/challenge-2012/1.0.0/\n"
            f"Extract set-a.tar.gz, set-b.tar.gz, set-c.tar.gz to {data_dir}/"
        )
    
    ts_3d, patient_info = load_physionet2012(data_path, n_hours=n_hours)
    
    # Set mortality from outcomes if available
    from load_physionet2012 import load_outcomes
    outcomes = load_outcomes(data_path)
    if not outcomes.empty:
        patient_info["mortality_28d"] = patient_info["patient_id"].astype(int).map(
            dict(zip(outcomes["RecordID"], outcomes["In-hospital_death"]))
        ).fillna(0).astype(int)
    else:
        # Derive mortality proxy from GCS and vital signs
        gcs_idx = PROJECT_FEATURES.index("gcs")
        gcs_last = ts_3d[:, -1, gcs_idx]
        map_idx = PROJECT_FEATURES.index("map")
        map_last = ts_3d[:, -1, map_idx]
        patient_info["mortality_28d"] = (
            ((gcs_last < 5) | (np.isnan(gcs_last))) & 
            ((map_last < 50) | (np.isnan(map_last)))
        ).astype(int)
    
    # Set icu_los (48h window for all patients in this dataset)
    patient_info["icu_los"] = n_hours
    patient_info["shock_onset"] = 0  # Not directly available
    
    return ts_3d, patient_info


# ============================================================
# Unified Data Loading Entry Point
# ============================================================

def load_data(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load data based on configured source.

    Returns
    -------
    time_series_3d : np.ndarray, shape (n_patients, n_timesteps, n_features)
    patient_info : pd.DataFrame
    """
    source = config["data"]["source"]
    logger.info(f"Data source: {source}")

    if source == "simulated":
        return generate_simulated_data(config)
    elif source == "mimic":
        return load_mimic_data(config)
    elif source == "sepsis2019":
        return load_sepsis2019_data(config)
    elif source == "eicu":
        return load_eicu_data(config)
    elif source == "physionet2012":
        return load_physionet_data(config)
    else:
        raise ValueError(f"Unsupported data source: {source}")


# ============================================================
# Processed Data Cache
# ============================================================

def save_processed_data(
    time_series_3d: np.ndarray,
    patient_info: pd.DataFrame,
    config: dict,
    tag: str = "default",
) -> None:
    """Save processed data to the processed directory."""
    out_dir = resolve_path(config["paths"]["processed_data"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"time_series_{tag}.npy", time_series_3d)
    patient_info.to_csv(out_dir / f"patient_info_{tag}.csv", index=False)
    logger.info(f"Data saved to {out_dir}, tag: {tag}")


def load_processed_data(
    config: dict, tag: str = "default"
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached processed data."""
    data_dir = resolve_path(config["paths"]["processed_data"])
    ts_path = data_dir / f"time_series_{tag}.npy"
    info_path = data_dir / f"patient_info_{tag}.csv"

    if not ts_path.exists() or not info_path.exists():
        raise FileNotFoundError(f"Processed data not found (tag: {tag})")

    time_series_3d = np.load(ts_path)
    patient_info = pd.read_csv(info_path)
    logger.info(f"Loaded cached data: {time_series_3d.shape}")
    return time_series_3d, patient_info


def get_feature_names(config: dict) -> list[str]:
    """Return the feature name list aligned with the 3rd dimension of the 3D tensor."""
    source = config["data"]["source"]
    if source == "physionet2012":
        from load_physionet2012 import PROJECT_FEATURES
        return PROJECT_FEATURES
    if source == "sepsis2019":
        from load_sepsis2019 import PROJECT_FEATURES
        return PROJECT_FEATURES
    if source == "mimic":
        mimic_cfg = config["data"].get("mimic", {})
        processed_dir = resolve_path(mimic_cfg.get("processed_dir") or config["paths"]["processed_data"])
        for ext in (".parquet", ".csv"):
            ts_path = processed_dir / f"patient_timeseries{ext}"
            if ts_path.exists():
                if ext == ".parquet":
                    cols = pd.read_parquet(ts_path).columns.tolist()
                else:
                    cols = pd.read_csv(ts_path, nrows=0).columns.tolist()
                meta = {"stay_id", "subject_id", "hr", "grid_time"}
                return [c for c in cols if c not in meta]
        return _mimic_default_features()
    else:
        v = config["variables"]
        return v["vitals"] + v["labs"] + v["treatments"]


def _mimic_default_features() -> list[str]:
    """Default MIMIC feature list (matches build_analysis_table.py output)."""
    return [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
        "gcs", "gcs_motor", "gcs_verbal", "gcs_eyes",
        "creatinine", "bun", "sodium", "potassium", "bicarbonate", "glucose",
        "wbc", "platelet", "hemoglobin", "inr",
        "sofa_total", "sofa_resp", "sofa_coag", "sofa_liver",
        "sofa_cardio", "sofa_cns", "sofa_renal",
        "norepi_rate", "urine_output_24hr", "meanbp_min",
    ]
