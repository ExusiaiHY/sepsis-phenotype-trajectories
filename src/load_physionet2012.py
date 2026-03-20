"""
load_physionet2012.py - Load and convert PhysioNet 2012 ICU Challenge data

Converts the PhysioNet 2012 Multi-parameter Intelligent Monitoring 
for Intensive Care (MIMIC-II) dataset into the project's standard format:
  - time_series_3d: np.ndarray, shape (n_patients, n_timesteps, n_features)
  - patient_info: pd.DataFrame

The dataset contains 4000 records from set-a (training), set-b (test), 
and set-c (additional hospital), each with hourly/irregular measurements
of vital signs, labs, and demographics.

Cross-center setup:
  - set-a + set-b = "center A" (original MIMIC-II)
  - set-c = "center B" (separate hospital, for external validation)

References:
  Silva et al., Predicting in-hospital mortality of ICU patients, PhysioNet/CinC 2012.
  Available: https://physionet.org/content/challenge-2012/1.0.0/
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import setup_logger, timer, resolve_path

logger = setup_logger(__name__)


# ============================================================
# Parameter Mapping: PhysioNet -> Project
# ============================================================
# The PhysioNet 2012 dataset uses these variable names.
# We map them to the project's standard feature names.

# Vitals mapping (PhysioNet name -> project name)
VITAL_MAP = {
    "HR": "heart_rate",
    "SysABP": "sbp",
    "NISysABP": "sbp",       # Non-invasive SBP (fallback)
    "DiasABP": "dbp",
    "NIDiasABP": "dbp",      # Non-invasive DBP (fallback)
    "MAP": "map",
    "NIMAP": "map",          # Non-invasive MAP (fallback)
    "RespRate": "resp_rate",
    "SaO2": "spo2",
    "Temp": "temperature",
    "GCS": "gcs",
}

# Lab mapping
LAB_MAP = {
    "Creatinine": "creatinine",
    "BUN": "bun",
    "Glucose": "glucose",
    "Na": "sodium",
    "K": "potassium",
    "HCO3": "bicarbonate",
    "WBC": "wbc",
    "Platelets": "platelet",
    "HCT": "hematocrit",
    "Mg": "magnesium",
    "PaO2": "pao2",
    "FiO2": "fio2",
    "PaCO2": "paco2",
    "pH": "ph",
}

# All project feature names (aligned with data_loader.py config)
PROJECT_FEATURES = [
    # Vitals
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
    # Labs (from config)
    "lactate", "creatinine", "bilirubin", "platelet", "wbc", "pao2_fio2", "inr",
    # Treatments (binary)
    "vasopressor", "mechanical_vent", "rrt",
    # Additional features available in PhysioNet
    "gcs", "bun", "sodium", "potassium", "bicarbonate", "glucose",
    "hematocrit", "magnesium", "pao2", "fio2", "paco2", "ph",
]

# Static parameters (recorded once at 00:00)
STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType", "Weight"}
# Treatment / binary parameters
TREATMENT_PARAMS = {"MechVent"}
# All dynamic (time-varying) parameters we want to extract
ALL_RELEVANT_PARAMS = set(VITAL_MAP.keys()) | set(LAB_MAP.keys()) | TREATMENT_PARAMS


# ============================================================
# Single File Parser
# ============================================================

def parse_patient_file(filepath: str, n_hours: int = 48) -> Optional[dict]:
    """
    Parse a single PhysioNet 2012 patient file.

    Returns dict with:
      - record_id, age, gender, height, weight, icu_type
      - dynamic_data: dict of {parameter: [(hour, value), ...]}
      - outcome: In-hospital mortality (from outcome files, or None if unknown)
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None

    if df.empty:
        return None

    # Parse time to hours
    # Parse HH:MM to hours (handles missing/zero padding)
    def _parse_time(t):
        try:
            parts = str(t).strip().split(":")
            if len(parts) >= 2:
                return int(parts[0]) + int(parts[1]) / 60.0
            return 0.0
        except (ValueError, IndexError):
            return 0.0
    df["hour"] = df["Time"].apply(_parse_time)

    # Filter to within n_hours window
    df = df[df["hour"] <= n_hours].copy()

    static_info = {}
    dynamic_data = {p: [] for p in ALL_RELEVANT_PARAMS}

    for _, row in df.iterrows():
        param = row["Parameter"]
        value = row["Value"]
        hour = row["hour"]

        if pd.isna(value):
            continue

        # Convert -1 to NaN (missing indicator in PhysioNet 2012)
        if value == -1:
            continue

        if param in STATIC_PARAMS:
            if param == "Age":
                # Age > 300 means months for infants
                static_info["age"] = value if value < 300 else value / 12.0
            else:
                static_info[param.lower()] = value
        elif param in dynamic_data:
            try:
                dynamic_data[param].append((hour, float(value)))
            except (ValueError, TypeError):
                continue

    # Must have at least some vital sign data
    has_vitals = any(len(v) > 0 for v in [dynamic_data.get(p, []) for p in ["HR", "SysABP", "NISysABP", "MAP", "NIMAP"]])
    if not has_vitals:
        return None

    static_info["dynamic_data"] = dynamic_data
    return static_info


# ============================================================
# Resample to Hourly Grid
# ============================================================

def resample_to_grid(dynamic_data: dict, n_hours: int = 48) -> dict:
    """
    Resample irregular measurements to a fixed hourly grid.
    
    Strategy:
    - For each parameter, collect measurements and forward-fill to hourly grid
    - Use last observation carried forward (LOCF) within the window
    """
    grid = {}
    
    for param, measurements in dynamic_data.items():
        if not measurements:
            grid[param] = [np.nan] * n_hours
            continue
        
        # Sort by time
        measurements = sorted(measurements, key=lambda x: x[0])
        
        # Create hourly grid using forward fill
        hourly_vals = []
        meas_idx = 0
        
        for h in range(n_hours):
            # Find the most recent measurement at or before hour h
            best_val = np.nan
            while meas_idx < len(measurements) and measurements[meas_idx][0] <= h:
                best_val = measurements[meas_idx][1]
                meas_idx += 1
            
            hourly_vals.append(best_val)
        
        grid[param] = hourly_vals
    
    return grid


# ============================================================
# Convert to Project Format
# ============================================================

def patient_to_project_format(patient_data: dict, n_hours: int = 48) -> tuple[np.ndarray, dict]:
    """
    Convert parsed patient data to project's 3D tensor format.

    Returns:
      - feature_vector: np.ndarray, shape (n_hours, n_features)
      - info: dict with static patient info
    """
    grid = resample_to_grid(patient_data["dynamic_data"], n_hours)
    
    # Build feature vector aligned with PROJECT_FEATURES
    feature_vector = np.full((n_hours, len(PROJECT_FEATURES)), np.nan)
    
    for param, project_name in {**VITAL_MAP, **LAB_MAP}.items():
        if project_name in PROJECT_FEATURES and param in grid:
            idx = PROJECT_FEATURES.index(project_name)
            feature_vector[:, idx] = grid[param]
    
    # Handle MechVent -> mechanical_vent
    if "MechVent" in grid:
        idx = PROJECT_FEATURES.index("mechanical_vent")
        mv_vals = np.array(grid["MechVent"])
        # MechVent is 0 or 1 (or intermittent), convert to binary
        feature_vector[:, idx] = np.where(pd.isna(mv_vals), 0, np.where(mv_vals > 0, 1.0, 0.0))
    
    # Derive vasopressor use (no direct variable, use MAP < 65 as proxy)
    map_idx = PROJECT_FEATURES.index("map")
    vasopressor_idx = PROJECT_FEATURES.index("vasopressor")
    map_vals = feature_vector[:, map_idx]
    feature_vector[:, vasopressor_idx] = np.where(
        (~np.isnan(map_vals)) & (map_vals < 65), 1.0, 0.0
    )
    
    # Derive PaO2/FiO2 ratio
    pao2_idx = PROJECT_FEATURES.index("pao2")
    fio2_idx = PROJECT_FEATURES.index("fio2")
    pf_idx = PROJECT_FEATURES.index("pao2_fio2")
    pao2_vals = feature_vector[:, pao2_idx]
    fio2_vals = feature_vector[:, fio2_idx]
    # FiO2 in PhysioNet is fraction (0-1), convert to percentage
    # Determine if FiO2 is fraction (0-1) or percentage (0-100)
    valid_fio2 = fio2_vals[~np.isnan(fio2_vals)]
    if len(valid_fio2) > 0 and np.nanmax(valid_fio2) <= 1.5:
        fio2_frac = fio2_vals * 100
    else:
        fio2_frac = fio2_vals
    # PaO2/FiO2 = PaO2 / (FiO2_fraction) where FiO2_fraction = FiO2_pct / 100
    fio2_frac_safe = np.where(fio2_frac > 0, fio2_frac, np.nan)
    pao2_over_fio2 = np.where(
        (~np.isnan(pao2_vals)) & (~np.isnan(fio2_frac_safe)),
        pao2_vals / (fio2_frac_safe / 100),
        np.nan
    )
    feature_vector[:, pf_idx] = pao2_over_fio2
    
    # Build patient info
    info = {
        "patient_id": str(int(patient_data.get("recordid", 0))),
        "age": patient_data.get("age", np.nan),
        "gender": "M" if patient_data.get("gender", 0) == 0 else "F",
        "icu_type": int(patient_data.get("icutype", 0)),
        "height": patient_data.get("height", np.nan),
        "weight": patient_data.get("weight", np.nan),
    }
    
    return feature_vector, info


# ============================================================
# Main Loader
# ============================================================

def _get_cache_path(data_dir: Path, sets: list[str], n_hours: int) -> Path:
    """Return cache file path based on parameters."""
    sets_tag = "_".join(sorted(sets))
    return data_dir / f".cache_{sets_tag}_{n_hours}h.npz"


def _save_cache(path: Path, time_series: np.ndarray, patient_info: pd.DataFrame) -> None:
    """Save parsed data to cache using only numpy arrays and CSV (no pickle)."""
    np.save(str(path) + ".ts.npy", time_series)
    patient_info.to_csv(str(path) + ".info.csv", index=False)


def _load_cache(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load cached data."""
    ts = np.load(str(path) + ".ts.npy")
    info = pd.read_csv(str(path) + ".info.csv")
    return ts, info


def _cache_exists(path: Path) -> bool:
    """Check if cache files exist."""
    return (Path(str(path) + ".ts.npy").exists() and
            Path(str(path) + ".info.csv").exists())


def _parse_single_file(args: tuple) -> Optional[tuple]:
    """Worker function for parallel parsing. Returns (feature_vector, info) or None."""
    filepath, n_hours = args
    patient = parse_patient_file(str(filepath), n_hours)
    if patient is None:
        return None
    feature_vector, info = patient_to_project_format(patient, n_hours)
    if "set-c" in str(filepath):
        info["data_source"] = "center_b"
    else:
        info["data_source"] = "center_a"
    return (feature_vector, info)


@timer
def load_physionet2012(
    data_dir: str | Path,
    n_hours: int = 48,
    max_patients: int | None = None,
    sets: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load PhysioNet 2012 ICU data and convert to project format.

    Optimizations:
    - NPY/CSV cache: After first parse, saves tensor + info to disk.
      Subsequent loads take <1 second instead of 100+ seconds.
    - Parallel parsing: Uses multiprocessing for first-time file parsing.

    Parameters
    ----------
    data_dir : Path
        Directory containing set-a/, set-b/, set-c/ subdirectories.
    n_hours : int
        Time window length in hours.
    max_patients : int | None
        Maximum number of patients to load (None = all).
    sets : list[str] | None
        Which sets to load, e.g. ["set-a", "set-c"].
        Default: ["set-a", "set-b", "set-c"].

    Returns
    -------
    time_series_3d : np.ndarray, shape (n_patients, n_timesteps, n_features)
    patient_info : pd.DataFrame
    """
    data_dir = Path(data_dir)
    if sets is None:
        sets = ["set-a", "set-b", "set-c"]

    # --- Check cache first ---
    cache_path = _get_cache_path(data_dir, sets, n_hours)
    if _cache_exists(cache_path) and max_patients is None:
        logger.info(f"Loading from cache: {cache_path.name}")
        time_series_3d, patient_info = _load_cache(cache_path)
        logger.info(f"Cache loaded: {time_series_3d.shape}, "
                     f"Center A: {(patient_info['data_source'] == 'center_a').sum()}, "
                     f"Center B: {(patient_info['data_source'] == 'center_b').sum()}")
        return time_series_3d, patient_info

    # --- Collect all patient files ---
    all_files = []
    for s in sets:
        sdir = data_dir / s
        if not sdir.exists():
            logger.warning(f"Directory not found: {sdir}")
            continue
        files = sorted(list(sdir.glob("*.txt")))
        all_files.extend(files)
        logger.info(f"  {s}: {len(files)} patient files")

    if max_patients:
        all_files = all_files[:max_patients]

    logger.info(f"Parsing {len(all_files)} patients from PhysioNet 2012 (parallel)...")

    # --- Parallel parsing ---
    import multiprocessing as mp
    n_workers = min(mp.cpu_count(), 8)
    args_list = [(f, n_hours) for f in all_files]

    patient_vectors = []
    patient_records = []
    skipped = 0

    with mp.Pool(n_workers) as pool:
        results = pool.map(_parse_single_file, args_list, chunksize=200)

    for r in results:
        if r is None:
            skipped += 1
        else:
            patient_vectors.append(r[0])
            patient_records.append(r[1])

    if not patient_vectors:
        raise ValueError("No valid patient data found!")

    # Stack into 3D tensor
    time_series_3d = np.stack(patient_vectors, axis=0)
    patient_info = pd.DataFrame(patient_records)

    logger.info(f"PhysioNet 2012 loaded: {time_series_3d.shape}")
    logger.info(f"  Features ({len(PROJECT_FEATURES)}): {PROJECT_FEATURES}")
    logger.info(f"  Skipped: {skipped} patients")
    logger.info(f"  Center A: {(patient_info['data_source'] == 'center_a').sum()} patients")
    logger.info(f"  Center B: {(patient_info['data_source'] == 'center_b').sum()} patients")

    # --- Save cache for next time ---
    if max_patients is None:
        _save_cache(cache_path, time_series_3d, patient_info)
        cache_size = Path(str(cache_path) + ".ts.npy").stat().st_size / 1024 / 1024
        logger.info(f"Cache saved: {cache_path.name}.* ({cache_size:.1f} MB)")

    return time_series_3d, patient_info


def load_single_center(
    data_dir: str | Path,
    center: str = "center_a",
    n_hours: int = 48,
    max_patients: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load a single center for cross-center validation."""
    if center == "center_a":
        sets = ["set-a", "set-b"]
    elif center == "center_b":
        sets = ["set-c"]
    else:
        raise ValueError(f"Unknown center: {center}")
    return load_physionet2012(data_dir, n_hours, max_patients, sets)


# ============================================================
# Outcome Labels (for evaluation)
# ============================================================

def load_outcomes(data_dir: str | Path) -> pd.DataFrame:
    """
    Load outcome files (In-hospital mortality).

    Returns DataFrame with RecordID and In-hospital_death columns.
    """
    data_dir = Path(data_dir)
    outcomes = []
    for s in ["set-a", "set-b", "set-c"]:
        outcome_file = data_dir / f"Outcomes-{s}.txt"
        if not outcome_file.exists():
            continue
        df = pd.read_csv(outcome_file)
        outcomes.append(df)
    
    if outcomes:
        return pd.concat(outcomes, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-hours", type=int, default=48)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--sets", nargs="+", default=["set-a", "set-b", "set-c"])
    args = parser.parse_args()

    ts, info = load_physionet2012(args.data_dir, args.n_hours, args.max_patients, args.sets)
    print(f"Shape: {ts.shape}")
    print(f"Missing rate: {np.isnan(ts).mean():.1%}")
    print(f"Patient info:\n{info.head()}")