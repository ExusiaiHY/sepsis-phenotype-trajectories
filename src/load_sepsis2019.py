"""
load_sepsis2019.py - Load and convert PhysioNet/CinC Challenge 2019 data

Converts the Sepsis Early Prediction dataset (PSV files) into the project's
standard format:
  - time_series_3d: np.ndarray, shape (n_patients, n_timesteps, n_features)
  - patient_info: pd.DataFrame

The 2019 dataset contains 40,336 ICU patients with:
  - 40 columns per record (37 clinical variables + demographics/outcomes)
  - Hourly measurements over variable-length ICU stays
  - SepsisLabel: binary sepsis onset annotation per hour
  - ICULOS: actual ICU length of stay (hours)

This dataset is SIGNIFICANTLY richer than PhysioNet 2012:
  - Has Lactate, Bilirubin, Glucose, Creatinine (missing in 2012)
  - Has actual ICU LOS and Sepsis labels
  - Has TroponinI, Fibrinogen, BaseExcess

References:
  Reyna et al., Early Prediction of Sepsis from Clinical Data, Crit Care Med 48(2), 2020.
  Available: https://physionet.org/content/challenge-2019/1.0.0/
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
# Column definitions
# ============================================================

# Raw CSV column names from PSV files
RAW_COLUMNS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
    'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel',
]

# Static columns (not time-varying clinical measurements)
STATIC_COLS = {'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel'}

# Clinical variable columns
CLINICAL_COLS = [c for c in RAW_COLUMNS if c not in STATIC_COLS]

# Mapping to project-standard feature names
# Sepsis 2019 has all the key variables that PhysioNet 2012 was missing!
FEATURE_MAP = {
    'HR': 'heart_rate',
    'O2Sat': 'spo2',
    'Temp': 'temperature',
    'SBP': 'sbp',
    'MAP': 'map',
    'DBP': 'dbp',
    'Resp': 'resp_rate',
    'BaseExcess': 'base_excess',
    'HCO3': 'bicarbonate',
    'FiO2': 'fio2',
    'pH': 'ph',
    'PaCO2': 'paco2',
    'SaO2': 'spo2_invasive',  # invasive SaO2 (keep separate from O2Sat)
    'AST': 'ast',
    'BUN': 'bun',
    'Alkalinephos': 'alkaline_phosphatase',
    'Calcium': 'calcium',
    'Chloride': 'chloride',
    'Creatinine': 'creatinine',
    'Bilirubin_direct': 'bilirubin_direct',
    'Glucose': 'glucose',
    'Lactate': 'lactate',  # KEY: available in 2019!
    'Magnesium': 'magnesium',
    'Phosphate': 'phosphate',
    'Potassium': 'potassium',
    'Bilirubin_total': 'bilirubin',
    'TroponinI': 'troponin_i',
    'Hct': 'hematocrit',
    'Hgb': 'hemoglobin',
    'PTT': 'ptt',
    'WBC': 'wbc',
    'Fibrinogen': 'fibrinogen',
    'Platelets': 'platelet',
}

# Treatment indicators (derived from clinical variables)
TREATMENT_MAP = {
    'vasopressor': 0,   # Will be derived from MAP < 65
    'mechanical_vent': 0,  # Not directly available
    'rrt': 0,           # Not directly available
}

# All project feature names (clinical + derived + treatment)
PROJECT_FEATURES = list(FEATURE_MAP.values()) + list(TREATMENT_MAP.keys())


def _parse_single_patient(args: tuple) -> dict | None:
    """Parse a single PSV file into a dict with time series and metadata."""
    filepath, n_hours = args

    try:
        df = pd.read_csv(filepath, sep='|', dtype=str)
        if df.empty or len(df) < 2:
            return None

        # Cap to n_hours
        if len(df) > n_hours:
            df = df.iloc[:n_hours]

        n_timesteps = len(df)

        # Extract clinical time series
        ts_data = np.full((n_timesteps, len(PROJECT_FEATURES)), np.nan)

        for i, raw_name in enumerate(CLINICAL_COLS):
            if raw_name in FEATURE_MAP:
                col_idx = PROJECT_FEATURES.index(FEATURE_MAP[raw_name])
                vals = pd.to_numeric(df[raw_name], errors='coerce').values
                ts_data[:len(vals), col_idx] = vals

        # Derive vasopressor use (MAP < 65 as proxy)
        if 'MAP' in df.columns:
            map_vals = pd.to_numeric(df['MAP'], errors='coerce').values
            vs_idx = PROJECT_FEATURES.index('vasopressor')
            ts_data[:len(map_vals), vs_idx] = (map_vals < 65).astype(float)

        # Extract static info
        record = {
            'time_series': ts_data,
            'n_timesteps': n_timesteps,
        }

        if 'Age' in df.columns:
            record['age'] = float(df['Age'].iloc[0])
        if 'Gender' in df.columns:
            record['gender'] = int(df['Gender'].iloc[0])
        if 'ICULOS' in df.columns:
            record['icu_los'] = float(df['ICULOS'].iloc[-1])
        if 'SepsisLabel' in df.columns:
            record['sepsis_label'] = int(df['SepsisLabel'].max())

        return record

    except Exception as e:
        return None


@timer
def load_sepsis2019(
    data_dir: str | Path,
    n_hours: int = 48,
    max_patients: int | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load Sepsis 2019 dataset from PSV files.

    Parameters
    ----------
    data_dir : str | Path
        Path to directory containing setA/ and setB/ subdirectories.
    n_hours : int
        Maximum number of hours to include per patient.
    max_patients : int | None
        Limit number of patients (for testing).
    use_cache : bool
        Use .npz cache if available.

    Returns
    -------
    time_series_3d : np.ndarray, shape (n_patients, n_hours, n_features)
    patient_info : pd.DataFrame
    """
    data_path = resolve_path(data_dir)

    # Check for cache
    cache_file = data_path / '.cache_sepsis2019.npz'
    cache_meta = data_path / '.cache_sepsis2019_meta.csv'

    if use_cache and cache_file.exists() and cache_meta.exists():
        logger.info(f"Loading from cache: {cache_file.name}")
        data = np.load(cache_file)
        meta = pd.read_csv(cache_meta)
        logger.info(f"Cache loaded: {data['time_series'].shape}")
        return data['time_series'], meta

    # Collect all PSV files
    all_files = []
    for set_name in ['setA', 'setB']:
        set_dir = data_path / set_name
        if set_dir.exists():
            for f in sorted(set_dir.glob('*.psv')):
                all_files.append((str(f), n_hours))

    if not all_files:
        raise FileNotFoundError(
            f"No PSV files found in {data_path}/setA/ or {data_path}/setB/.\n"
            f"Please download from: https://physionet.org/content/challenge-2019/1.0.0/\n"
            f"Extract to {data_path}/setA/ and {data_path}/setB/"
        )

    if max_patients:
        all_files = all_files[:max_patients]

    logger.info(f"Loading {len(all_files)} patients from Sepsis 2019 (parallel)...")

    # Parallel parsing
    records = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        for result in pool.map(_parse_single_patient, all_files):
            if result is not None:
                records.append(result)
            else:
                skipped += 1

    if not records:
        raise ValueError("No valid patient records could be parsed.")

    # Pad to uniform length
    ts_list = []
    patient_rows = []

    for rec in records:
        n_t = rec['n_timesteps']
        ts = rec['time_series']
        if n_t < n_hours:
            # Pad with NaN
            padded = np.full((n_hours, ts.shape[1]), np.nan)
            padded[:n_t, :] = ts
            ts_list.append(padded)
        else:
            ts_list.append(ts)

        patient_rows.append({
            'patient_id': f"SEPSIS2019_{len(ts_list)}",
            'age': rec.get('age', np.nan),
            'gender': rec.get('gender', 0),
            'icu_los': rec.get('icu_los', n_hours),
            'mortality_28d': 0,  # Not directly available in 2019 dataset
            'sepsis_label': rec.get('sepsis_label', 0),
            'subtype_true': 0,  # No ground truth subtypes
            'actual_timesteps': n_t,
        })

    time_series_3d = np.stack(ts_list)
    patient_info = pd.DataFrame(patient_rows)

    # Derive mortality proxy from last vital signs
    hr_idx = PROJECT_FEATURES.index('heart_rate')
    map_idx = PROJECT_FEATURES.index('map')
    hr_last = time_series_3d[:, -1, hr_idx]
    map_last = time_series_3d[:, -1, map_idx]
    # Patients with very low MAP or missing critical vitals at end may have died
    patient_info['mortality_28d'] = (
        ((np.isnan(map_last) | (map_last < 50)) &
         (np.isnan(hr_last) | (hr_last > 120)))
    ).astype(int)

    logger.info(f"Sepsis 2019 loaded: {time_series_3d.shape}")
    logger.info(f"  Features ({len(PROJECT_FEATURES)}): {PROJECT_FEATURES}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Set A+B patients: {len(patient_info)}")
    logger.info(f"  Sepsis positive: {patient_info['sepsis_label'].sum()} ({patient_info['sepsis_label'].mean():.1%})")
    logger.info(f"  Mean ICU LOS: {patient_info['icu_los'].mean():.1f}h")

    # Cache
    if use_cache:
        np.savez_compressed(cache_file, time_series=time_series_3d)
        patient_info.to_csv(cache_meta, index=False)
        cache_size = cache_file.stat().st_size / 1024 / 1024
        logger.info(f"Cache saved: {cache_file.name} ({cache_size:.1f} MB)")

    return time_series_3d, patient_info
