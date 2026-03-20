"""
physionet2012_extractor.py - Extract PhysioNet 2012 data into S0 unified schema.

Purpose:
  Read raw PhysioNet 2012 patient .txt files and produce raw_aligned tensors:
  - continuous.npy (N, T, 21) + masks
  - interventions.npy (N, T, 2) all NaN (unavailable in this dataset) + masks
  - proxy_indicators.npy (N, T, 2) derived from MAP/FiO2 + masks
  - static.csv with patient metadata

  This module performs ONLY extraction and hourly alignment.
  NO imputation. NO normalization. Those happen in preprocessor.py.

Connects to:
  - schema.py for variable definitions and mappings
  - preprocessor.py consumes the raw_aligned outputs
  - scripts/s0_prepare.py calls extract_physionet2012()

How to run:
  Called by scripts/s0_prepare.py. Not intended as standalone.
  For testing: python3.14 -c "from s0.physionet2012_extractor import extract_physionet2012; ..."

Expected output artifacts:
  data/s0/raw_aligned/continuous.npy
  data/s0/raw_aligned/interventions.npy
  data/s0/raw_aligned/proxy_indicators.npy
  data/s0/raw_aligned/masks_continuous.npy
  data/s0/raw_aligned/masks_interventions.npy
  data/s0/raw_aligned/masks_proxy.npy
  data/s0/static.csv
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from s0.schema import (
    CONTINUOUS_SCHEMA, N_CONTINUOUS, CONTINUOUS_INDEX,
    INTERVENTION_SCHEMA, N_INTERVENTIONS,
    PROXY_SCHEMA, N_PROXY,
    PHYSIONET_ALL_MAP, PHYSIONET_DEMO_FIELDS,
    STATIC_FIELDS,
    ANCHOR_ICU_ADMISSION, MORTALITY_OUTCOMES_FILE, MORTALITY_PROXY_GCS_MAP, MORTALITY_UNAVAILABLE,
)

logger = logging.getLogger("s0.extractor")


def extract_physionet2012(
    data_dir: Path,
    output_dir: Path,
    n_hours: int = 48,
    sets: list[str] = ("set-a", "set-b", "set-c"),
    center_a_sets: set[str] = frozenset({"set-a", "set-b"}),
    center_b_sets: set[str] = frozenset({"set-c"}),
    min_measurements: int = 1,
) -> dict:
    """
    Extract PhysioNet 2012 data into S0 raw_aligned format.

    Returns dict with cohort statistics.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_aligned"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Collect all patient file paths ---
    file_list = []
    for set_name in sets:
        set_dir = data_dir / set_name
        if not set_dir.exists():
            logger.warning(f"Set directory not found: {set_dir}")
            continue
        for fpath in sorted(set_dir.iterdir()):
            if fpath.suffix == ".txt":
                file_list.append((fpath, set_name))

    logger.info(f"Found {len(file_list)} patient files across {len(sets)} sets")

    # --- Phase 2: Try loading outcome files ---
    outcomes = _load_outcomes(data_dir, sets)
    has_outcomes = len(outcomes) > 0
    logger.info(f"Outcome labels: {'loaded' if has_outcomes else 'not found (will use proxy)'} "
                f"({len(outcomes)} records)")

    # --- Phase 3: First pass — parse all files, collect static info ---
    static_records = []
    patient_data = []  # list of (patient_idx, hour, continuous_idx, value)
    skipped = 0

    for fpath, set_name in file_list:
        result = _parse_patient_file(fpath, set_name, center_a_sets, center_b_sets, n_hours)
        if result is None:
            skipped += 1
            continue

        static_row, measurements = result
        n_meas = len(measurements)

        if n_meas < min_measurements:
            skipped += 1
            continue

        pat_idx = len(static_records)

        # Resolve mortality
        pid = int(static_row["patient_id"])
        if has_outcomes and pid in outcomes:
            static_row["mortality_inhospital"] = outcomes[pid]
            static_row["mortality_source"] = MORTALITY_OUTCOMES_FILE
        else:
            static_row["mortality_inhospital"] = np.nan
            static_row["mortality_source"] = MORTALITY_UNAVAILABLE

        static_records.append(static_row)
        patient_data.append((pat_idx, measurements))

        if len(static_records) % 3000 == 0:
            logger.info(f"  Parsed {len(static_records)} patients...")

    n_patients = len(static_records)
    logger.info(f"Parsed {n_patients} patients, {skipped} skipped")

    # --- Phase 4: Build tensors ---
    continuous = np.full((n_patients, n_hours, N_CONTINUOUS), np.nan, dtype=np.float32)
    masks_cont = np.zeros((n_patients, n_hours, N_CONTINUOUS), dtype=np.float32)
    interventions = np.full((n_patients, n_hours, N_INTERVENTIONS), np.nan, dtype=np.float32)
    masks_int = np.zeros((n_patients, n_hours, N_INTERVENTIONS), dtype=np.float32)
    proxy = np.zeros((n_patients, n_hours, N_PROXY), dtype=np.float32)
    masks_proxy = np.zeros((n_patients, n_hours, N_PROXY), dtype=np.float32)

    for pat_idx, measurements in patient_data:
        for hour, var_name, value in measurements:
            if hour < 0 or hour >= n_hours:
                continue
            if var_name in CONTINUOUS_INDEX:
                fidx = CONTINUOUS_INDEX[var_name]
                continuous[pat_idx, hour, fidx] = value
                masks_cont[pat_idx, hour, fidx] = 1.0

    # --- Phase 5: Compute proxy indicators from continuous values ---
    map_idx = CONTINUOUS_INDEX["map"]
    fio2_idx = CONTINUOUS_INDEX["fio2"]

    for pat_idx in range(n_patients):
        for t in range(n_hours):
            # Vasopressor proxy: MAP < 65
            map_val = continuous[pat_idx, t, map_idx]
            if np.isfinite(map_val):
                proxy[pat_idx, t, 0] = 1.0 if map_val < 65 else 0.0
                masks_proxy[pat_idx, t, 0] = 1.0

            # MechVent proxy: FiO2 > 0.21
            fio2_val = continuous[pat_idx, t, fio2_idx]
            if np.isfinite(fio2_val):
                proxy[pat_idx, t, 1] = 1.0 if fio2_val > 0.21 else 0.0
                masks_proxy[pat_idx, t, 1] = 1.0

    # Intervention tensor: all NaN / mask=0 for PhysioNet 2012
    # (structural placeholder; masks_int stays all-zero)

    # --- Phase 6: Apply proxy mortality if no outcomes file ---
    if not has_outcomes:
        logger.info("No outcome files found. Computing mortality proxy from GCS + MAP.")
        for pat_idx in range(n_patients):
            gcs_idx = CONTINUOUS_INDEX["gcs"]
            gcs_vals = continuous[pat_idx, :, gcs_idx]
            map_vals = continuous[pat_idx, :, map_idx]

            valid_gcs = gcs_vals[np.isfinite(gcs_vals)]
            valid_map = map_vals[np.isfinite(map_vals)]

            has_low_gcs = len(valid_gcs) > 0 and np.min(valid_gcs) <= 5
            has_low_map = len(valid_map) >= 3 and np.sum(valid_map < 55) >= 3

            if has_low_gcs or has_low_map:
                static_records[pat_idx]["mortality_inhospital"] = 1
            else:
                static_records[pat_idx]["mortality_inhospital"] = 0
            static_records[pat_idx]["mortality_source"] = MORTALITY_PROXY_GCS_MAP

    # --- Phase 7: Save ---
    np.save(raw_dir / "continuous.npy", continuous)
    np.save(raw_dir / "masks_continuous.npy", masks_cont)
    np.save(raw_dir / "interventions.npy", interventions)
    np.save(raw_dir / "masks_interventions.npy", masks_int)
    np.save(raw_dir / "proxy_indicators.npy", proxy)
    np.save(raw_dir / "masks_proxy.npy", masks_proxy)

    static_df = pd.DataFrame(static_records)
    # Ensure all static fields exist
    for field in STATIC_FIELDS:
        if field not in static_df.columns:
            static_df[field] = np.nan
    static_df = static_df[STATIC_FIELDS]
    static_df.to_csv(output_dir / "static.csv", index=False)

    logger.info(f"Raw aligned data saved to {raw_dir}")
    logger.info(f"  continuous: {continuous.shape}")
    logger.info(f"  interventions: {interventions.shape}")
    logger.info(f"  proxy: {proxy.shape}")
    logger.info(f"  static: {static_df.shape}")

    # Compute stats
    overall_missing = 1.0 - np.mean(masks_cont)
    stats = {
        "n_patients": n_patients,
        "n_hours": n_hours,
        "n_continuous": N_CONTINUOUS,
        "n_interventions": N_INTERVENTIONS,
        "n_proxy": N_PROXY,
        "overall_continuous_missing_rate": float(overall_missing),
        "skipped_patients": skipped,
        "has_outcome_labels": has_outcomes,
    }
    return stats


# ============================================================
# Internal Helpers
# ============================================================

def _parse_patient_file(
    fpath: Path,
    set_name: str,
    center_a: set,
    center_b: set,
    n_hours: int,
) -> Optional[tuple[dict, list]]:
    """Parse one PhysioNet 2012 .txt file. Returns (static_dict, measurements_list) or None."""
    try:
        df = pd.read_csv(fpath, header=0)
    except Exception:
        return None

    if len(df) < 2:
        return None

    # Extract demographics from 00:00 rows
    record_id = None
    age = np.nan
    sex = np.nan
    height = np.nan
    weight = np.nan
    icu_type = np.nan

    measurements = []  # (hour, variable_name, value)
    max_hour = 0.0

    for _, row in df.iterrows():
        time_str = str(row.get("Time", ""))
        param = str(row.get("Parameter", ""))
        raw_val = row.get("Value", np.nan)

        # Parse hour
        hour = _parse_time_to_hour(time_str)
        if hour is None:
            continue

        if hour == 0 and param in PHYSIONET_DEMO_FIELDS:
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                continue

            if param == "RecordID":
                record_id = int(val)
            elif param == "Age":
                age = val
            elif param == "Gender":
                sex = int(val)
            elif param == "Height":
                height = val if val > 0 else np.nan
            elif param == "Weight":
                weight = val if val > 0 else np.nan
            elif param == "ICUType":
                icu_type = int(val)
            continue

        # Map to standard name
        std_name = PHYSIONET_ALL_MAP.get(param)
        if std_name is None:
            continue

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            continue

        # Basic validity check (negative values are suspect except temperature)
        if val < 0 and param != "Temp":
            continue

        int_hour = int(hour)
        if 0 <= int_hour < n_hours:
            measurements.append((int_hour, std_name, val))

        max_hour = max(max_hour, hour)

    if record_id is None:
        return None

    center_id = "center_a" if set_name in center_a else ("center_b" if set_name in center_b else "unknown")

    static = {
        "patient_id": str(record_id),
        "age": age,
        "sex": sex,
        "height_cm": height,
        "weight_kg": weight,
        "icu_type": icu_type,
        "icu_los_hours": max_hour,
        "center_id": center_id,
        "set_name": set_name,
        "data_source": "physionet2012",
        "sepsis_onset_hour": np.nan,  # Not available
        "anchor_time_type": ANCHOR_ICU_ADMISSION,
    }

    return static, measurements


def _parse_time_to_hour(time_str: str) -> Optional[float]:
    """Parse 'HH:MM' to fractional hours."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) >= 2:
            return int(parts[0]) + int(parts[1]) / 60.0
        return None
    except (ValueError, IndexError):
        return None


def _load_outcomes(data_dir: Path, sets: list[str]) -> dict[int, int]:
    """Load outcome files (Outcomes-a.txt, etc.) if they exist. Returns {RecordID: mortality}."""
    outcomes = {}
    suffix_map = {"set-a": "a", "set-b": "b", "set-c": "c"}

    for set_name in sets:
        suffix = suffix_map.get(set_name, "")
        if not suffix:
            continue
        outcome_path = data_dir / f"Outcomes-{suffix}.txt"
        if not outcome_path.exists():
            continue

        try:
            odf = pd.read_csv(outcome_path, header=0)
            for _, row in odf.iterrows():
                rid = int(row.get("RecordID", 0))
                mort = int(row.get("In-hospital_death", 0))
                if rid > 0:
                    outcomes[rid] = mort
            logger.info(f"  Loaded outcomes: {outcome_path.name} ({len(odf)} records)")
        except Exception as e:
            logger.warning(f"  Failed to load {outcome_path}: {e}")

    return outcomes
