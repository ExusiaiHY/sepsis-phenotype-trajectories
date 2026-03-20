"""
compat.py - Backward-compatibility adapter: S0 → V1 format.

Purpose:
  Convert S0 processed outputs back to V1's (N, T, F) format so that
  the existing feature_engineering.py and clustering.py can consume them
  without modification.

Two modes:
  exact_v1:    Produces the exact feature list V1 expects (24 features),
               including pao2_fio2_ratio (derived) and vasopressor/mechanical_vent
               (proxy indicators relabeled to match V1 names).
               NOTE: vasopressor and mechanical_vent are PROXY indicators,
               not true treatment records. This is documented but preserved
               for exact backward compatibility.

  extended_v1: Same as exact_v1 but appends proxy indicators with their
               correct schema names (vasopressor_proxy, mechvent_proxy)
               as additional channels. This avoids the relabeling ambiguity.

Connects to:
  - s0/schema.py for variable mappings
  - data/s0/processed/ for input tensors
  - src/feature_engineering.py consumes the output

How to run:
  from s0.compat import to_v1_format
  ts_3d, patient_info, feature_names = to_v1_format("data/s0", mode="exact_v1")

Expected output:
  Tuple of (np.ndarray, pd.DataFrame, list[str]) matching V1 interface.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from s0.schema import (
    CONTINUOUS_NAMES, CONTINUOUS_INDEX, N_CONTINUOUS,
    V1_FEATURE_ORDER, V1_EXTENDED_ADDITIONS,
)

logger = logging.getLogger("s0.compat")


def to_v1_format(
    s0_dir: str | Path,
    mode: str = "exact_v1",
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Load S0 processed data and convert to V1 format.

    Parameters
    ----------
    s0_dir : path to data/s0/
    mode : "exact_v1" or "extended_v1"

    Returns
    -------
    time_series_3d : (N, T, F) matching V1 feature order
    patient_info : DataFrame with V1-expected columns
    feature_names : list[str] matching dim 2 of time_series_3d
    """
    s0_dir = Path(s0_dir)
    proc_dir = s0_dir / "processed"

    continuous = np.load(proc_dir / "continuous.npy")    # (N, T, 21)
    proxy = np.load(proc_dir / "proxy_indicators.npy")   # (N, T, 2)
    static = pd.read_csv(s0_dir / "static.csv")

    n_patients, n_hours, _ = continuous.shape

    if mode == "exact_v1":
        return _build_exact_v1(continuous, proxy, static, n_patients, n_hours)
    elif mode == "extended_v1":
        return _build_extended_v1(continuous, proxy, static, n_patients, n_hours)
    else:
        raise ValueError(f"Unknown compat mode: {mode}. Use 'exact_v1' or 'extended_v1'.")


def _build_exact_v1(continuous, proxy, static, n_patients, n_hours):
    """
    Build exact V1 format: 24 features matching V1_FEATURE_ORDER.

    V1 expects: 21 continuous + pao2_fio2_ratio (derived) + vasopressor + mechanical_vent
    vasopressor and mechanical_vent are mapped from proxy indicators.
    """
    # V1 feature list has 24 entries
    n_v1_features = len(V1_FEATURE_ORDER)
    result = np.full((n_patients, n_hours, n_v1_features), np.nan, dtype=np.float32)

    # Map continuous features (first 21)
    for v1_idx, v1_name in enumerate(V1_FEATURE_ORDER):
        if v1_name in CONTINUOUS_INDEX:
            s0_idx = CONTINUOUS_INDEX[v1_name]
            result[:, :, v1_idx] = continuous[:, :, s0_idx]
        elif v1_name == "pao2_fio2_ratio":
            # Derived: pao2 / fio2
            pao2 = continuous[:, :, CONTINUOUS_INDEX["pao2"]]
            fio2 = continuous[:, :, CONTINUOUS_INDEX["fio2"]]
            valid = np.isfinite(pao2) & np.isfinite(fio2) & (fio2 > 0)
            ratio = np.where(valid, pao2 / fio2, np.nan)
            result[:, :, v1_idx] = ratio
        elif v1_name == "vasopressor":
            # PROXY indicator relabeled for V1 compatibility
            result[:, :, v1_idx] = proxy[:, :, 0]
        elif v1_name == "mechanical_vent":
            # PROXY indicator relabeled for V1 compatibility
            result[:, :, v1_idx] = proxy[:, :, 1]

    # NaN safety
    result = np.nan_to_num(result, nan=0.0)

    patient_info = _convert_static_to_v1(static)

    logger.info(f"exact_v1: {result.shape}, {len(V1_FEATURE_ORDER)} features")
    return result, patient_info, list(V1_FEATURE_ORDER)


def _build_extended_v1(continuous, proxy, static, n_patients, n_hours):
    """
    Build extended V1 format: exact_v1 features + proxy indicators with correct names.
    """
    ts_v1, patient_info, v1_names = _build_exact_v1(continuous, proxy, static, n_patients, n_hours)

    # Append proxy indicators with their correct schema names
    extended = np.concatenate([ts_v1, proxy], axis=2)
    extended_names = list(v1_names) + list(V1_EXTENDED_ADDITIONS)

    logger.info(f"extended_v1: {extended.shape}, {len(extended_names)} features")
    return extended, patient_info, extended_names


def _convert_static_to_v1(static: pd.DataFrame) -> pd.DataFrame:
    """Convert S0 static metadata to V1 patient_info format."""
    info = static.copy()

    # V1 expects these column names
    rename = {
        "mortality_inhospital": "mortality_28d",
    }
    for old, new in rename.items():
        if old in info.columns:
            info[new] = info[old]

    # V1 expects 'icu_los' (not icu_los_hours)
    if "icu_los_hours" in info.columns and "icu_los" not in info.columns:
        info["icu_los"] = info["icu_los_hours"]

    # V1 expects 'shock_onset'
    if "shock_onset" not in info.columns:
        info["shock_onset"] = 0

    # V1 expects 'data_source' (already present)

    return info
