"""
feature_engineering.py - Time-series feature extraction module

Responsibilities:
1. Extract patient-level statistical features from 3D time-series tensors (MVP core)
2. Support multi-window feature extraction (e.g., first 12h, 24h, full 48h)
3. Compute clinically derived indicators (shock index, lactate clearance, etc.)
4. Output feature matrix X: (n_patients, n_extracted_features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from utils import setup_logger, timer

logger = setup_logger(__name__)


# ============================================================
# Main Feature Extraction
# ============================================================

@timer
def extract_features(
    time_series_3d: np.ndarray,
    config: dict,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract patient-level statistical features from 3D time-series tensor.

    Parameters
    ----------
    time_series_3d : np.ndarray, shape (n_patients, n_timesteps, n_features)
    config : dict
    feature_names : list[str]  Original feature names.

    Returns
    -------
    feature_df : pd.DataFrame, shape (n_patients, n_extracted_features)
    """
    feat_cfg = config["features"]["statistical"]
    window_functions = feat_cfg["window_functions"]
    sub_windows = feat_cfg["sub_windows"]

    n_patients, n_timesteps, n_raw_features = time_series_3d.shape
    logger.info(f"Feature extraction: {n_patients} patients, functions {window_functions}, "
                f"sub-windows {sub_windows}")

    all_features = {}

    for win_hours in sub_windows:
        win_end = n_timesteps
        win_start = max(0, n_timesteps - win_hours)
        win_label = f"w{win_hours}h"

        window_data = time_series_3d[:, win_start:win_end, :]

        for j, var_name in enumerate(feature_names):
            series_batch = window_data[:, :, j]

            for func_name in window_functions:
                col_name = f"{var_name}_{func_name}_{win_label}"
                all_features[col_name] = _compute_stat(series_batch, func_name)

    # Additional: clinically derived indicators
    derived = _compute_derived_features(time_series_3d, feature_names)
    all_features.update(derived)

    feature_df = pd.DataFrame(all_features)

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    nan_count = feature_df.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Features contain {nan_count} NaN/Inf values, filling with 0")
        feature_df = feature_df.fillna(0)

    logger.info(f"Extraction complete: {feature_df.shape[1]} features")
    return feature_df


# ============================================================
# Statistical Functions
# ============================================================

def _compute_stat(series_batch: np.ndarray, func_name: str) -> np.ndarray:
    """
    Compute a statistical measure over a batch of time series.

    Parameters
    ----------
    series_batch : np.ndarray, shape (n_patients, window_len)
    func_name : str

    Returns
    -------
    np.ndarray, shape (n_patients,)
    """
    if func_name == "mean":
        return np.nanmean(series_batch, axis=1)
    elif func_name == "std":
        return np.nanstd(series_batch, axis=1)
    elif func_name == "min":
        return np.nanmin(series_batch, axis=1)
    elif func_name == "max":
        return np.nanmax(series_batch, axis=1)
    elif func_name == "trend":
        return _batch_linear_trend(series_batch)
    elif func_name == "last_value":
        return series_batch[:, -1]
    elif func_name == "range":
        return np.nanmax(series_batch, axis=1) - np.nanmin(series_batch, axis=1)
    elif func_name == "cv":
        means = np.nanmean(series_batch, axis=1)
        stds = np.nanstd(series_batch, axis=1)
        cv = np.divide(stds, means, out=np.zeros_like(stds), where=(means != 0))
        return cv
    elif func_name == "skew":
        return _batch_skewness(series_batch)
    else:
        raise ValueError(f"Unsupported statistical function: {func_name}")


def _batch_linear_trend(series_batch: np.ndarray) -> np.ndarray:
    """
    Compute linear trend slope for a batch of time series.

    Uses least squares: slope = Cov(t, y) / Var(t).
    Positive = upward trend, negative = downward trend.
    """
    n_patients, window_len = series_batch.shape
    t = np.arange(window_len, dtype=float)

    t_mean = t.mean()
    t_var = np.var(t)
    if t_var == 0:
        return np.zeros(n_patients)

    y_mean = np.nanmean(series_batch, axis=1, keepdims=True)
    cov = np.nanmean((t[np.newaxis, :] - t_mean) * (series_batch - y_mean), axis=1)
    slopes = cov / t_var

    return slopes


def _batch_skewness(series_batch: np.ndarray) -> np.ndarray:
    """Compute skewness (distribution asymmetry measure)."""
    mean = np.nanmean(series_batch, axis=1, keepdims=True)
    std = np.nanstd(series_batch, axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    skew = np.nanmean(((series_batch - mean) / std) ** 3, axis=1)
    return skew


# ============================================================
# Clinically Derived Indicators
# ============================================================

def _compute_derived_features(
    time_series_3d: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute clinically meaningful derived indicators:
    - Lactate clearance: reflects tissue perfusion improvement
    - Shock index (HR/SBP): classic early warning indicator
    - Renal-hepatic burden: joint organ deterioration
    - Oxygenation trend: rate of PaO2/FiO2 decline
    """
    n_patients = time_series_3d.shape[0]
    derived = {}

    def _get_col(name):
        if name in feature_names:
            return feature_names.index(name)
        return None

    # 1. Lactate clearance = (first - last) / first
    lac_idx = _get_col("lactate")
    if lac_idx is not None:
        lac_first = time_series_3d[:, 0, lac_idx]
        lac_last = time_series_3d[:, -1, lac_idx]
        # Safe division: avoid division by zero and NaN propagation
        with np.errstate(invalid="ignore", divide="ignore"):
            clearance = np.where(
                (~np.isnan(lac_first)) & (lac_first > 0),
                (lac_first - lac_last) / lac_first,
                0.0
            )
        derived["lactate_clearance"] = clearance

    # 2. Shock index = heart_rate / sbp (mean over window)
    hr_idx = _get_col("heart_rate")
    sbp_idx = _get_col("sbp")
    if hr_idx is not None and sbp_idx is not None:
        hr_mean = np.nanmean(time_series_3d[:, :, hr_idx], axis=1)
        sbp_mean = np.nanmean(time_series_3d[:, :, sbp_idx], axis=1)
        shock_index = np.where(sbp_mean > 0, hr_mean / sbp_mean, 0)
        derived["shock_index_mean"] = shock_index

        si_series = time_series_3d[:, :, hr_idx] / np.where(
            time_series_3d[:, :, sbp_idx] > 0,
            time_series_3d[:, :, sbp_idx],
            1
        )
        derived["shock_index_trend"] = _batch_linear_trend(si_series)

    # 3. Renal-hepatic burden: creatinine * bilirubin (joint deterioration)
    cr_idx = _get_col("creatinine")
    bil_idx = _get_col("bilirubin")
    if cr_idx is not None and bil_idx is not None:
        cr_last = time_series_3d[:, -1, cr_idx]
        bil_last = time_series_3d[:, -1, bil_idx]
        derived["renal_hepatic_burden"] = cr_last * bil_last

    # 4. Oxygenation deterioration rate
    pf_idx = _get_col("pao2_fio2")
    if pf_idx is not None:
        derived["pao2_fio2_trend"] = _batch_linear_trend(time_series_3d[:, :, pf_idx])

    logger.info(f"Derived features: {list(derived.keys())}")
    return derived
