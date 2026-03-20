"""
preprocess.py - Data cleaning and preprocessing module

Responsibilities:
1. Time resampling and alignment
2. Missing value handling (forward fill -> median fill -> zero fill)
3. Outlier detection and clipping (sigma-based)
4. Data normalization (StandardScaler / MinMax / Robust)
5. Output a clean 3D time-series tensor

Design notes:
- Processing order: outliers -> missing fill -> normalization
  (outliers must be handled first to avoid polluting mean/std calculations)
- Normalization parameters are saved for reuse on external validation data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import warnings
from pathlib import Path

from utils import setup_logger, timer, resolve_path

logger = setup_logger(__name__)


# ============================================================
# Main Preprocessing Pipeline
# ============================================================

@timer
def preprocess_pipeline(
    time_series_3d: np.ndarray,
    config: dict,
    fit: bool = True,
    scaler_params: dict | None = None,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    time_series_3d : np.ndarray, shape (n_patients, n_timesteps, n_features)
        Raw time-series data.
    config : dict
        Global configuration.
    fit : bool
        True = fit normalization params on this data (training set).
        False = use externally provided params (validation/test set).
    scaler_params : dict | None
        When fit=False, provide pre-computed normalization params.
    feature_names : list[str] | None
        Feature name list. Used to identify continuous vs binary columns.
        If None, derives from config variables.

    Returns
    -------
    processed : np.ndarray
        Preprocessed data (same shape, missing values filled).
    params : dict
        Normalization parameters for external validation alignment.
    """
    prep_cfg = config["preprocess"]

    n_patients, n_timesteps, n_features = time_series_3d.shape
    logger.info(f"Preprocessing started: {n_patients} patients, {n_timesteps} timesteps, {n_features} features")

    data = time_series_3d.copy()

    original_missing = np.isnan(data).sum()
    logger.info(f"Original missing values: {original_missing} ({original_missing / data.size:.1%})")

    # Determine continuous columns (normalize) vs binary columns (skip normalization)
    continuous_cols = _get_continuous_cols(config, n_features, feature_names)

    logger.info(f"  Continuous features: {len(continuous_cols)}, Binary/skip: {n_features - len(continuous_cols)}")

    # Step 1: Outlier handling
    if prep_cfg["outlier_method"] == "clip":
        data = _clip_outliers(data, continuous_cols, sigma=prep_cfg["outlier_sigma"])
    elif prep_cfg["outlier_method"] == "remove":
        data = _remove_outliers(data, continuous_cols, sigma=prep_cfg["outlier_sigma"])

    # Step 2: Missing value filling
    data = _fill_missing(data, strategy=prep_cfg["missing_strategy"])

    remaining_missing = np.isnan(data).sum()
    logger.info(f"Remaining missing after fill: {remaining_missing}")

    # Final fallback: fill all remaining NaN with 0
    data = np.nan_to_num(data, nan=0.0)

    # Step 3: Normalization (continuous variables only)
    if fit:
        data, params = _normalize(data, continuous_cols, method=prep_cfg["normalization"])
    else:
        if scaler_params is None:
            raise ValueError("scaler_params must be provided when fit=False")
        data, params = _normalize(
            data, continuous_cols,
            method=prep_cfg["normalization"],
            precomputed_params=scaler_params,
        )

    logger.info(f"Preprocessing complete. Data range: [{data.min():.3f}, {data.max():.3f}]")

    return data, params


def _get_continuous_cols(
    config: dict, n_features: int, feature_names: list[str] | None = None
) -> list[int]:
    """
    Determine which feature columns are continuous (should be normalized).

    Binary treatment columns (vasopressor, mechanical_vent, rrt) should NOT
    be normalized. SOFA scores and derived indices are normalized as they
    represent clinical severity scales.
    """
    # Known binary columns that should NOT be normalized
    binary_names = {
        "vasopressor", "mechanical_vent", "rrt",
        "shock_onset", "mortality_28d",
    }

    if feature_names is not None and len(feature_names) == n_features:
        # Use actual feature names to identify continuous columns
        return [i for i, name in enumerate(feature_names) if name not in binary_names]

    # Fallback: use config variables to count continuous cols
    n_vitals = len(config["variables"]["vitals"])
    n_labs = len(config["variables"]["labs"])
    n_continuous = n_vitals + n_labs

    if n_continuous <= n_features:
        return list(range(n_continuous))

    # If n_features > expected, assume all are continuous except known binary ones
    return list(range(n_features))


# ============================================================
# Outlier Handling
# ============================================================

def _clip_outliers(
    data: np.ndarray,
    col_indices: list[int],
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Clip values beyond +/- sigma standard deviations to the boundary.

    Uses 4-sigma rather than 3-sigma because ICU data contains extreme but
    clinically real values (e.g., heart rate >150 during shock).
    """
    clipped_count = 0
    for col in col_indices:
        values = data[:, :, col]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            continue
        lower, upper = mean - sigma * std, mean + sigma * std
        mask = (~np.isnan(values)) & ((values < lower) | (values > upper))
        clipped_count += mask.sum()
        values[mask] = np.clip(values[mask], lower, upper)
        data[:, :, col] = values

    logger.info(f"Outlier clipping ({sigma}sigma): {clipped_count} values clipped")
    return data


def _remove_outliers(
    data: np.ndarray,
    col_indices: list[int],
    sigma: float = 4.0,
) -> np.ndarray:
    """Set outlier values to NaN (handled later by missing fill strategy)."""
    for col in col_indices:
        values = data[:, :, col]
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            continue
        lower, upper = mean - sigma * std, mean + sigma * std
        mask = (~np.isnan(values)) & ((values < lower) | (values > upper))
        values[mask] = np.nan
        data[:, :, col] = values
    return data


# ============================================================
# Missing Value Filling
# ============================================================

def _fill_missing(data: np.ndarray, strategy: str = "forward_fill_then_median") -> np.ndarray:
    """
    Fill missing values.

    "forward_fill_then_median":
    1. Forward fill per patient (use most recent valid value)
    2. Fill remaining NaN with global median of the variable
    """
    n_patients, n_timesteps, n_features = data.shape

    if strategy == "forward_fill_then_median":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            global_medians = np.nanmedian(data.reshape(-1, n_features), axis=0)
        # Replace any NaN medians (all-NaN columns) with 0
        global_medians = np.where(np.isnan(global_medians), 0.0, global_medians)

        for i in range(n_patients):
            for j in range(n_features):
                series = data[i, :, j]
                mask = np.isnan(series)
                if mask.all():
                    data[i, :, j] = global_medians[j]
                    continue
                if mask.any():
                    filled = pd.Series(series).ffill().values
                    still_nan = np.isnan(filled)
                    filled[still_nan] = global_medians[j]
                    data[i, :, j] = filled

    elif strategy == "median":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            global_medians = np.nanmedian(data.reshape(-1, n_features), axis=0)
        # Replace any NaN medians (all-NaN columns) with 0
        global_medians = np.where(np.isnan(global_medians), 0.0, global_medians)
        for j in range(n_features):
            mask = np.isnan(data[:, :, j])
            data[:, :, j][mask] = global_medians[j]

    elif strategy == "zero":
        data = np.nan_to_num(data, nan=0.0)

    else:
        raise ValueError(f"Unsupported missing fill strategy: {strategy}")

    return data


# ============================================================
# Normalization
# ============================================================

def _normalize(
    data: np.ndarray,
    col_indices: list[int],
    method: str = "standard",
    precomputed_params: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Normalize continuous variables.
    """
    if precomputed_params is not None:
        params = precomputed_params
    else:
        params = _compute_norm_params(data, col_indices, method)

    for col in col_indices:
        if method == "standard":
            mean = params["means"][col]
            std = params["stds"][col]
            if std > 0:
                data[:, :, col] = (data[:, :, col] - mean) / std
        elif method == "minmax":
            vmin = params["mins"][col]
            vmax = params["maxs"][col]
            if vmax > vmin:
                data[:, :, col] = (data[:, :, col] - vmin) / (vmax - vmin)
        elif method == "robust":
            median = params["medians"][col]
            iqr = params["iqrs"][col]
            if iqr > 0:
                data[:, :, col] = (data[:, :, col] - median) / iqr

    return data, params


def _compute_norm_params(
    data: np.ndarray, col_indices: list[int], method: str
) -> dict:
    """Compute normalization parameters."""
    params = {"method": method}
    flat = data.reshape(-1, data.shape[2])

    if method == "standard":
        params["means"] = {col: np.nanmean(flat[:, col]) for col in col_indices}
        params["stds"] = {col: np.nanstd(flat[:, col]) for col in col_indices}
    elif method == "minmax":
        params["mins"] = {col: np.nanmin(flat[:, col]) for col in col_indices}
        params["maxs"] = {col: np.nanmax(flat[:, col]) for col in col_indices}
    elif method == "robust":
        params["medians"] = {col: np.nanmedian(flat[:, col]) for col in col_indices}
        params["iqrs"] = {
            col: np.nanpercentile(flat[:, col], 75) - np.nanpercentile(flat[:, col], 25)
            for col in col_indices
        }

    return params


# ============================================================
# Missing Pattern Analysis
# ============================================================

def analyze_missing_pattern(
    time_series_3d: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Analyze missing patterns for each variable.

    Returns
    -------
    pd.DataFrame with per-variable missing statistics.
    """
    n_patients, n_timesteps, n_features = time_series_3d.shape
    records = []
    for j, name in enumerate(feature_names):
        values = time_series_3d[:, :, j]
        total = values.size
        missing = np.isnan(values).sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
        records.append({
            "variable": name,
            "total_values": total,
            "missing_count": missing,
            "missing_rate": missing / total,
            "mean": float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan"),
            "std": float(np.nanstd(values)) if not np.all(np.isnan(values)) else float("nan"),
            "min": float(np.nanmin(values)) if not np.all(np.isnan(values)) else float("nan"),
            "max": float(np.nanmax(values)) if not np.all(np.isnan(values)) else float("nan"),
        })
    return pd.DataFrame(records)