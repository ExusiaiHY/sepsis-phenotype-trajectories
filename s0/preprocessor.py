"""
preprocessor.py - Transform raw_aligned → processed tensors.

Purpose:
  Apply forward fill, median imputation, outlier clipping, and normalization
  to raw_aligned data. Respects per-variable metadata (imputation_allowed,
  normalization_group). Masks are preserved through processing.

  Operates on continuous tensor only. Intervention and proxy tensors are
  copied without modification (binary values should not be imputed or normalized).

Connects to:
  - schema.py for per-variable metadata
  - physionet2012_extractor.py produces raw_aligned inputs
  - dataset.py consumes processed outputs

How to run:
  Called by scripts/s0_prepare.py. Not standalone.

Expected output artifacts:
  data/s0/processed/continuous.npy (imputed + normalized)
  data/s0/processed/interventions.npy (copied from raw_aligned)
  data/s0/processed/proxy_indicators.npy (copied from raw_aligned)
  data/s0/processed/masks_*.npy (copied from raw_aligned, unchanged)
  data/s0/processed/preprocess_stats.json
"""
from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np

from s0.schema import CONTINUOUS_SCHEMA, N_CONTINUOUS

logger = logging.getLogger("s0.preprocessor")


def preprocess_raw_aligned(
    input_dir: Path,
    output_dir: Path,
    max_forward_fill_hours: int = 6,
    outlier_sigma: float = 4.0,
    normalization: str = "standard",
    reference_stats_path: Path | None = None,
) -> dict:
    """
    Preprocess raw_aligned data and save to processed directory.

    Steps for continuous tensor:
      1. Forward fill (per patient, per variable, up to max_forward_fill_hours)
      2. Global median imputation (only where imputation_allowed=True)
      3. Outlier clipping (per normalization_group)
      4. Standardization (z-score, per normalization_group)

    Intervention / proxy / mask tensors are copied unchanged.

    Returns preprocessing statistics dict.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw_aligned continuous data
    continuous = np.load(input_dir / "continuous.npy").copy()
    masks_cont = np.load(input_dir / "masks_continuous.npy")
    n_patients, n_hours, n_features = continuous.shape

    logger.info(f"Preprocessing: {n_patients} patients, {n_hours} hours, {n_features} features")

    # --- Step 1: Forward fill ---
    for p in range(n_patients):
        for f in range(n_features):
            if not CONTINUOUS_SCHEMA[f].imputation_allowed:
                continue
            _forward_fill_inplace(continuous[p, :, f], max_gap=max_forward_fill_hours)

    # --- Step 2: Load reference preprocessing stats if requested ---
    reference_stats = None
    if reference_stats_path is not None:
        reference_stats_path = Path(reference_stats_path)
        if not reference_stats_path.exists():
            raise FileNotFoundError(f"Reference preprocessing stats not found: {reference_stats_path}")
        with open(reference_stats_path) as f:
            reference_stats = json.load(f)
        logger.info("Using reference preprocessing stats from %s", reference_stats_path)

    # --- Step 3: Global median imputation ---
    if reference_stats is not None:
        feature_medians = np.asarray(reference_stats["feature_medians"], dtype=np.float32)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feature_medians = np.nanmedian(continuous.reshape(-1, n_features), axis=0)
        feature_medians = np.nan_to_num(feature_medians, nan=0.0)

    for f in range(n_features):
        if not CONTINUOUS_SCHEMA[f].imputation_allowed:
            continue
        nan_mask = np.isnan(continuous[:, :, f])
        continuous[:, :, f] = np.where(nan_mask, feature_medians[f], continuous[:, :, f])

    # --- Step 4: Outlier clipping ---
    if reference_stats is not None:
        feat_means = np.asarray(reference_stats["norm_means"], dtype=np.float32)
        feat_stds = np.asarray(reference_stats["norm_stds"], dtype=np.float32)
        feat_stds = np.where(feat_stds < 1e-8, 1.0, feat_stds)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flat = continuous.reshape(-1, n_features)
            feat_means = np.nanmean(flat, axis=0)
            feat_stds = np.nanstd(flat, axis=0)
            feat_stds = np.where(feat_stds < 1e-8, 1.0, feat_stds)

    for f in range(n_features):
        group = CONTINUOUS_SCHEMA[f].normalization_group
        if group == "binary" or group == "none":
            continue
        lower = feat_means[f] - outlier_sigma * feat_stds[f]
        upper = feat_means[f] + outlier_sigma * feat_stds[f]
        continuous[:, :, f] = np.clip(continuous[:, :, f], lower, upper)

    # --- Step 5: Standardization ---
    if reference_stats is not None:
        norm_means = np.asarray(reference_stats["norm_means"], dtype=np.float32)
        norm_stds = np.asarray(reference_stats["norm_stds"], dtype=np.float32)
        norm_stds = np.where(norm_stds < 1e-8, 1.0, norm_stds)
    else:
        # Recompute stats after clipping
        flat = continuous.reshape(-1, n_features)
        norm_means = np.mean(flat, axis=0)
        norm_stds = np.std(flat, axis=0)
        norm_stds = np.where(norm_stds < 1e-8, 1.0, norm_stds)

    if normalization == "standard":
        for f in range(n_features):
            group = CONTINUOUS_SCHEMA[f].normalization_group
            if group == "binary" or group == "none":
                continue
            continuous[:, :, f] = (continuous[:, :, f] - norm_means[f]) / norm_stds[f]

    # Final NaN safety
    continuous = np.nan_to_num(continuous, nan=0.0)

    # --- Save processed continuous ---
    np.save(output_dir / "continuous.npy", continuous)

    # --- Copy other tensors unchanged ---
    for fname in ["interventions.npy", "proxy_indicators.npy",
                   "masks_continuous.npy", "masks_interventions.npy", "masks_proxy.npy"]:
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # --- Save preprocessing stats ---
    stats = {
        "n_patients": n_patients,
        "n_hours": n_hours,
        "n_features": n_features,
        "max_forward_fill_hours": max_forward_fill_hours,
        "outlier_sigma": outlier_sigma,
        "normalization": normalization,
        "reference_stats_path": str(reference_stats_path) if reference_stats_path is not None else None,
        "feature_medians": feature_medians.tolist(),
        "norm_means": norm_means.tolist(),
        "norm_stds": norm_stds.tolist(),
        "per_feature_missing_rate_before_imputation": [
            float(1.0 - masks_cont[:, :, f].mean()) for f in range(n_features)
        ],
    }
    with open(output_dir / "preprocess_stats.json", "w") as fp:
        json.dump(stats, fp, indent=2)

    overall_missing = 1.0 - masks_cont.mean()
    logger.info(f"Preprocessing complete. Overall missing rate (pre-imputation): {overall_missing:.1%}")
    logger.info(f"Saved to {output_dir}")

    return stats


def _forward_fill_inplace(arr: np.ndarray, max_gap: int = 6) -> None:
    """Forward fill NaN values in a 1D array in-place, up to max_gap steps."""
    last_valid = np.nan
    gap_count = 0
    for i in range(len(arr)):
        if np.isfinite(arr[i]):
            last_valid = arr[i]
            gap_count = 0
        elif np.isfinite(last_valid) and gap_count < max_gap:
            arr[i] = last_valid
            gap_count += 1
