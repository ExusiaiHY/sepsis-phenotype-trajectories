"""
missingness_encoder.py - Informative missingness feature engineering.

Enhances the existing mask-aware encoder by extracting temporal patterns
from observation masks that distinguish clinical informative missingness
from random technical noise.

Inspired by Glocal-IB (arXiv:2510.04910) and SAITS (WenjieDu et al.),
but implemented as a lightweight feature engineering layer that integrates
directly with the existing S1/S1.5 encoder without architectural changes.

Produces an augmented mask tensor:
  Original: (N, T, F)   binary 0/1 observation mask
  Enhanced: (N, T, F*3) [original_mask, gap_length, density_change]

This feeds into the existing `concat([x, enhanced_mask])` input path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("s6.missingness")

DEFAULT_SELECTED_MISSINGNESS_FEATURES = [
    "map",
    "lactate",
    "creatinine",
    "bilirubin",
    "gcs",
]


def compute_gap_lengths(masks: np.ndarray) -> np.ndarray:
    """
    Compute per-variable consecutive gap length at each timestep.

    For each position (patient, time, feature):
      - If observed (mask=1): gap = 0
      - If missing (mask=0): gap = number of consecutive missing hours ending here

    Parameters
    ----------
    masks: (N, T, F) binary observation masks

    Returns
    -------
    gaps: (N, T, F) float32, normalized by T to range [0, 1]
    """
    N, T, F = masks.shape
    gaps = np.zeros_like(masks, dtype=np.float32)

    for f in range(F):
        for i in range(N):
            current_gap = 0
            for t in range(T):
                if masks[i, t, f] > 0.5:
                    current_gap = 0
                else:
                    current_gap += 1
                gaps[i, t, f] = current_gap

    # Normalize by sequence length
    gaps = gaps / T
    return gaps


def compute_gap_lengths_vectorized(masks: np.ndarray) -> np.ndarray:
    """
    Vectorized version of gap length computation.

    Parameters
    ----------
    masks: (N, T, F) binary observation masks

    Returns
    -------
    gaps: (N, T, F) float32, normalized to [0, 1]
    """
    N, T, F = masks.shape
    gaps = np.zeros((N, T, F), dtype=np.float32)
    missing = (masks < 0.5).astype(np.float32)

    for t in range(T):
        if t == 0:
            gaps[:, t, :] = missing[:, t, :]
        else:
            gaps[:, t, :] = (gaps[:, t - 1, :] + missing[:, t, :]) * missing[:, t, :]

    return gaps / T


def compute_density_change(masks: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Compute local observation density change rate.

    For each timestep, compare the observation density in the surrounding
    window to the previous window. A sharp drop in density may indicate
    clinical withdrawal of monitoring (informative missingness).

    Parameters
    ----------
    masks: (N, T, F) binary observation masks
    window: sliding window size in hours (default: 6h)

    Returns
    -------
    density_change: (N, T, F) float32, positive = increasing monitoring,
                    negative = decreasing monitoring
    """
    N, T, F = masks.shape
    density_change = np.zeros((N, T, F), dtype=np.float32)

    for t in range(window, T):
        current_window = masks[:, max(0, t - window):t, :]
        prev_window = masks[:, max(0, t - 2 * window):max(0, t - window), :]

        current_density = current_window.mean(axis=1)
        prev_density = prev_window.mean(axis=1) if prev_window.shape[1] > 0 else current_density

        density_change[:, t, :] = current_density - prev_density

    return density_change


def compute_missingness_features(
    masks: np.ndarray,
    gap_window: int = 6,
) -> np.ndarray:
    """
    Compute the full informative missingness feature tensor.

    Combines three signals:
      1. Original binary mask (already in pipeline)
      2. Gap length: how long since last observation (clinical attention decay)
      3. Density change: local monitoring intensity trend (withdrawal signal)

    Parameters
    ----------
    masks: (N, T, F) binary observation masks
    gap_window: window for density change computation

    Returns
    -------
    enhanced: (N, T, F*3) concatenated [mask, gap_length, density_change]
    """
    logger.info("Computing informative missingness features...")

    # Gap lengths (vectorized)
    gaps = compute_gap_lengths_vectorized(masks)
    logger.info("  Gap lengths: shape=%s, mean=%.4f", gaps.shape, gaps.mean())

    # Density change
    density = compute_density_change(masks, window=gap_window)
    logger.info("  Density change: shape=%s, range=[%.4f, %.4f]",
                density.shape, density.min(), density.max())

    # Concatenate along feature axis
    enhanced = np.concatenate([
        masks.astype(np.float32),
        gaps,
        density,
    ], axis=-1)

    logger.info("  Enhanced mask: (N=%d, T=%d, F=%d) -> (N, T, %d)",
                masks.shape[0], masks.shape[1], masks.shape[2], enhanced.shape[2])
    return enhanced


def compute_patient_missingness_summary(masks: np.ndarray) -> dict:
    """
    Compute cohort-level missingness statistics for reporting.

    Returns
    -------
    dict with per-feature and overall missingness metrics
    """
    N, T, F = masks.shape

    overall_density = float(masks.mean())
    per_feature_density = masks.mean(axis=(0, 1)).tolist()

    # Informative missingness heuristics
    # Patients with sudden density drops (>50% in any 6h window)
    gaps = compute_gap_lengths_vectorized(masks)
    max_gap_per_patient = gaps.max(axis=(1, 2))

    return {
        "n_patients": N,
        "n_timesteps": T,
        "n_features": F,
        "overall_observation_density": round(overall_density, 4),
        "per_feature_density": [round(d, 4) for d in per_feature_density],
        "mean_max_gap_per_patient": round(float(max_gap_per_patient.mean()), 4),
        "patients_with_long_gaps": int((max_gap_per_patient > 0.5).sum()),
        "fraction_patients_long_gaps": round(float((max_gap_per_patient > 0.5).mean()), 4),
    }


def build_patient_missingness_features(
    masks: np.ndarray,
    feature_names: list[str] | None = None,
    *,
    gap_window: int = 6,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build patient-level informative-missingness covariates for downstream S6 use.

    This is a compact Glocal-IB-style proxy:
      - global summary signals across the full mask tensor
      - local signals on a small set of clinically important features

    Returns
    -------
    DataFrame with one row per patient.
    """
    N, _, F = masks.shape
    if feature_names is None:
        feature_names = [f"feature_{i:03d}" for i in range(F)]

    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    chosen = selected_features or DEFAULT_SELECTED_MISSINGNESS_FEATURES
    selected = [name for name in chosen if name in feature_to_idx]

    gaps = compute_gap_lengths_vectorized(masks)
    density_change = compute_density_change(masks, window=gap_window)
    observed_any = (masks.max(axis=1) > 0.5).astype(np.float32)

    features = {
        "miss_global_density": masks.mean(axis=(1, 2)).astype(np.float32),
        "miss_global_gap_mean": gaps.mean(axis=(1, 2)).astype(np.float32),
        "miss_global_gap_max": gaps.max(axis=(1, 2)).astype(np.float32),
        "miss_global_density_trend_mean": density_change.mean(axis=(1, 2)).astype(np.float32),
        "miss_global_density_trend_std": density_change.std(axis=(1, 2)).astype(np.float32),
        "miss_global_feature_coverage": observed_any.mean(axis=1).astype(np.float32),
    }

    for name in selected:
        idx = feature_to_idx[name]
        feature_mask = masks[:, :, idx]
        feature_gap = gaps[:, :, idx]
        feature_density_change = density_change[:, :, idx]
        features[f"miss_{name}_density"] = feature_mask.mean(axis=1).astype(np.float32)
        features[f"miss_{name}_gap_max"] = feature_gap.max(axis=1).astype(np.float32)
        features[f"miss_{name}_gap_last"] = feature_gap[:, -1].astype(np.float32)
        features[f"miss_{name}_density_trend_mean"] = feature_density_change.mean(axis=1).astype(np.float32)

    return pd.DataFrame(features).fillna(0.0)


def run_missingness_stage(
    *,
    masks: np.ndarray,
    output_dir: Path,
    config: dict | None = None,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Shared Stage-1 runner used by S6 local and external pipelines.

    Besides the original summary + enhanced tensor, this now also emits
    patient-level missingness covariates so Stage 1 is consumed by Stage 2.
    """
    cfg = {
        "gap_window": 6,
        "selected_features": DEFAULT_SELECTED_MISSINGNESS_FEATURES,
        "append_patient_features": True,
    }
    if config:
        cfg.update({k: v for k, v in config.items() if v is not None})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    masks = np.asarray(masks, dtype=np.float32)
    miss_summary = compute_patient_missingness_summary(masks)
    miss_summary_path = output_dir / "missingness_summary.json"
    with open(miss_summary_path, "w", encoding="utf-8") as f:
        json.dump(miss_summary, f, ensure_ascii=False, indent=2)
    logger.info("Missingness summary saved: %s", miss_summary_path)

    enhanced_masks = compute_missingness_features(
        masks,
        gap_window=int(cfg["gap_window"]),
    )
    enhanced_path = output_dir / "missingness_enhanced.npy"
    np.save(enhanced_path, enhanced_masks)
    logger.info("Enhanced mask tensor: %s -> %s", masks.shape, enhanced_masks.shape)

    if cfg.get("append_patient_features", True):
        features_df = build_patient_missingness_features(
            masks,
            feature_names=feature_names,
            gap_window=int(cfg["gap_window"]),
            selected_features=list(cfg.get("selected_features") or DEFAULT_SELECTED_MISSINGNESS_FEATURES),
        )
        features_path = output_dir / "missingness_patient_features.csv"
        feature_summary_path = output_dir / "missingness_covariate_summary.json"
        features_df.to_csv(features_path, index=False)
        feature_summary = {
            "enabled": True,
            "n_patients": int(len(features_df)),
            "n_output_columns": int(features_df.shape[1]),
            "selected_features": list(cfg.get("selected_features") or DEFAULT_SELECTED_MISSINGNESS_FEATURES),
            "artifacts": {
                "patient_features": str(features_path),
                "summary": str(feature_summary_path),
            },
        }
        with open(feature_summary_path, "w", encoding="utf-8") as f:
            json.dump(feature_summary, f, ensure_ascii=False, indent=2)
    else:
        features_df = pd.DataFrame(index=np.arange(len(masks)))
        feature_summary = {"enabled": False}

    return {
        "summary": miss_summary,
        "artifacts": {
            "summary": str(miss_summary_path),
            "enhanced_masks": str(enhanced_path),
        },
        "features_df": features_df,
        "feature_summary": feature_summary,
    }
