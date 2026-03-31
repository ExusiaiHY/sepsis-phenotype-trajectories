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

import logging

import numpy as np

logger = logging.getLogger("s6.missingness")


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
