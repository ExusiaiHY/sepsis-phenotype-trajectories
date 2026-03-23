"""
sepsis2019_bridge.py - Bridge PhysioNet/CinC 2019 sepsis data into S0-compatible tensors.

Purpose:
  Convert locally available Sepsis 2019 PSV-derived tensors into the same
  directory structure used by `data/s0/`, so downstream S1.5 fine-tuning can
  reuse the existing loaders with minimal branching.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from s0.preprocessor import _forward_fill_inplace
from s0.schema import (
    CONTINUOUS_INDEX,
    CONTINUOUS_NAMES,
    N_CONTINUOUS,
    N_INTERVENTIONS,
    N_PROXY,
    STATIC_FIELDS,
    schema_to_feature_dict,
)
from s0.splits import build_splits

logger = logging.getLogger("s15.sepsis2019_bridge")

# Matches `src/load_sepsis2019.py` PROJECT_FEATURES.
DEFAULT_SEPSIS2019_FEATURES = [
    "heart_rate",
    "spo2",
    "temperature",
    "sbp",
    "map",
    "dbp",
    "resp_rate",
    "base_excess",
    "bicarbonate",
    "fio2",
    "ph",
    "paco2",
    "spo2_invasive",
    "ast",
    "bun",
    "alkaline_phosphatase",
    "calcium",
    "chloride",
    "creatinine",
    "bilirubin_direct",
    "glucose",
    "lactate",
    "magnesium",
    "phosphate",
    "potassium",
    "bilirubin",
    "troponin_i",
    "hematocrit",
    "hemoglobin",
    "ptt",
    "wbc",
    "fibrinogen",
    "platelet",
    "vasopressor",
    "mechanical_vent",
    "rrt",
]


def build_bridge_bundle(
    time_series: np.ndarray,
    patient_info: pd.DataFrame,
    *,
    source_feature_names: list[str] | tuple[str, ...] = DEFAULT_SEPSIS2019_FEATURES,
) -> dict:
    """
    Convert Sepsis 2019 tensors into S0-aligned raw arrays + static frame.

    Parameters
    ----------
    time_series:
        Raw Sepsis 2019 tensor of shape (N, 48, F_src) with NaNs preserved.
    patient_info:
        Metadata frame returned by the Sepsis 2019 loader.
    source_feature_names:
        Feature names corresponding to axis 2 of `time_series`.
    """
    if time_series.ndim != 3:
        raise ValueError(f"Expected time_series to be 3D, got {time_series.shape}")

    n_patients, n_hours, _ = time_series.shape
    source_index = {name: idx for idx, name in enumerate(source_feature_names)}

    continuous = np.full((n_patients, n_hours, N_CONTINUOUS), np.nan, dtype=np.float32)
    mapped_features = []
    missing_features = []

    for feature_name in CONTINUOUS_NAMES:
        dst_idx = CONTINUOUS_INDEX[feature_name]
        src_idx = source_index.get(feature_name)
        if src_idx is None:
            missing_features.append(feature_name)
            continue
        continuous[:, :, dst_idx] = time_series[:, :, src_idx].astype(np.float32, copy=False)
        mapped_features.append(feature_name)

    masks_cont = np.isfinite(continuous).astype(np.float32)
    interventions = np.full((n_patients, n_hours, N_INTERVENTIONS), np.nan, dtype=np.float32)
    masks_int = np.zeros((n_patients, n_hours, N_INTERVENTIONS), dtype=np.float32)
    proxy, masks_proxy = _derive_proxy_indicators(continuous)
    static = _build_static_frame(patient_info, n_patients)

    report = {
        "n_patients": int(n_patients),
        "n_hours": int(n_hours),
        "mapped_features": mapped_features,
        "missing_features": missing_features,
        "feature_coverage": round(len(mapped_features) / max(len(CONTINUOUS_NAMES), 1), 4),
        "sepsis_prevalence": round(float(static["sepsis_label"].mean()), 4),
        "mortality_proxy_prevalence": round(float(static["mortality_inhospital"].mean()), 4),
        "overall_missing_rate_before_imputation": round(float(1.0 - masks_cont.mean()), 4),
    }

    return {
        "continuous": continuous,
        "masks_continuous": masks_cont,
        "interventions": interventions,
        "masks_interventions": masks_int,
        "proxy_indicators": proxy,
        "masks_proxy": masks_proxy,
        "static": static,
        "report": report,
    }


def preprocess_continuous(
    continuous: np.ndarray,
    *,
    reference_stats_path: Path | None = None,
    max_forward_fill_hours: int = 6,
    outlier_sigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """
    Forward-fill, impute, clip, and standardize continuous channels.

    When `reference_stats_path` is provided, the S0 preprocessing statistics are
    reused so the bridged source is numerically closer to the already-pretrained
    S1.5 encoder input space.
    """
    processed = continuous.astype(np.float32, copy=True)
    n_patients, _, n_features = processed.shape

    for patient_idx in range(n_patients):
        for feature_idx in range(n_features):
            _forward_fill_inplace(processed[patient_idx, :, feature_idx], max_gap=max_forward_fill_hours)

    with np.errstate(all="ignore"):
        if reference_stats_path is not None:
            with open(reference_stats_path, encoding="utf-8") as f:
                ref = json.load(f)
            feature_medians = np.asarray(ref["feature_medians"], dtype=np.float32)
            norm_means = np.asarray(ref["norm_means"], dtype=np.float32)
            norm_stds = np.asarray(ref["norm_stds"], dtype=np.float32)
            norm_stds = np.where(norm_stds < 1e-8, 1.0, norm_stds)
            stats_source = "reference"
        else:
            flat = processed.copy().reshape(-1, n_features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                feature_medians = np.nanmedian(flat, axis=0)
                means_preclip = np.nanmean(flat, axis=0)
                stds_preclip = np.nanstd(flat, axis=0)
            feature_medians = np.nan_to_num(feature_medians, nan=0.0)
            means_preclip = np.nan_to_num(means_preclip, nan=0.0)
            stds_preclip = np.nan_to_num(stds_preclip, nan=1.0)
            lower = means_preclip - outlier_sigma * np.where(stds_preclip < 1e-8, 1.0, stds_preclip)
            upper = means_preclip + outlier_sigma * np.where(stds_preclip < 1e-8, 1.0, stds_preclip)
            processed = np.clip(np.where(np.isnan(processed), feature_medians, processed), lower, upper)
            flat = processed.reshape(-1, n_features)
            norm_means = np.mean(flat, axis=0)
            norm_stds = np.std(flat, axis=0)
            norm_stds = np.where(norm_stds < 1e-8, 1.0, norm_stds)
            processed = (processed - norm_means) / norm_stds
            processed = np.nan_to_num(processed, nan=0.0).astype(np.float32, copy=False)
            return processed, {
                "n_features": int(n_features),
                "max_forward_fill_hours": max_forward_fill_hours,
                "outlier_sigma": outlier_sigma,
                "stats_source": "self",
                "feature_medians": feature_medians.tolist(),
                "norm_means": norm_means.tolist(),
                "norm_stds": norm_stds.tolist(),
            }

    processed = np.where(np.isnan(processed), feature_medians[None, None, :], processed)
    lower = norm_means - outlier_sigma * norm_stds
    upper = norm_means + outlier_sigma * norm_stds
    processed = np.clip(processed, lower[None, None, :], upper[None, None, :])
    processed = (processed - norm_means[None, None, :]) / norm_stds[None, None, :]
    processed = np.nan_to_num(processed, nan=0.0).astype(np.float32, copy=False)

    return processed, {
        "n_features": int(n_features),
        "max_forward_fill_hours": max_forward_fill_hours,
        "outlier_sigma": outlier_sigma,
        "stats_source": stats_source,
        "reference_stats_path": str(reference_stats_path) if reference_stats_path else None,
        "feature_medians": feature_medians.tolist(),
        "norm_means": norm_means.tolist(),
        "norm_stds": norm_stds.tolist(),
    }


def write_bridge_dataset(
    output_dir: Path,
    bundle: dict,
    *,
    reference_stats_path: Path | None = None,
    split_method: str = "random",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = "sepsis_label",
) -> dict:
    """Persist the bridged dataset in an S0-compatible on-disk layout."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_aligned"
    proc_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "continuous",
        "masks_continuous",
        "interventions",
        "masks_interventions",
        "proxy_indicators",
        "masks_proxy",
    ):
        np.save(raw_dir / f"{name}.npy", bundle[name])

    processed, preprocess_stats = preprocess_continuous(
        bundle["continuous"],
        reference_stats_path=reference_stats_path,
    )
    np.save(proc_dir / "continuous.npy", processed)
    np.save(proc_dir / "masks_continuous.npy", bundle["masks_continuous"])
    np.save(proc_dir / "interventions.npy", bundle["interventions"])
    np.save(proc_dir / "masks_interventions.npy", bundle["masks_interventions"])
    np.save(proc_dir / "proxy_indicators.npy", bundle["proxy_indicators"])
    np.save(proc_dir / "masks_proxy.npy", bundle["masks_proxy"])

    static = bundle["static"].copy()
    static_path = output_dir / "static.csv"
    static.to_csv(static_path, index=False)

    with open(output_dir / "feature_dict.json", "w", encoding="utf-8") as f:
        json.dump(schema_to_feature_dict(), f, indent=2)

    with open(proc_dir / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(preprocess_stats, f, indent=2)

    splits = build_splits(
        static_path=static_path,
        output_path=output_dir / "splits.json",
        method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        stratify_by=stratify_by,
    )

    bridge_report = dict(bundle["report"])
    bridge_report["split_method"] = split_method
    bridge_report["split_sizes"] = splits["metadata"]["sizes"]
    bridge_report["stratify_by"] = stratify_by
    bridge_report["reference_stats_path"] = str(reference_stats_path) if reference_stats_path else None

    with open(output_dir / "bridge_report.json", "w", encoding="utf-8") as f:
        json.dump(bridge_report, f, indent=2)

    logger.info("Saved Sepsis2019 bridge dataset to %s", output_dir)
    logger.info(
        "Patients=%d, sepsis prevalence=%.3f, coverage=%s/%s",
        bridge_report["n_patients"],
        bridge_report["sepsis_prevalence"],
        len(bridge_report["mapped_features"]),
        len(CONTINUOUS_NAMES),
    )
    return bridge_report


def _derive_proxy_indicators(continuous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proxy = np.zeros((continuous.shape[0], continuous.shape[1], N_PROXY), dtype=np.float32)
    masks_proxy = np.zeros_like(proxy)

    map_idx = CONTINUOUS_INDEX["map"]
    fio2_idx = CONTINUOUS_INDEX["fio2"]

    map_vals = continuous[:, :, map_idx]
    fio2_vals = continuous[:, :, fio2_idx]

    map_valid = np.isfinite(map_vals)
    fio2_valid = np.isfinite(fio2_vals)

    proxy[:, :, 0] = np.where(map_valid & (map_vals < 65.0), 1.0, 0.0)
    proxy[:, :, 1] = np.where(fio2_valid & (fio2_vals > 0.21), 1.0, 0.0)
    masks_proxy[:, :, 0] = map_valid.astype(np.float32)
    masks_proxy[:, :, 1] = fio2_valid.astype(np.float32)
    return proxy, masks_proxy


def _build_static_frame(patient_info: pd.DataFrame, n_patients: int) -> pd.DataFrame:
    patient_info = patient_info.reset_index(drop=True).copy()

    def get_col(name: str, default):
        if name in patient_info.columns:
            return patient_info[name]
        if np.isscalar(default):
            return pd.Series([default] * n_patients)
        return pd.Series(default)

    patient_id = get_col("patient_id", [f"SEPSIS2019_{idx + 1:05d}" for idx in range(n_patients)])
    gender = pd.to_numeric(get_col("gender", 0), errors="coerce").fillna(0).astype(int)
    sepsis_label = pd.to_numeric(get_col("sepsis_label", 0), errors="coerce").fillna(0).astype(int)
    mortality_proxy = pd.to_numeric(get_col("mortality_28d", 0), errors="coerce").fillna(0).astype(int)

    static = pd.DataFrame({
        "patient_id": patient_id.astype(str),
        "age": pd.to_numeric(get_col("age", np.nan), errors="coerce"),
        "sex": np.where(gender > 0, "male", "female"),
        "height_cm": np.nan,
        "weight_kg": np.nan,
        "icu_type": pd.to_numeric(get_col("icu_type", np.nan), errors="coerce"),
        "icu_los_hours": pd.to_numeric(get_col("icu_los", 48.0), errors="coerce"),
        "mortality_inhospital": mortality_proxy,
        "mortality_source": np.where(mortality_proxy > 0, "proxy_terminal_vitals", "proxy_terminal_vitals"),
        "center_id": get_col("center_id", "sepsis2019").astype(str),
        "set_name": get_col("set_name", "all").astype(str),
        "data_source": "physionet2019_sepsis",
        "sepsis_onset_hour": pd.to_numeric(get_col("sepsis_onset_hour", np.nan), errors="coerce"),
        "anchor_time_type": np.where(sepsis_label > 0, "sepsis_onset", "icu_admission"),
        "sepsis_label": sepsis_label,
        "actual_timesteps": pd.to_numeric(get_col("actual_timesteps", 48), errors="coerce"),
    })

    for field in STATIC_FIELDS:
        if field not in static.columns:
            static[field] = np.nan

    ordered = static[STATIC_FIELDS].copy()
    ordered["sepsis_label"] = static["sepsis_label"].astype(int)
    ordered["actual_timesteps"] = static["actual_timesteps"].astype(int)
    return ordered
