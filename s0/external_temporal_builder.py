"""
external_temporal_builder.py - Build S0-compatible external temporal bundles.

Purpose:
  Convert prepared external MIMIC-IV / eICU artifacts into the same S0 layout
  consumed by the S1.5 and S2 temporal pipeline:
    - raw_aligned/*.npy
    - processed/*.npy
    - static.csv
    - feature_dict.json
    - splits.json
    - data_manifest.json

  The external cohorts reuse the PhysioNet 2012 preprocessing statistics so the
  frozen S1.5 encoder sees numerically compatible inputs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from s0.manifest import generate_manifest
from s0.preprocessor import preprocess_raw_aligned
from s0.schema import (
    ANCHOR_ICU_ADMISSION,
    ANCHOR_SEPSIS_ONSET,
    CONTINUOUS_INDEX,
    CONTINUOUS_NAMES,
    N_CONTINUOUS,
    N_INTERVENTIONS,
    N_PROXY,
    STATIC_FIELDS,
    schema_to_feature_dict,
)
from s0.splits import build_splits

logger = logging.getLogger("s0.external")

MIMIC_CONTINUOUS_MAP = {
    "heart_rate": "heart_rate",
    "sbp": "sbp",
    "dbp": "dbp",
    "map": "map",
    "resp_rate": "resp_rate",
    "spo2": "spo2",
    "temperature": "temperature",
    "gcs": "gcs",
    "creatinine": "creatinine",
    "bun": "bun",
    "glucose": "glucose",
    "wbc": "wbc",
    "platelet": "platelet",
    "potassium": "potassium",
    "sodium": "sodium",
}

EICU_CONTINUOUS_MAP = {
    "heart_rate": "heart_rate",
    "sbp": "sbp",
    "dbp": "dbp",
    "map": "map",
    "resp_rate": "resp_rate",
    "spo2": "spo2",
    "temperature": "temperature",
    "creatinine": "creatinine",
    "wbc": "wbc",
    "platelet": "platelet",
    "lactate": "lactate",
    "bilirubin": "bilirubin",
}


def prepare_external_temporal_s0(
    source: str,
    output_dir: Path,
    processed_dir: Path,
    reference_stats_path: Path | None,
    n_hours: int = 48,
    split_method: str = "random",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_by: str = "mortality_inhospital",
    max_patients: int | None = None,
) -> dict:
    """
    Build a full external S0 bundle and return a compact run summary.
    """
    source = source.lower()
    output_dir = Path(output_dir)
    processed_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw_aligned"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source == "mimic":
        extraction_stats = _prepare_mimic_raw_aligned(
            processed_dir=processed_dir,
            output_dir=output_dir,
            raw_dir=raw_dir,
            n_hours=n_hours,
            max_patients=max_patients,
        )
    elif source == "eicu":
        extraction_stats = _prepare_eicu_raw_aligned(
            processed_dir=processed_dir,
            output_dir=output_dir,
            raw_dir=raw_dir,
            n_hours=n_hours,
            max_patients=max_patients,
        )
    else:
        raise ValueError(f"Unsupported external source: {source}")

    with open(output_dir / "feature_dict.json", "w", encoding="utf-8") as f:
        json.dump(schema_to_feature_dict(), f, indent=2)

    preprocess_stats = preprocess_raw_aligned(
        input_dir=raw_dir,
        output_dir=output_dir / "processed",
        max_forward_fill_hours=6,
        outlier_sigma=4.0,
        normalization="standard",
        reference_stats_path=reference_stats_path,
    )

    split_info = build_splits(
        static_path=output_dir / "static.csv",
        output_path=output_dir / "splits.json",
        method=split_method,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        stratify_by=stratify_by,
    )

    manifest_config = {
        "data": {
            "source": source,
            "n_hours": n_hours,
            "prepared_source_dir": str(processed_dir),
            "output_dir": str(output_dir),
            "reference_stats_path": str(reference_stats_path) if reference_stats_path is not None else None,
            "max_patients": max_patients,
        },
        "preprocess": {
            "max_forward_fill_hours": 6,
            "outlier_sigma": 4.0,
            "normalization": "standard",
        },
        "splits": {
            "method": split_method,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "stratify_by": stratify_by,
        },
        "manifest": {
            "notes": _manifest_notes_for_source(source, reference_stats_path),
        },
    }
    manifest = generate_manifest(output_dir, extraction_stats, preprocess_stats, manifest_config)

    return {
        "source": source,
        "output_dir": str(output_dir),
        "prepared_source_dir": str(processed_dir),
        "reference_stats_path": str(reference_stats_path) if reference_stats_path is not None else None,
        "extraction_stats": extraction_stats,
        "preprocess_stats_path": str(output_dir / "processed" / "preprocess_stats.json"),
        "splits_path": str(output_dir / "splits.json"),
        "manifest_path": str(output_dir / "data_manifest.json"),
        "split_sizes": split_info["metadata"]["sizes"],
        "manifest_generated_at": manifest["generated_at"],
    }


def _prepare_mimic_raw_aligned(
    processed_dir: Path,
    output_dir: Path,
    raw_dir: Path,
    n_hours: int,
    max_patients: int | None,
) -> dict:
    static_path = _first_existing(
        processed_dir / "patient_static.parquet",
        processed_dir / "patient_static.csv",
    )
    ts_path = _first_existing(
        processed_dir / "patient_timeseries.parquet",
        processed_dir / "patient_timeseries.csv",
    )

    static = _read_table(static_path)
    ts = _read_table(ts_path)

    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce")
    ts["hr"] = pd.to_numeric(ts["hr"], errors="coerce")
    ts = ts.dropna(subset=["stay_id", "hr"]).copy()
    ts["stay_id"] = ts["stay_id"].astype(np.int64)

    stay_ids = np.sort(ts["stay_id"].unique())
    if max_patients is not None:
        stay_ids = stay_ids[: int(max_patients)]

    stay_set = set(stay_ids.tolist())
    ts = ts[ts["stay_id"].isin(stay_set)].copy()
    ts = ts[(ts["hr"] >= 0) & (ts["hr"] < n_hours)].copy()
    ts["hr"] = ts["hr"].astype(np.int64)
    ts = ts.sort_values(["stay_id", "hr"]).reset_index(drop=True)

    static["stay_id"] = pd.to_numeric(static["stay_id"], errors="coerce")
    static = static.dropna(subset=["stay_id"]).copy()
    static["stay_id"] = static["stay_id"].astype(np.int64)
    static = static[static["stay_id"].isin(stay_set)].drop_duplicates("stay_id")
    static = static.set_index("stay_id").reindex(stay_ids).reset_index()

    feature_cols = [col for col in MIMIC_CONTINUOUS_MAP if col in ts.columns]
    stay_to_index = {stay_id: idx for idx, stay_id in enumerate(stay_ids)}
    patient_index = ts["stay_id"].map(stay_to_index).to_numpy(dtype=np.int64)
    hour_index = ts["hr"].to_numpy(dtype=np.int64)

    continuous, masks_cont, interventions, masks_int, proxy, masks_proxy = _init_raw_arrays(
        raw_dir=raw_dir,
        n_patients=len(stay_ids),
        n_hours=n_hours,
    )

    for source_name in feature_cols:
        target_name = MIMIC_CONTINUOUS_MAP[source_name]
        target_idx = CONTINUOUS_INDEX[target_name]
        values = pd.to_numeric(ts[source_name], errors="coerce").astype(np.float32).to_numpy()
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        continuous[patient_index[valid], hour_index[valid], target_idx] = values[valid]
        masks_cont[patient_index[valid], hour_index[valid], target_idx] = 1.0

    _derive_proxy_indicators(continuous, proxy, masks_proxy)
    _flush_arrays(continuous, masks_cont, interventions, masks_int, proxy, masks_proxy)

    static_df = pd.DataFrame(
        {
            "patient_id": static["stay_id"].astype(str),
            "age": pd.to_numeric(static.get("age"), errors="coerce"),
            "sex": _encode_sex(static.get("gender")),
            "height_cm": np.nan,
            "weight_kg": np.nan,
            "icu_type": np.nan,
            "icu_los_hours": pd.to_numeric(static.get("los_icu_days"), errors="coerce") * 24.0,
            "mortality_inhospital": pd.to_numeric(static.get("hospital_expire_flag"), errors="coerce"),
            "mortality_source": np.where(
                static.get("hospital_expire_flag").notna(),
                "hospital_expire_flag",
                "unavailable",
            ),
            "center_id": "mimic_iv",
            "set_name": "mimic_iv_external",
            "data_source": "mimiciv",
            "sepsis_onset_hour": np.where(
                pd.to_numeric(static.get("is_sepsis3"), errors="coerce").fillna(0).astype(int).to_numpy() == 1,
                0.0,
                np.nan,
            ),
            "anchor_time_type": np.where(
                pd.to_numeric(static.get("is_sepsis3"), errors="coerce").fillna(0).astype(int).to_numpy() == 1,
                ANCHOR_SEPSIS_ONSET,
                ANCHOR_ICU_ADMISSION,
            ),
        }
    )
    _write_static(static_df, output_dir / "static.csv")

    mapped = sorted(MIMIC_CONTINUOUS_MAP.values())
    overall_missing = 1.0 - float(np.load(raw_dir / "masks_continuous.npy", mmap_mode="r").mean())
    return {
        "n_patients": int(len(stay_ids)),
        "n_hours": int(n_hours),
        "n_continuous": N_CONTINUOUS,
        "n_interventions": N_INTERVENTIONS,
        "n_proxy": N_PROXY,
        "overall_continuous_missing_rate": overall_missing,
        "has_outcome_labels": True,
        "prepared_source_format": static_path.suffix.lstrip("."),
        "source_feature_names": feature_cols,
        "mapped_continuous_features": mapped,
        "missing_continuous_features": [name for name in CONTINUOUS_NAMES if name not in mapped],
        "max_patients_applied": int(max_patients) if max_patients is not None else None,
        "output_static_path": str(output_dir / "static.csv"),
    }


def _prepare_eicu_raw_aligned(
    processed_dir: Path,
    output_dir: Path,
    raw_dir: Path,
    n_hours: int,
    max_patients: int | None,
) -> dict:
    tensor_path = _first_existing(
        processed_dir / "time_series_eicu_demo.npy",
        *_glob_candidates(processed_dir, "time_series_*.npy"),
    )
    info_path = _first_existing(
        processed_dir / "patient_info_eicu_demo.csv",
        *_glob_candidates(processed_dir, "patient_info_*.csv"),
    )
    feature_path = _first_existing(
        processed_dir / "feature_names_eicu_demo.json",
        *_glob_candidates(processed_dir, "feature_names_*.json"),
    )

    tensor = np.load(tensor_path, mmap_mode="r")
    patient_info = pd.read_csv(info_path)
    feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    if tensor.shape[1] < n_hours:
        raise ValueError(f"eICU tensor only has {tensor.shape[1]} hours, requested {n_hours}")

    n_patients = tensor.shape[0] if max_patients is None else min(int(max_patients), tensor.shape[0])
    tensor = tensor[:n_patients, :n_hours, :]
    patient_info = patient_info.iloc[:n_patients].copy().reset_index(drop=True)

    continuous, masks_cont, interventions, masks_int, proxy, masks_proxy = _init_raw_arrays(
        raw_dir=raw_dir,
        n_patients=n_patients,
        n_hours=n_hours,
    )

    mapped = []
    for source_name, target_name in EICU_CONTINUOUS_MAP.items():
        if source_name not in feature_to_idx:
            continue
        target_idx = CONTINUOUS_INDEX[target_name]
        source_idx = feature_to_idx[source_name]
        values = tensor[:, :, source_idx].astype(np.float32, copy=False)
        continuous[:, :, target_idx] = values
        masks_cont[:, :, target_idx] = np.isfinite(values).astype(np.float32)
        mapped.append(target_name)

    if "rrt" in feature_to_idx:
        rrt_values = tensor[:, :, feature_to_idx["rrt"]].astype(np.float32, copy=False)
        finite_rrt = np.isfinite(rrt_values)
        interventions[:, :, 1] = np.where(finite_rrt, rrt_values, np.nan)
        masks_int[:, :, 1] = finite_rrt.astype(np.float32)

    _derive_proxy_indicators(continuous, proxy, masks_proxy)
    _flush_arrays(continuous, masks_cont, interventions, masks_int, proxy, masks_proxy)

    hospital_ids = pd.to_numeric(patient_info.get("hospitalid"), errors="coerce")
    center_id = hospital_ids.round().astype("Int64").astype(str)
    center_id = center_id.where(hospital_ids.notna(), "eicu_unknown")
    center_id = center_id.map(lambda value: f"hospital_{value}")

    static_df = pd.DataFrame(
        {
            "patient_id": patient_info["patient_id"].astype(str),
            "age": pd.to_numeric(patient_info.get("age"), errors="coerce"),
            "sex": _encode_sex(patient_info.get("gender")),
            "height_cm": np.nan,
            "weight_kg": np.nan,
            "icu_type": patient_info.get("unittype"),
            "icu_los_hours": pd.to_numeric(patient_info.get("icu_los"), errors="coerce"),
            "mortality_inhospital": pd.to_numeric(patient_info.get("mortality_28d"), errors="coerce"),
            "mortality_source": patient_info.get("mortality_source", "hospitaldischargestatus_proxy"),
            "center_id": center_id,
            "set_name": "eicu_external",
            "data_source": "eicu",
            "sepsis_onset_hour": np.nan,
            "anchor_time_type": ANCHOR_ICU_ADMISSION,
        }
    )
    _write_static(static_df, output_dir / "static.csv")

    overall_missing = 1.0 - float(np.load(raw_dir / "masks_continuous.npy", mmap_mode="r").mean())
    mapped = sorted(set(mapped))
    return {
        "n_patients": int(n_patients),
        "n_hours": int(n_hours),
        "n_continuous": N_CONTINUOUS,
        "n_interventions": N_INTERVENTIONS,
        "n_proxy": N_PROXY,
        "overall_continuous_missing_rate": overall_missing,
        "has_outcome_labels": True,
        "prepared_source_format": "npy+csv",
        "source_feature_names": feature_names,
        "mapped_continuous_features": mapped,
        "missing_continuous_features": [name for name in CONTINUOUS_NAMES if name not in mapped],
        "max_patients_applied": int(max_patients) if max_patients is not None else None,
        "output_static_path": str(output_dir / "static.csv"),
    }


def _init_raw_arrays(raw_dir: Path, n_patients: int, n_hours: int):
    continuous = np.lib.format.open_memmap(
        raw_dir / "continuous.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_CONTINUOUS)
    )
    continuous[:] = np.nan

    masks_cont = np.lib.format.open_memmap(
        raw_dir / "masks_continuous.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_CONTINUOUS)
    )
    masks_cont[:] = 0.0

    interventions = np.lib.format.open_memmap(
        raw_dir / "interventions.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_INTERVENTIONS)
    )
    interventions[:] = np.nan

    masks_int = np.lib.format.open_memmap(
        raw_dir / "masks_interventions.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_INTERVENTIONS)
    )
    masks_int[:] = 0.0

    proxy = np.lib.format.open_memmap(
        raw_dir / "proxy_indicators.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_PROXY)
    )
    proxy[:] = 0.0

    masks_proxy = np.lib.format.open_memmap(
        raw_dir / "masks_proxy.npy", mode="w+", dtype=np.float32, shape=(n_patients, n_hours, N_PROXY)
    )
    masks_proxy[:] = 0.0
    return continuous, masks_cont, interventions, masks_int, proxy, masks_proxy


def _derive_proxy_indicators(
    continuous: np.memmap,
    proxy: np.memmap,
    masks_proxy: np.memmap,
) -> None:
    map_vals = continuous[:, :, CONTINUOUS_INDEX["map"]]
    map_valid = np.isfinite(map_vals)
    proxy[:, :, 0] = np.where(map_valid, (map_vals < 65).astype(np.float32), 0.0)
    masks_proxy[:, :, 0] = map_valid.astype(np.float32)

    fio2_vals = continuous[:, :, CONTINUOUS_INDEX["fio2"]]
    fio2_valid = np.isfinite(fio2_vals)
    proxy[:, :, 1] = np.where(fio2_valid, (fio2_vals > 0.21).astype(np.float32), 0.0)
    masks_proxy[:, :, 1] = fio2_valid.astype(np.float32)


def _write_static(static_df: pd.DataFrame, path: Path) -> None:
    for field in STATIC_FIELDS:
        if field not in static_df.columns:
            static_df[field] = np.nan
    static_df = static_df[STATIC_FIELDS]
    static_df.to_csv(path, index=False)


def _encode_sex(series: pd.Series | None) -> np.ndarray:
    if series is None:
        return np.full(0, np.nan)
    normalized = pd.Series(series).fillna("").astype(str).str.strip().str.lower()
    return np.where(
        normalized.str.startswith("m"),
        1.0,
        np.where(normalized.str.startswith("f"), 0.0, np.nan),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _first_existing(*paths: Path) -> Path:
    candidates = [path for path in paths if path is not None]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No matching file found:\n" + "\n".join(str(path) for path in candidates))


def _glob_candidates(parent: Path, pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(parent.glob(pattern)))


def _flush_arrays(*arrays) -> None:
    for arr in arrays:
        arr.flush()


def _manifest_notes_for_source(source: str, reference_stats_path: Path | None) -> list[str]:
    notes = [
        "This manifest records the exact external temporal transfer configuration and generated artifacts.",
        "No V1 clustering metrics are propagated into this S0 bundle.",
        "Continuous channels are aligned to the 21-feature S0 schema used by the PhysioNet 2012 S1.5 encoder.",
    ]
    if reference_stats_path is not None:
        notes.append(
            f"Preprocessing reuses reference statistics from {reference_stats_path} to keep external inputs numerically aligned."
        )
    if source == "mimic":
        notes.append(
            "MIMIC-IV rows reuse the legacy prepared analysis tables; Sepsis-3 stays are anchored at sepsis onset, others at ICU admission."
        )
        notes.append(
            "Unsupported S0 channels in the prepared MIMIC tables remain structurally missing rather than being synthetically derived."
        )
    elif source == "eicu":
        notes.append(
            "The cached eICU tensor provides 17 channels; unsupported S0 channels remain structurally missing."
        )
        notes.append(
            "eICU mortality_inhospital is populated from the locally cached hospital discharge status proxy label."
        )
    return notes
