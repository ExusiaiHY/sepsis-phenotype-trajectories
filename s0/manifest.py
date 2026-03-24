"""
manifest.py - Generate data_manifest.json for full provenance.

Purpose:
  Record exactly what data was processed, how, and when, so any result
  can be traced back to its input data and preprocessing decisions.

Connects to:
  - schema.py for variable dictionary
  - scripts/s0_prepare.py calls generate_manifest()

Expected output artifacts:
  data/s0/data_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from s0.schema import schema_to_feature_dict


def generate_manifest(
    s0_dir: Path,
    extraction_stats: dict,
    preprocess_stats: dict,
    config: dict,
) -> dict:
    """
    Generate and save data_manifest.json.

    NOTE: No performance metrics are included in the manifest.
    The Sepsis 2019 silhouette inconsistency is unresolved pending rerun;
    no V1 metric values are propagated here.
    """
    s0_dir = Path(s0_dir)

    # File checksums
    checksums = {}
    for subdir in ["raw_aligned", "processed"]:
        dirpath = s0_dir / subdir
        if dirpath.exists():
            for fpath in sorted(dirpath.iterdir()):
                if fpath.suffix in (".npy", ".json"):
                    checksums[f"{subdir}/{fpath.name}"] = _sha256_file(fpath)

    static_path = s0_dir / "static.csv"
    if static_path.exists():
        checksums["static.csv"] = _sha256_file(static_path)

    # Cohort summary
    cohort_summary = {}
    if static_path.exists():
        static = pd.read_csv(static_path)
        cohort_summary = {
            "n_patients": len(static),
            "centers": static["center_id"].value_counts().to_dict(),
            "data_sources": static["data_source"].value_counts().to_dict(),
            "mortality_source": static["mortality_source"].value_counts().to_dict() if "mortality_source" in static.columns else {},
            "age_mean": float(static["age"].mean()) if "age" in static.columns else None,
            "sex_male_ratio": float((static["sex"] == 1).mean()) if "sex" in static.columns else None,
            "anchor_time_types": static["anchor_time_type"].value_counts().to_dict() if "anchor_time_type" in static.columns else {},
        }
        # Mortality rate (only if not all NaN)
        if "mortality_inhospital" in static.columns:
            mort = static["mortality_inhospital"]
            if not mort.isna().all():
                cohort_summary["mortality_rate"] = float(mort.mean())
                cohort_summary["mortality_n_available"] = int(mort.notna().sum())
            else:
                cohort_summary["mortality_rate"] = None
                cohort_summary["mortality_n_available"] = 0

    notes = config.get("manifest", {}).get("notes")
    if notes is None:
        notes = [
            "This manifest records the exact data pipeline configuration and outputs.",
            "No V1 performance metrics are included.",
            "V1 clustering quality metrics are unresolved pending rerun and are NOT referenced here.",
        ]
        source = str(config.get("data", {}).get("source", ""))
        if source == "physionet2012":
            notes.extend(
                [
                    "Intervention channels (antibiotics_on, rrt_on) are unavailable for PhysioNet 2012.",
                    "vasopressor_proxy and mechvent_proxy are PROXY indicators, not true treatment records.",
                    "sepsis_onset_hour is NaN for PhysioNet 2012; anchor_time_type is icu_admission.",
                ]
            )

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "schema": schema_to_feature_dict(),
        "config": config,
        "extraction_stats": extraction_stats,
        "preprocess_stats": _filter_stats(preprocess_stats),
        "cohort_summary": cohort_summary,
        "file_checksums": checksums,
        "notes": notes,
    }

    manifest_path = s0_dir / "data_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _filter_stats(stats: dict) -> dict:
    """Remove large arrays from stats for JSON serialization."""
    filtered = {}
    for k, v in stats.items():
        if isinstance(v, (list, tuple)) and len(v) > 100:
            filtered[k] = f"[array of {len(v)} elements, omitted]"
        elif isinstance(v, np.ndarray):
            filtered[k] = f"[ndarray shape {v.shape}, omitted]"
        else:
            filtered[k] = v
    return filtered
