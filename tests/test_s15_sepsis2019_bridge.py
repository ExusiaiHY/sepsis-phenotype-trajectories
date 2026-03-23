"""
test_s15_sepsis2019_bridge.py - Unit tests for Sepsis 2019 bridge utilities.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s15.sepsis2019_bridge import (
    DEFAULT_SEPSIS2019_FEATURES,
    build_bridge_bundle,
    write_bridge_dataset,
)


def test_bridge_bundle_and_write_roundtrip():
    n_patients = 20
    n_hours = 48
    n_features = len(DEFAULT_SEPSIS2019_FEATURES)
    feature_index = {name: idx for idx, name in enumerate(DEFAULT_SEPSIS2019_FEATURES)}

    series = np.full((n_patients, n_hours, n_features), np.nan, dtype=np.float32)
    series[:, :, feature_index["heart_rate"]] = 80.0
    series[:, :, feature_index["map"]] = 72.0
    series[:, :, feature_index["fio2"]] = 0.25
    series[::2, :, feature_index["map"]] = 55.0
    series[1::2, :, feature_index["fio2"]] = 0.18

    patient_info = pd.DataFrame({
        "patient_id": [f"S19_{idx}" for idx in range(n_patients)],
        "age": np.linspace(40, 75, n_patients),
        "gender": [idx % 2 for idx in range(n_patients)],
        "icu_los": np.full(n_patients, 48.0),
        "mortality_28d": [idx % 2 for idx in range(n_patients)],
        "sepsis_label": [idx % 2 for idx in range(n_patients)],
        "actual_timesteps": np.full(n_patients, 48),
    })

    bundle = build_bridge_bundle(series, patient_info)
    assert bundle["continuous"].shape == (n_patients, n_hours, 21)
    assert bundle["proxy_indicators"].shape == (n_patients, n_hours, 2)
    assert bundle["static"]["sepsis_label"].mean() == 0.5
    assert "heart_rate" in bundle["report"]["mapped_features"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "s19"
        report = write_bridge_dataset(output_dir, bundle, split_method="random", stratify_by="sepsis_label")
        assert (output_dir / "raw_aligned" / "continuous.npy").exists()
        assert (output_dir / "processed" / "continuous.npy").exists()
        assert (output_dir / "static.csv").exists()
        assert (output_dir / "splits.json").exists()
        assert report["n_patients"] == n_patients

        with open(output_dir / "splits.json", encoding="utf-8") as f:
            splits = json.load(f)
        assert set(splits.keys()) >= {"train", "val", "test"}


if __name__ == "__main__":
    test_bridge_bundle_and_write_roundtrip()
    print("1 passed, 0 failed")
