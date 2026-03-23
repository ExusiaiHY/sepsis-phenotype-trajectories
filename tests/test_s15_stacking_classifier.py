"""
test_s15_stacking_classifier.py - Unit tests for OOF stacking downstream classifier.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s15.stacking_classifier import train_stacking_mortality_classifier
from s15.stacking_validation import validate_stacking_classifier


def test_train_and_validate_stacking_classifier():
    rng = np.random.default_rng(11)
    n_samples = 60
    n_hours = 48
    n_features = 21

    labels = np.array([0] * 30 + [1] * 30)
    continuous = rng.normal(size=(n_samples, n_hours, n_features)).astype(np.float32)
    continuous[labels == 1, :, 0] += 1.5
    continuous[labels == 1, :, 1] += 0.8
    masks = rng.binomial(1, 0.7, size=(n_samples, n_hours, n_features)).astype(np.float32)
    proxy = np.zeros((n_samples, n_hours, 2), dtype=np.float32)
    proxy[labels == 1, :, 0] = 1.0
    embeddings = np.column_stack(
        [
            labels * 2.5 + rng.normal(scale=0.25, size=n_samples),
            labels * 0.9 + rng.normal(scale=0.3, size=n_samples),
            rng.normal(size=n_samples),
        ]
    ).astype(np.float32)

    splits = {
        "train": list(range(0, 22)) + list(range(30, 52)),
        "val": list(range(22, 26)) + list(range(52, 56)),
        "test": list(range(26, 30)) + list(range(56, 60)),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        s0_dir = root / "s0"
        proc_dir = s0_dir / "processed"
        proc_dir.mkdir(parents=True)

        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks)
        np.save(proc_dir / "proxy_indicators.npy", proxy)

        static = pd.DataFrame(
            {
                "patient_id": np.arange(n_samples),
                "age": 58 + labels * 7,
                "sex": ["male" if i % 2 == 0 else "female" for i in range(n_samples)],
                "height_cm": 170 + rng.normal(size=n_samples),
                "weight_kg": 72 + labels * 2 + rng.normal(size=n_samples),
                "icu_type": np.where(labels == 1, 2, 1),
                "icu_los_hours": 48,
                "mortality_inhospital": labels,
                "mortality_source": "outcomes_file",
                "center_id": "center_a",
                "set_name": "set-a",
                "data_source": "physionet2012",
                "sepsis_onset_hour": 0,
                "anchor_time_type": "icu_admission",
            }
        )
        static.to_csv(s0_dir / "static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(splits, f)

        emb_path = root / "embeddings.npy"
        np.save(emb_path, embeddings)

        out_dir = root / "stacking_out"
        train_report = train_stacking_mortality_classifier(
            s0_dir=s0_dir,
            splits_path=s0_dir / "splits.json",
            output_dir=out_dir,
            embeddings_path=emb_path,
            threshold_metric="accuracy",
            n_splits=3,
        )

        assert train_report["splits"]["test"]["accuracy"] >= 0.5
        assert train_report["operating_points"]["balanced_accuracy"]["test"]["balanced_accuracy"] >= 0.5

        validation_report = validate_stacking_classifier(
            model_path=out_dir / "stacking_mortality_classifier.pkl",
            s0_dir=s0_dir,
            splits_path=s0_dir / "splits.json",
            output_dir=out_dir,
            embeddings_path=emb_path,
            n_bootstrap=30,
            permutation_repeats=3,
        )

        assert validation_report["splits"]["test"]["accuracy"] >= 0.5
        assert validation_report["test_calibration"]["brier"] >= 0.0
        assert len(validation_report["meta_feature_importance"]["ranked_features"]) == 3
        assert (out_dir / "stacking_mortality_classifier_report.json").exists()
        assert (out_dir / "stacking_validation_report.json").exists()


if __name__ == "__main__":
    test_train_and_validate_stacking_classifier()
    print("1 passed, 0 failed")
