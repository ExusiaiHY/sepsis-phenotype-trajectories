"""
test_s15_advanced_classifier.py - Unit tests for advanced downstream classifier.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s15.advanced_classifier import build_feature_views, train_advanced_mortality_classifier


def test_build_feature_views_and_train_hgb():
    rng = np.random.default_rng(7)
    n_samples = 40
    n_hours = 48
    n_features = 21

    labels = np.array([0] * 20 + [1] * 20)
    continuous = rng.normal(size=(n_samples, n_hours, n_features)).astype(np.float32)
    continuous[labels == 1, :, 0] += 2.0
    masks = rng.binomial(1, 0.6, size=(n_samples, n_hours, n_features)).astype(np.float32)
    proxy = np.zeros((n_samples, n_hours, 2), dtype=np.float32)
    proxy[labels == 1, :, 0] = 1.0
    interventions = np.zeros((n_samples, n_hours, 2), dtype=np.float32)
    embeddings = np.column_stack([
        labels * 3.0 + rng.normal(scale=0.2, size=n_samples),
        rng.normal(size=n_samples),
    ]).astype(np.float32)

    splits = {
        "train": list(range(0, 14)) + list(range(20, 34)),
        "val": list(range(14, 17)) + list(range(34, 37)),
        "test": list(range(17, 20)) + list(range(37, 40)),
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        s0_dir = root / "s0"
        proc_dir = s0_dir / "processed"
        proc_dir.mkdir(parents=True)

        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks)
        np.save(proc_dir / "proxy_indicators.npy", proxy)
        np.save(proc_dir / "interventions.npy", interventions)
        np.save(proc_dir / "masks_interventions.npy", interventions)
        np.save(proc_dir / "masks_proxy.npy", proxy)

        static = pd.DataFrame({
            "patient_id": np.arange(n_samples),
            "age": 60 + labels * 5,
            "sex": ["male" if i % 2 == 0 else "female" for i in range(n_samples)],
            "height_cm": 170 + rng.normal(size=n_samples),
            "weight_kg": 70 + rng.normal(size=n_samples),
            "icu_type": np.where(labels == 1, 2, 1),
            "icu_los_hours": 48,
            "mortality_inhospital": labels,
            "mortality_source": "outcomes_file",
            "center_id": "center_a",
            "set_name": "set-a",
            "data_source": "physionet2012",
            "sepsis_onset_hour": 0,
            "anchor_time_type": "icu_admission",
        })
        static.to_csv(s0_dir / "static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(splits, f)

        emb_path = root / "embeddings.npy"
        np.save(emb_path, embeddings)

        bundle = build_feature_views(s0_dir=s0_dir, embeddings_path=emb_path)
        assert "stats_mask_proxy_static" in bundle["views"]
        assert "fused_all" in bundle["views"]
        assert bundle["views"]["fused_all"].shape[0] == n_samples

        report = train_advanced_mortality_classifier(
            s0_dir=s0_dir,
            splits_path=s0_dir / "splits.json",
            output_dir=root / "out",
            embeddings_path=emb_path,
            model_type="hgb",
            feature_set="stats_mask_proxy_static",
            threshold_metric="balanced_accuracy",
            hgb_max_iter=80,
        )

        assert report["splits"]["val"]["balanced_accuracy"] >= 0.5
        assert report["splits"]["test"]["balanced_accuracy"] >= 0.5
        assert report["splits"]["test"]["auroc"] >= 0.5
        assert (root / "out" / "advanced_mortality_classifier_report.json").exists()


if __name__ == "__main__":
    test_build_feature_views_and_train_hgb()
    print("1 passed, 0 failed")
