"""
test_s15_finetune_supervised.py - Smoke test for end-to-end supervised fine-tuning.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s15.finetune_supervised import train_end_to_end_classifier


def test_end_to_end_classifier_learns_synthetic_signal():
    rng = np.random.default_rng(42)
    n_samples = 120
    n_hours = 48
    n_features = 21

    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=int)
    rng.shuffle(labels)

    continuous = rng.normal(scale=0.15, size=(n_samples, n_hours, n_features)).astype(np.float32)
    masks = np.ones_like(continuous, dtype=np.float32)

    continuous[labels == 1, :24, 0] += 2.0
    continuous[labels == 1, 24:, 1] += 1.0
    continuous[labels == 0, :24, 0] -= 2.0
    continuous[labels == 0, 24:, 1] -= 1.0

    train_idx = np.arange(0, 80)
    val_idx = np.arange(80, 100)
    test_idx = np.arange(100, 120)

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = Path(tmp_dir) / "synthetic_s0"
        proc_dir = data_dir / "processed"
        proc_dir.mkdir(parents=True)

        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks)

        static = pd.DataFrame({
            "mortality_inhospital": labels,
            "center_id": ["center_a"] * n_samples,
        })
        static.to_csv(data_dir / "static.csv", index=False)

        with open(data_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump({
                "train": train_idx.tolist(),
                "val": val_idx.tolist(),
                "test": test_idx.tolist(),
            }, f)

        report = train_end_to_end_classifier(
            s0_dir=data_dir,
            output_dir=Path(tmp_dir) / "out",
            pretrained_checkpoint=None,
            batch_size=16,
            epochs=6,
            lr_encoder=1.0e-3,
            lr_head=2.0e-3,
            patience=3,
            threshold_metric="accuracy",
            monitor_metric="accuracy",
            device="cpu",
            n_features=n_features,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            head_hidden_dim=32,
            head_dropout=0.1,
            encoder_dropout=0.1,
        )

        assert report["main_task"]["splits"]["val"]["accuracy"] >= 0.85
        assert report["main_task"]["splits"]["test"]["accuracy"] >= 0.85
        assert (Path(tmp_dir) / "out" / "finetune_report.json").exists()


if __name__ == "__main__":
    test_end_to_end_classifier_learns_synthetic_signal()
    print("1 passed, 0 failed")
