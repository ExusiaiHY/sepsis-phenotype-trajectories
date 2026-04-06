from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s6.multitask_model import train_multitask_student


def _build_synthetic_multitask_dataset(data_dir: Path, n_samples: int = 40) -> None:
    rng = np.random.default_rng(123)
    n_steps = 8
    n_cont = 5
    n_treat = 1

    mortality = np.array([0, 1] * (n_samples // 2), dtype=np.int64)
    rng.shuffle(mortality)

    time_series = rng.normal(0.0, 0.25, size=(n_samples, n_steps, n_cont + n_treat)).astype(np.float32)
    time_series[mortality == 1, :, 0] += 1.0
    time_series[mortality == 1, :, 1] += 0.6
    time_series[:, :, -1] = 0.0
    time_series[mortality == 1, :4, -1] = 1.0

    immune_labels = np.arange(n_samples, dtype=np.int64) % 4
    organ_labels = np.arange(n_samples, dtype=np.int64) % 3
    fluid_labels = np.where(mortality == 1, 2, 1).astype(np.int64)

    patient_info = pd.DataFrame(
        {
            "stay_id": np.arange(1, n_samples + 1, dtype=np.int64),
            "mortality_28d": mortality,
            "immune_subtype_label": immune_labels,
            "organ_subtype_label": organ_labels,
            "fluid_benefit_label": fluid_labels,
        }
    )

    np.save(data_dir / "time_series_enhanced.npy", time_series)
    patient_info.to_csv(data_dir / "patient_info_enhanced.csv", index=False)


def test_train_multitask_student_supports_masked_npz_targets(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()

    _build_synthetic_multitask_dataset(data_dir)

    report = train_multitask_student(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=8,
        epochs=2,
        patience=1,
        lr=1.0e-3,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=7,
        device="cpu",
        student_arch="tcn",
        student_d_model=16,
        lambda_mortality=1.0,
        lambda_immune=1.0,
        lambda_organ=0.5,
        lambda_fluid=0.5,
    )

    saved_report = json.loads((output_dir / "multitask_student_report.json").read_text(encoding="utf-8"))

    assert report["model"]["student_arch"] == "tcn"
    assert report["training"]["lambda_mortality"] == 1.0
    assert report["training"]["lambda_immune"] == 1.0
    assert report["training"]["lambda_organ"] == 0.5
    assert report["training"]["lambda_fluid"] == 0.5
    assert len(report["history"]) >= 1
    assert report["deployment"]["float_n_parameters"] > 0

    test_split = report["splits"]["test"]
    assert "mortality" in test_split
    assert "immune" in test_split
    assert "organ" in test_split
    assert "fluid" in test_split

    assert (output_dir / "multitask_student_report.json").exists()
    assert (output_dir / "multitask_student.pt").exists()
    assert (output_dir / "checkpoints" / "student_best.pt").exists()
