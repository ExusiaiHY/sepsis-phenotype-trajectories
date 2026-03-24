"""
test_s5_realtime.py - Stage 5 realtime student, buffer, and dashboard tests.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s5.dashboard import render_clinical_dashboard_html
from s5.realtime_model import (
    RealtimePatientBuffer,
    RealtimePhenotypeMonitor,
    RealtimeStudentClassifier,
    distill_realtime_student,
)


def test_realtime_student_and_dashboard():
    rng = np.random.default_rng(7)
    n_samples = 100
    n_hours = 48
    n_cont = 21
    n_treat = 7

    labels = np.array([0] * 50 + [1] * 50, dtype=int)
    rng.shuffle(labels)
    continuous = rng.normal(scale=0.2, size=(n_samples, n_hours, n_cont)).astype(np.float32)
    masks_cont = np.ones_like(continuous, dtype=np.float32)
    treatments = np.zeros((n_samples, n_hours, n_treat), dtype=np.float32)
    masks_treat = np.ones_like(treatments, dtype=np.float32)
    teacher_embeddings = rng.normal(size=(n_samples, 24)).astype(np.float32)

    continuous[labels == 1, :, 0] += 1.2
    treatments[labels == 1, :8, 0] = 1.0
    treatments[labels == 1, :8, 3] = 250.0
    teacher_embeddings[labels == 1, 0] += 2.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        s0_dir = root / "s0"
        proc_dir = s0_dir / "processed"
        treatment_dir = root / "treatment"
        proc_dir.mkdir(parents=True)
        treatment_dir.mkdir(parents=True)
        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks_cont)
        np.save(treatment_dir / "treatments.npy", treatments)
        np.save(treatment_dir / "masks_treatments.npy", masks_treat)
        teacher_path = root / "teacher_embeddings.npy"
        np.save(teacher_path, teacher_embeddings)

        pd.DataFrame({"mortality_inhospital": labels}).to_csv(s0_dir / "static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": list(range(0, 70)),
                    "val": list(range(70, 85)),
                    "test": list(range(85, 100)),
                },
                f,
            )

        report = distill_realtime_student(
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "out",
            teacher_embeddings_path=teacher_path,
            batch_size=16,
            epochs=4,
            lr=1.0e-3,
            patience=2,
            threshold_metric="accuracy",
            device="cpu",
            student_d_model=24,
            teacher_dim=24,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            dropout=0.1,
            head_hidden_dim=24,
            head_dropout=0.1,
        )
        assert report["deployment"]["cpu_latency_ms_per_sample"] > 0.0
        assert report["splits"]["test"]["auroc"] >= 0.6

        model = RealtimeStudentClassifier(
            n_cont_features=n_cont,
            n_treat_features=n_treat,
            student_d_model=24,
            teacher_dim=24,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            dropout=0.1,
            head_hidden_dim=24,
            head_dropout=0.1,
        )
        state = torch.load(root / "out" / "realtime_student.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])

        buffer = RealtimePatientBuffer(seq_len=n_hours, n_cont_features=n_cont, n_treat_features=n_treat)
        for hour in range(n_hours):
            buffer.update(
                values=continuous[0, hour],
                mask=masks_cont[0, hour],
                treatments=treatments[0, hour],
                treatment_mask=masks_treat[0, hour],
            )
        monitor = RealtimePhenotypeMonitor(
            model=model,
            threshold=float(state["threshold"]),
            phenotype_centroids=np.zeros((4, 24), dtype=np.float32),
            device="cpu",
        )
        snapshot = monitor.predict(buffer)
        assert snapshot is not None
        assert "risk_probability" in snapshot

        html_path = root / "dashboard.html"
        html = render_clinical_dashboard_html(
            patient_id="demo-icu-01",
            snapshots=[snapshot, {**snapshot, "risk_probability": min(0.99, snapshot["risk_probability"] + 0.1)}],
            output_path=html_path,
            model_meta=report["model"],
        )
        assert html_path.exists()
        assert "demo-icu-01" in html


if __name__ == "__main__":
    test_realtime_student_and_dashboard()
    print("1 passed, 0 failed")
