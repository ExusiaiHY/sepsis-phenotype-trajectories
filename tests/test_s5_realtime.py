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

from s15.classification_eval import _select_threshold
from s5.dashboard import render_clinical_dashboard_html
from s5.deployment_policy import (
    build_policy_grid,
    evaluate_policy_grid,
    load_policy_artifact,
    load_replay_bundle,
    select_best_policy,
    simulate_alert_policy,
    write_policy_artifacts,
)
from s5.realtime_model import (
    RealtimePatientBuffer,
    RealtimePhenotypeMonitor,
    RealtimeStudentClassifier,
    distill_realtime_student,
)
from s5.silent_deployment import run_silent_deployment_replay


class _DummyRealtimeRiskModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        treatments: torch.Tensor,
        treatment_mask: torch.Tensor,
        note_embeddings: torch.Tensor | None = None,
    ) -> dict:
        logits = x[:, -1, 0]
        teacher_embedding = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
        return {
            "logits": logits,
            "teacher_embedding": teacher_embedding,
        }


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
            apply_temperature_scaling=True,
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
        assert report["training"]["apply_temperature_scaling"] is True
        assert report["posthoc_calibration"]["method"] == "temperature_scaling"

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
        assert float(state["temperature"]) > 0.0

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
            temperature=float(state["temperature"]),
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


def test_realtime_student_v2_tcn_and_alert_rate_threshold():
    rng = np.random.default_rng(11)
    n_samples = 96
    n_hours = 48
    n_cont = 21
    n_treat = 7

    labels = np.array([0] * 48 + [1] * 48, dtype=int)
    rng.shuffle(labels)
    continuous = rng.normal(scale=0.25, size=(n_samples, n_hours, n_cont)).astype(np.float32)
    masks_cont = np.ones_like(continuous, dtype=np.float32)
    treatments = rng.normal(scale=0.1, size=(n_samples, n_hours, n_treat)).astype(np.float32)
    masks_treat = np.ones_like(treatments, dtype=np.float32)
    teacher_embeddings = rng.normal(size=(n_samples, 24)).astype(np.float32)
    teacher_probabilities = np.where(labels == 1, 0.82, 0.18).astype(np.float32)

    continuous[labels == 1, :, 0] += 1.4
    continuous[labels == 1, :, 1] += 0.8
    treatments[labels == 1, :10, 0] += 1.0
    treatments[labels == 1, :10, 3] += 180.0
    teacher_embeddings[labels == 1, 0] += 2.4
    teacher_embeddings[labels == 1, 1] += 1.0

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
        teacher_prob_path = root / "teacher_probabilities.npy"
        np.save(teacher_prob_path, teacher_probabilities)

        pd.DataFrame({"mortality_inhospital": labels}).to_csv(s0_dir / "static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": list(range(0, 64)),
                    "val": list(range(64, 80)),
                    "test": list(range(80, 96)),
                },
                f,
            )

        report = distill_realtime_student(
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "out_tcn",
            teacher_embeddings_path=teacher_path,
            teacher_probabilities_path=teacher_prob_path,
            batch_size=16,
            epochs=4,
            lr=1.0e-3,
            patience=2,
            bce_weight=1.0,
            distill_weight=1.0,
            distill_cosine_weight=0.25,
            distill_prob_weight=0.2,
            distill_temperature=2.0,
            threshold_metric="predicted_positive_rate",
            target_positive_rate=0.35,
            device="cpu",
            student_arch="tcn",
            student_d_model=24,
            teacher_dim=24,
            head_hidden_dim=24,
            head_dropout=0.1,
            tcn_kernel_size=3,
            tcn_dilations=(1, 2, 4),
        )

        assert report["model"]["student_arch"] == "tcn"
        assert report["training"]["distill_cosine_weight"] == 0.25
        assert report["training"]["distill_prob_weight"] == 0.2
        assert abs(report["splits"]["val"]["predicted_positive_rate"] - 0.35) <= 0.15
        assert report["splits"]["test"]["auroc"] >= 0.6

        model = RealtimeStudentClassifier(
            n_cont_features=n_cont,
            n_treat_features=n_treat,
            student_arch="tcn",
            student_d_model=24,
            teacher_dim=24,
            head_hidden_dim=24,
            head_dropout=0.1,
            tcn_kernel_size=3,
            tcn_dilations=(1, 2, 4),
        )
        state = torch.load(root / "out_tcn" / "realtime_student.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        with torch.no_grad():
            out = model(
                torch.from_numpy(continuous[:2]),
                torch.from_numpy(masks_cont[:2]),
                torch.from_numpy(treatments[:2]),
                torch.from_numpy(masks_treat[:2]),
            )
        assert out["logits"].shape[0] == 2


def test_realtime_student_supports_checkpoint_warm_start():
    rng = np.random.default_rng(17)
    n_samples = 48
    n_hours = 24
    n_cont = 6
    n_treat = 3

    labels = np.array([0] * 24 + [1] * 24, dtype=int)
    rng.shuffle(labels)
    continuous = rng.normal(scale=0.2, size=(n_samples, n_hours, n_cont)).astype(np.float32)
    masks_cont = np.ones_like(continuous, dtype=np.float32)
    treatments = rng.normal(scale=0.1, size=(n_samples, n_hours, n_treat)).astype(np.float32)
    masks_treat = np.ones_like(treatments, dtype=np.float32)
    teacher_embeddings = rng.normal(size=(n_samples, 12)).astype(np.float32)

    continuous[labels == 1, :, 0] += 0.8
    teacher_embeddings[labels == 1, 0] += 1.3

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
                    "train": list(range(0, 32)),
                    "val": list(range(32, 40)),
                    "test": list(range(40, 48)),
                },
                f,
            )

        init_model = RealtimeStudentClassifier(
            n_cont_features=n_cont,
            n_treat_features=n_treat,
            student_arch="transformer",
            student_d_model=12,
            teacher_dim=12,
            n_heads=4,
            n_layers=1,
            d_ff=24,
            dropout=0.1,
            head_hidden_dim=12,
            head_dropout=0.1,
            max_seq_len=n_hours,
        )
        with torch.no_grad():
            for param in init_model.parameters():
                if param.ndim > 1:
                    param.fill_(0.125)
                else:
                    param.fill_(0.025)
        init_checkpoint = root / "warm_start.pt"
        torch.save(
            {
                "model_state_dict": init_model.state_dict(),
                "config": {
                    "student_arch": "transformer",
                },
            },
            init_checkpoint,
        )

        report = distill_realtime_student(
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "warm_started_out",
            init_checkpoint_path=init_checkpoint,
            teacher_embeddings_path=teacher_path,
            batch_size=16,
            epochs=0,
            lr=1.0e-3,
            patience=1,
            threshold_metric="accuracy",
            device="cpu",
            student_arch="transformer",
            student_d_model=12,
            teacher_dim=12,
            n_heads=4,
            n_layers=1,
            d_ff=24,
            dropout=0.1,
            head_hidden_dim=12,
            head_dropout=0.1,
        )

        saved_state = torch.load(root / "warm_started_out" / "realtime_student.pt", map_location="cpu", weights_only=False)
        assert report["training"]["initialization"]["mode"] == "checkpoint"
        assert report["training"]["initialization"]["loaded_tensors"] > 0
        assert report["training"]["epochs_trained"] == 0
        assert torch.allclose(
            saved_state["model_state_dict"]["head.0.weight"],
            init_model.state_dict()["head.0.weight"],
        )


def test_threshold_selection_supports_alert_budget():
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    probs = np.array([0.05, 0.15, 0.35, 0.55, 0.75, 0.95], dtype=np.float32)
    threshold, search = _select_threshold(
        y_true,
        probs,
        metric_name="predicted_positive_rate",
        target_positive_rate=0.5,
    )

    assert 0.05 <= threshold <= 0.95
    best_entry = min(search, key=lambda item: item["distance_to_target"])
    assert abs(best_entry["predicted_positive_rate"] - 0.5) <= 0.17


def test_silent_deployment_replay_outputs_summary_and_dashboard():
    rng = np.random.default_rng(23)
    n_samples = 80
    n_hours = 12
    n_cont = 8
    n_treat = 4

    labels = np.array([0] * 40 + [1] * 40, dtype=int)
    rng.shuffle(labels)
    continuous = rng.normal(scale=0.2, size=(n_samples, n_hours, n_cont)).astype(np.float32)
    masks_cont = np.ones_like(continuous, dtype=np.float32)
    treatments = rng.normal(scale=0.1, size=(n_samples, n_hours, n_treat)).astype(np.float32)
    masks_treat = np.ones_like(treatments, dtype=np.float32)
    teacher_embeddings = rng.normal(size=(n_samples, 16)).astype(np.float32)

    continuous[labels == 1, :, 0] += 1.1
    continuous[labels == 1, :, 1] += 0.6
    treatments[labels == 1, :5, 0] += 1.0
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
        (treatment_dir / "treatment_feature_names.json").write_text(
            json.dumps(["vasopressor_on", "antibiotic_on", "fluid_ml", "rrt_on"]),
            encoding="utf-8",
        )
        teacher_path = root / "teacher_embeddings.npy"
        np.save(teacher_path, teacher_embeddings)

        pd.DataFrame(
            {
                "patient_id": np.arange(n_samples).astype(str),
                "mortality_inhospital": labels,
                "icu_los_hours": np.full(n_samples, float(n_hours)),
                "data_source": ["synthetic"] * n_samples,
                "center_id": ["test_center"] * n_samples,
            }
        ).to_csv(s0_dir / "static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": list(range(0, 56)),
                    "val": list(range(56, 68)),
                    "test": list(range(68, 80)),
                },
                f,
            )

        distill_realtime_student(
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "student_out",
            teacher_embeddings_path=teacher_path,
            batch_size=16,
            epochs=3,
            lr=1.0e-3,
            patience=2,
            apply_temperature_scaling=True,
            threshold_metric="balanced_accuracy",
            device="cpu",
            student_d_model=16,
            teacher_dim=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.1,
            head_hidden_dim=16,
            head_dropout=0.1,
        )

        summary = run_silent_deployment_replay(
            model_artifact_path=root / "student_out" / "realtime_student.pt",
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "silent_out",
            split="test",
            min_history_hours=3,
            landmark_hours=(3, 6, 12),
            batch_size=8,
            device="cpu",
            save_replay_bundle=True,
        )

        assert summary["n_patients"] == 12
        assert summary["sample_patient_id"] is not None
        assert (root / "silent_out" / "silent_deployment_summary.json").exists()
        assert (root / "silent_out" / "patient_summary.csv").exists()
        assert (root / "silent_out" / "landmark_metrics.csv").exists()
        assert (root / "silent_out" / "cumulative_alert_metrics.csv").exists()
        assert (root / "silent_out" / "sample_patient_dashboard.html").exists()
        assert (root / "silent_out" / "sample_patient_snapshots.json").exists()
        assert (root / "silent_out" / "replay_bundle.npz").exists()
        assert len(summary["landmark_metrics"]) == 3
        assert 0.0 <= summary["patient_alert_rate"] <= 1.0
        bundle = load_replay_bundle(root / "silent_out" / "replay_bundle.npz")
        assert bundle["risk_matrix"].shape == (12, n_hours)
        assert bundle["labels"].shape == (12,)
        assert bundle["patient_ids"].shape == (12,)
        assert bundle["split"] == "test"
        assert int(bundle["min_history_hours"]) == 3

        policy_path = root / "synthetic_policy.json"
        policy_path.write_text(
            json.dumps(
                {
                    "source": "synthetic_test",
                    "constraints": {"max_negative_patient_alert_rate": 0.4},
                    "best_policy": {
                        "policy_name": "synthetic_tight",
                        "enter_threshold": 0.2,
                        "exit_threshold": 0.2,
                        "min_history_hours": 4,
                        "min_consecutive_hours": 2,
                        "refractory_hours": 3,
                        "max_alerts_per_stay": 1,
                    },
                }
            ),
            encoding="utf-8",
        )

        policy_summary = run_silent_deployment_replay(
            model_artifact_path=root / "student_out" / "realtime_student.pt",
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "silent_policy_out",
            policy_path=policy_path,
            split="test",
            min_history_hours=3,
            landmark_hours=(3, 6, 12),
            batch_size=8,
            device="cpu",
        )

        assert policy_summary["deployment_mode"] == "policy"
        assert policy_summary["policy"] is not None
        assert policy_summary["policy"]["policy"]["policy_name"] == "synthetic_tight"
        assert policy_summary["threshold"] == 0.2
        assert "model_threshold" in policy_summary
        assert policy_summary["alert_events_per_patient_day"] <= policy_summary["alert_state_hours_per_patient_day"]
        assert (root / "silent_policy_out" / "silent_deployment_summary.json").exists()


def test_deployment_policy_state_machine_controls_repeat_alerts():
    risk_matrix = np.array([[0.10, 0.80, 0.85, 0.60, 0.40, 0.90, 0.95]], dtype=np.float32)
    active_hours = np.array([7], dtype=int)

    policy = simulate_alert_policy(
        risk_matrix=risk_matrix,
        active_hours=active_hours,
        enter_threshold=0.8,
        exit_threshold=0.5,
        min_history_hours=1,
        min_consecutive_hours=2,
        refractory_hours=2,
        max_alerts_per_stay=2,
    )

    event_hours = np.flatnonzero(policy["event_matrix"][0]) + 1
    assert event_hours.tolist() == [3, 7]
    assert policy["alert_state_matrix"][0].tolist() == [False, False, True, True, True, False, True]


def test_policy_grid_selection_and_artifact_export():
    labels = np.array([1, 1, 0, 0], dtype=int)
    risk_matrix = np.array(
        [
            [0.20, 0.55, 0.80, 0.86, 0.82, 0.78],
            [0.15, 0.52, 0.76, 0.88, 0.91, 0.87],
            [0.10, 0.22, 0.25, 0.30, 0.32, 0.28],
            [0.12, 0.35, 0.45, 0.55, 0.42, 0.38],
        ],
        dtype=np.float32,
    )
    active_hours = np.array([6, 6, 6, 6], dtype=int)
    candidates = build_policy_grid(
        enter_thresholds=[0.5, 0.8],
        hysteresis_gaps=[0.0, 0.1],
        min_consecutive_hours=[1, 2],
        refractory_hours=[0, 2],
        max_alerts_per_stay=[1],
        min_history_hours=2,
    )

    policy_frame = evaluate_policy_grid(
        labels=labels,
        risk_matrix=risk_matrix,
        active_hours=active_hours,
        candidates=candidates,
        landmark_hours=(2, 4, 6),
    )
    best_policy, ranked_candidates = select_best_policy(
        policy_frame,
        max_negative_patient_alert_rate=0.25,
        max_alert_events_per_patient_day=4.0,
        min_positive_patient_alert_rate=0.5,
        min_positive_alert_rate_24h=0.0,
    )

    assert not ranked_candidates.empty
    assert best_policy["constraint_penalty"] >= 0.0
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts = write_policy_artifacts(
            output_dir=Path(tmp_dir),
            source_name="synthetic",
            ranked_candidates=ranked_candidates,
            best_policy=best_policy,
            constraints={"min_positive_alert_rate_24h": 0.0},
        )
        assert Path(artifacts["candidates_csv"]).exists()
        assert Path(artifacts["best_policy_json"]).exists()


def test_policy_grid_supports_multiple_min_history_values():
    candidates = build_policy_grid(
        enter_thresholds=[0.7],
        min_history_hours=[6, 12],
        hysteresis_gaps=[0.0],
        min_consecutive_hours=[2],
        refractory_hours=[6],
        max_alerts_per_stay=[1],
    )

    assert len(candidates) == 2
    assert {candidate["min_history_hours"] for candidate in candidates} == {6, 12}


def test_policy_selection_burden_first_prefers_lower_alert_burden():
    policy_frame = pd.DataFrame(
        [
            {
                "policy_name": "high_recall",
                "positive_patient_alert_rate": 0.72,
                "negative_patient_alert_rate": 0.24,
                "alert_events_per_patient_day": 0.82,
                "alert_state_hours_per_patient_day": 3.8,
                "patient_alert_rate": 0.33,
                "positive_alert_rate_at_24h": 0.64,
                "median_first_alert_hour_positive": 7.0,
                "enter_threshold": 0.35,
                "min_consecutive_hours": 2,
                "min_history_hours": 6,
            },
            {
                "policy_name": "lower_burden",
                "positive_patient_alert_rate": 0.66,
                "negative_patient_alert_rate": 0.18,
                "alert_events_per_patient_day": 0.31,
                "alert_state_hours_per_patient_day": 1.6,
                "patient_alert_rate": 0.24,
                "positive_alert_rate_at_24h": 0.58,
                "median_first_alert_hour_positive": 8.0,
                "enter_threshold": 0.55,
                "min_consecutive_hours": 3,
                "min_history_hours": 12,
            },
        ]
    )

    best_policy, ranked = select_best_policy(
        policy_frame,
        max_negative_patient_alert_rate=0.25,
        max_alert_events_per_patient_day=1.0,
        min_positive_patient_alert_rate=0.6,
        min_positive_alert_rate_24h=0.5,
        ranking_mode="burden_first",
    )

    assert bool(ranked.iloc[0]["feasible"]) is True
    assert best_policy["policy_name"] == "lower_burden"


def test_realtime_monitor_supports_partial_buffer_and_deployment_policy():
    policy_payload = {
        "source": "synthetic_eicu",
        "constraints": {"max_negative_patient_alert_rate": 0.25},
        "best_policy": {
            "policy_name": "policy_live",
            "enter_threshold": 0.7,
            "exit_threshold": 0.5,
            "min_history_hours": 2,
            "min_consecutive_hours": 2,
            "refractory_hours": 2,
            "max_alerts_per_stay": 2,
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        policy_path = Path(tmp_dir) / "best_policy.json"
        policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
        policy_artifact = load_policy_artifact(policy_path)

        monitor = RealtimePhenotypeMonitor(
            model=_DummyRealtimeRiskModel(),
            threshold=0.5,
            phenotype_centroids=np.zeros((2, 4), dtype=np.float32),
            device="cpu",
            temperature=1.0,
            deployment_policy=policy_artifact,
        )
        buffer = RealtimePatientBuffer(seq_len=6, n_cont_features=1, n_treat_features=1)

        probabilities = [0.40, 0.75, 0.80, 0.45, 0.40, 0.85, 0.90]
        snapshots = []
        for prob in probabilities:
            logit = np.log(prob / (1.0 - prob))
            buffer.update(
                values=np.array([logit], dtype=np.float32),
                mask=np.array([1.0], dtype=np.float32),
                treatments=np.array([0.0], dtype=np.float32),
                treatment_mask=np.array([0.0], dtype=np.float32),
            )
            snapshots.append(monitor.predict(buffer))

        assert snapshots[0] is not None
        assert snapshots[0]["deployment_ready"] is False
        assert snapshots[1]["deployment_ready"] is True
        assert snapshots[1]["risk_alert"] is False
        assert snapshots[2]["alert_event"] is True
        assert snapshots[2]["risk_alert"] is True
        assert snapshots[2]["alerts_emitted"] == 1
        assert snapshots[2]["first_alert_hour"] == 3
        assert snapshots[2]["phenotype"] == 0
        assert snapshots[3]["risk_alert"] is True
        assert snapshots[3]["alert_event"] is False
        assert snapshots[4]["risk_alert"] is False
        assert snapshots[5]["risk_alert"] is False
        assert snapshots[6]["alert_event"] is True
        assert snapshots[6]["alerts_emitted"] == 2

        fresh_buffer = RealtimePatientBuffer(seq_len=6, n_cont_features=1, n_treat_features=1)
        fresh_buffer.update(
            values=np.array([0.0], dtype=np.float32),
            mask=np.array([1.0], dtype=np.float32),
            treatments=np.array([0.0], dtype=np.float32),
            treatment_mask=np.array([0.0], dtype=np.float32),
        )
        reset_snapshot = monitor.predict(fresh_buffer)
        assert reset_snapshot is not None
        assert reset_snapshot["deployment_ready"] is False
        assert reset_snapshot["alerts_emitted"] == 0
        assert reset_snapshot["first_alert_hour"] is None


if __name__ == "__main__":
    test_realtime_student_and_dashboard()
    test_realtime_student_v2_tcn_and_alert_rate_threshold()
    test_realtime_student_supports_checkpoint_warm_start()
    test_threshold_selection_supports_alert_budget()
    test_silent_deployment_replay_outputs_summary_and_dashboard()
    test_deployment_policy_state_machine_controls_repeat_alerts()
    test_policy_grid_selection_and_artifact_export()
    test_realtime_monitor_supports_partial_buffer_and_deployment_policy()
    print("1 passed, 0 failed")
