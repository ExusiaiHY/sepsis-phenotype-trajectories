from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient

from s6.api_service import app, load_artifacts
from s6.masked_npz_runtime import MaskedNPZMultitaskStudent, SepsisSubtypeInferenceEngine
from s6.treatment_recommender import SubtypeTreatmentRecommender


def _build_temp_model_dir(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    classification_tasks = [
        {"name": "gold_mals", "n_classes": 2},
        {"name": "proxy_clinical_phenotype", "n_classes": 5},
        {"name": "proxy_trajectory_phenotype", "n_classes": 5},
        {"name": "proxy_fluid_strategy", "n_classes": 3},
    ]
    regression_tasks = [
        {"name": "score_alpha"},
        {"name": "score_trajectory_a"},
        {"name": "score_restrictive_fluid_benefit"},
    ]
    config = {
        "n_cont_features": 2,
        "n_treat_features": 1,
        "note_dim": 0,
        "student_arch": "transformer",
        "student_d_model": 8,
        "teacher_dim": 0,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 16,
        "dropout": 0.1,
        "max_seq_len": 6,
        "treatment_layers": 1,
        "head_hidden_dim": 8,
        "head_dropout": 0.1,
        "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2],
        "classification_tasks": classification_tasks,
        "regression_tasks": regression_tasks,
    }
    model = MaskedNPZMultitaskStudent(
        n_cont_features=2,
        n_treat_features=1,
        classification_tasks=classification_tasks,
        regression_tasks=regression_tasks,
        student_arch="transformer",
        student_d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        max_seq_len=6,
        head_hidden_dim=8,
        tcn_dilations=(1, 2),
    )
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
        },
        model_dir / "multitask_student.pt",
    )
    np.save(model_dir / "feat_mean.npy", np.zeros(2, dtype=np.float32))
    np.save(model_dir / "feat_std.npy", np.ones(2, dtype=np.float32))
    report = {
        "model": config,
        "targets_source": "npz",
        "training": {"epochs_trained": 1},
        "threshold_selection": {"selected_threshold": 0.4},
        "splits": {
            "test": {
                "mortality": {"auroc": 0.5, "f1": 0.5, "balanced_accuracy": 0.5},
                "classification": {
                    "gold_mals": {"auroc": 0.8},
                    "proxy_clinical_phenotype": {"macro_f1": 0.7},
                    "proxy_trajectory_phenotype": {"macro_f1": 0.5},
                    "proxy_fluid_strategy": {"macro_f1": 0.6},
                },
                "regression": {
                    "score_restrictive_fluid_benefit": {"rmse": 0.1},
                },
            }
        },
        "deployment": {"float_n_parameters": sum(p.numel() for p in model.parameters()), "cpu_latency_ms_per_sample": 1.0},
    }
    (model_dir / "multitask_student_report.json").write_text(json.dumps(report), encoding="utf-8")
    schema = {
        "classification_tasks": [
            {"name": "gold_mals", "classes": ["negative", "positive"], "description": "gold mals"},
            {
                "name": "proxy_clinical_phenotype",
                "classes": ["Unclassified", "alpha-like", "beta-like", "gamma-like", "delta-like"],
                "description": "clinical phenotype",
            },
            {
                "name": "proxy_trajectory_phenotype",
                "classes": ["Unclassified", "group-a", "group-b", "group-c", "group-d"],
                "description": "trajectory phenotype",
            },
            {
                "name": "proxy_fluid_strategy",
                "classes": ["Unclassified", "restrictive-fluid-benefit-like", "resuscitation-fluid-benefit-like"],
                "description": "fluid strategy",
            },
        ],
        "regression_tasks": [
            {"name": "score_alpha", "description": "alpha score"},
            {"name": "score_trajectory_a", "description": "trajectory a score"},
            {"name": "score_restrictive_fluid_benefit", "description": "restrictive fluid benefit"},
        ],
    }
    schema_path = model_dir / "sepsis_multitask_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    return model_dir, schema_path


def test_inference_engine_exposes_distinct_clinical_and_trajectory_families(tmp_path: Path):
    model_dir, schema_path = _build_temp_model_dir(tmp_path / "model")
    engine = SepsisSubtypeInferenceEngine(model_dir=model_dir, schema_path=schema_path, device="cpu")
    meta = engine.metadata()

    family_notes = {item["family_id"]: item.get("frontend_note", "") for item in meta["families"]}
    assert "alpha / beta / gamma / delta" in family_notes["clinical_phenotype"]
    assert "Trajectory A / B / C / D" in family_notes["trajectory_phenotype"]

    batch = np.random.default_rng(7).normal(size=(1, 4, 3)).astype(np.float32)
    payload = engine.predict_batch(batch)
    prediction = payload["predictions"][0]

    assert "proxy_clinical_phenotype" in prediction["classification_tasks"]
    assert "proxy_trajectory_phenotype" in prediction["classification_tasks"]
    assert prediction["families"]["clinical_phenotype"]["task_name"] == "proxy_clinical_phenotype"
    assert prediction["families"]["trajectory_phenotype"]["task_name"] == "proxy_trajectory_phenotype"

    recommender = SubtypeTreatmentRecommender()
    recommendation = recommender.recommend_from_prediction(prediction)
    assert "family_recommendations" in recommendation
    assert "clinical_phenotype" in recommendation["family_recommendations"]


def test_fastapi_contract_returns_metadata_predict_and_recommend(tmp_path: Path):
    model_dir, schema_path = _build_temp_model_dir(tmp_path / "service_model")
    load_artifacts(model_dir=model_dir, schema_path=schema_path, device="cpu")
    client = TestClient(app)

    metadata_response = client.get("/metadata")
    assert metadata_response.status_code == 200
    metadata_payload = metadata_response.json()
    assert metadata_payload["contract_version"] == "s6-masked-npz-v1"

    sample = np.random.default_rng(11).normal(size=(1, 4, 3)).astype(np.float32).tolist()
    predict_response = client.post("/predict", json={"time_series": sample})
    assert predict_response.status_code == 200
    predict_payload = predict_response.json()
    assert len(predict_payload["predictions"]) == 1

    recommend_response = client.post("/recommend", json={"time_series": sample})
    assert recommend_response.status_code == 200
    recommend_payload = recommend_response.json()
    assert len(recommend_payload["recommendations"]) == 1
    assert "metadata" in recommend_payload
