"""
test_s5_bedside_service.py - Stage 5 bedside session manager and HTTP API tests.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s5.bedside_service import (
    BedsideMonitoringService,
    dispatch_bedside_request,
)


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


def test_bedside_monitoring_service_tracks_patient_sessions_and_dashboard():
    policy_artifact = {
        "source": "synthetic_site",
        "policy": {
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
        dashboard_dir = Path(tmp_dir) / "dashboards"
        service = BedsideMonitoringService(
            model=_DummyRealtimeRiskModel(),
            model_config={
                "n_cont_features": 1,
                "n_treat_features": 2,
                "note_dim": 0,
                "max_seq_len": 6,
            },
            threshold=0.5,
            temperature=1.0,
            policy_artifact=policy_artifact,
            phenotype_centroids=np.zeros((2, 4), dtype=np.float32),
            device="cpu",
            max_snapshots=4,
            dashboard_dir=dashboard_dir,
            treatment_feature_names=["vasopressor_on", "fluid_ml"],
        )

        for prob in (0.40, 0.75, 0.80):
            logit = np.log(prob / (1.0 - prob))
            response = service.ingest_update(
                patient_id="icu-1",
                values=[logit],
                mask=[1.0],
                treatments=[1.0, 120.0],
                treatment_mask=[1.0, 1.0],
            )

        assert response["snapshot"]["alert_event"] is True
        assert response["snapshot"]["risk_alert"] is True
        assert response["snapshot"]["top_treatment_signal"] == "vasopressor_on"
        assert service.describe()["n_active_patients"] == 1
        assert service.list_patients()[0]["patient_id"] == "icu-1"
        record = service.get_patient_record("icu-1")
        assert len(record["snapshots"]) == 3

        rendered = service.render_patient_dashboard("icu-1")
        assert Path(rendered["output_path"]).exists()
        assert "icu-1" in rendered["html"]
        assert "Alert event emitted" in rendered["html"]

        reset = service.reset_patient("icu-1")
        assert reset["removed"] is True
        assert service.describe()["n_active_patients"] == 0


def test_bedside_http_dispatch_accepts_updates_and_renders_dashboard():
    policy_artifact = {
        "source": "synthetic_site",
        "policy": {
            "policy_name": "policy_http",
            "enter_threshold": 0.7,
            "exit_threshold": 0.5,
            "min_history_hours": 2,
            "min_consecutive_hours": 2,
            "refractory_hours": 2,
            "max_alerts_per_stay": 1,
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        dashboard_dir = Path(tmp_dir) / "dashboards"
        service = BedsideMonitoringService(
            model=_DummyRealtimeRiskModel(),
            model_config={
                "n_cont_features": 1,
                "n_treat_features": 1,
                "note_dim": 0,
                "max_seq_len": 6,
            },
            threshold=0.5,
            temperature=1.0,
            policy_artifact=policy_artifact,
            phenotype_centroids=np.zeros((2, 4), dtype=np.float32),
            device="cpu",
            max_snapshots=8,
            dashboard_dir=dashboard_dir,
            treatment_feature_names=["vasopressor_on"],
        )
        health = dispatch_bedside_request(service=service, method="GET", path="/health", body=None)
        assert health["body"]["status"] == "ok"

        for prob in (0.40, 0.75, 0.80):
            logit = float(np.log(prob / (1.0 - prob)))
            update = dispatch_bedside_request(
                service=service,
                method="POST",
                path="/patients/icu-http/update",
                body=json.dumps(
                    {
                        "values": [logit],
                        "mask": [1.0],
                        "treatments": [1.0],
                        "treatment_mask": [1.0],
                    }
                ).encode("utf-8"),
            )
        assert update["body"]["snapshot"]["alert_event"] is True
        assert update["body"]["snapshot"]["alerts_emitted"] == 1

        patient = dispatch_bedside_request(service=service, method="GET", path="/patients/icu-http", body=None)
        assert patient["body"]["summary"]["hours_seen"] == 3
        assert len(patient["body"]["snapshots"]) == 3

        html = dispatch_bedside_request(service=service, method="GET", path="/patients/icu-http/dashboard", body=None)
        assert "icu-http" in html["body"]
        assert "Alert event emitted" in html["body"]

        reset = dispatch_bedside_request(service=service, method="POST", path="/patients/icu-http/reset", body=b"")
        assert reset["body"]["removed"] is True
