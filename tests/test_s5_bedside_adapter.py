"""
test_s5_bedside_adapter.py - Stage 5 bedside prospective replay tests.
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

from s5.bedside_adapter import (
    BedsideFeedAdapter,
    iter_stage5_bedside_feed,
    load_bedside_feed_jsonl,
    write_bedside_feed_jsonl,
)
from s5.bedside_service import BedsideMonitoringService


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


def test_bedside_feed_adapter_replays_and_audits_multi_patient_stream():
    policy_artifact = {
        "source": "synthetic_site",
        "policy": {
            "policy_name": "policy_adapter",
            "enter_threshold": 0.7,
            "exit_threshold": 0.5,
            "min_history_hours": 2,
            "min_consecutive_hours": 2,
            "refractory_hours": 2,
            "max_alerts_per_stay": 1,
        },
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
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
            dashboard_dir=root / "service_dashboards",
            treatment_feature_names=["vasopressor_on"],
        )
        adapter = BedsideFeedAdapter(
            service=service,
            audit_jsonl_path=root / "audit.jsonl",
            dashboard_dir=root / "alert_dashboards",
            render_dashboard_on_alert=True,
        )

        def _event(patient_id: str, prob: float, hour: int) -> dict:
            return {
                "patient_id": patient_id,
                "action": "update",
                "hour_index": hour,
                "step_index": hour,
                "values": [float(np.log(prob / (1.0 - prob)))],
                "mask": [1.0],
                "treatments": [1.0],
                "treatment_mask": [1.0],
                "context": {"label": int(patient_id == "icu-1"), "site": "synthetic"},
            }

        events = [
            _event("icu-1", 0.40, 1),
            _event("icu-2", 0.30, 1),
            _event("icu-1", 0.75, 2),
            _event("icu-2", 0.35, 2),
            _event("icu-1", 0.82, 3),
            {"patient_id": "icu-1", "action": "reset", "step_index": 6, "context": {"site": "synthetic"}},
        ]
        summary = adapter.replay_events(events=events, output_dir=root / "out")

        assert summary["n_events"] == 6
        assert summary["n_updates"] == 5
        assert summary["n_resets"] == 1
        assert summary["n_patients_seen"] == 2
        assert summary["n_alert_events"] == 1
        assert summary["patients_with_alert_events"] == 1
        assert summary["dashboards_rendered"] == 1

        audit_rows = list(load_bedside_feed_jsonl(root / "audit.jsonl"))
        assert len(audit_rows) == 6
        assert any(row.get("snapshot", {}).get("alert_event") is True for row in audit_rows)
        assert any(Path(row["dashboard_path"]).exists() for row in audit_rows if row.get("dashboard_path"))

        patient_summary = pd.read_csv(root / "out" / "bedside_patient_summary.csv")
        assert set(patient_summary["patient_id"]) == {"icu-1", "icu-2"}
        icu1 = patient_summary.loc[patient_summary["patient_id"] == "icu-1"].iloc[0]
        assert int(icu1["n_alert_events"]) == 1
        assert int(icu1["n_resets"]) == 1
        assert bool(icu1["active_session"]) is False

        alert_events = pd.read_csv(root / "out" / "bedside_alert_events.csv")
        assert len(alert_events) == 1
        assert alert_events.iloc[0]["patient_id"] == "icu-1"


def test_iter_stage5_bedside_feed_round_robin_and_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        s0_dir = root / "s0"
        proc_dir = s0_dir / "processed"
        treatment_dir = root / "treatment"
        proc_dir.mkdir(parents=True)
        treatment_dir.mkdir(parents=True)

        continuous = np.asarray(
            [
                [[1.0], [2.0], [0.0], [0.0]],
                [[10.0], [20.0], [30.0], [0.0]],
            ],
            dtype=np.float32,
        )
        masks_cont = np.asarray(
            [
                [[1.0], [1.0], [0.0], [0.0]],
                [[1.0], [1.0], [1.0], [0.0]],
            ],
            dtype=np.float32,
        )
        treatments = np.asarray(
            [
                [[0.1], [0.2], [0.0], [0.0]],
                [[1.1], [1.2], [1.3], [0.0]],
            ],
            dtype=np.float32,
        )
        masks_treat = np.asarray(
            [
                [[1.0], [1.0], [0.0], [0.0]],
                [[1.0], [1.0], [1.0], [0.0]],
            ],
            dtype=np.float32,
        )
        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks_cont)
        np.save(treatment_dir / "treatments.npy", treatments)
        np.save(treatment_dir / "masks_treatments.npy", masks_treat)

        static = pd.DataFrame(
            {
                "patient_id": ["p1", "p2"],
                "mortality_inhospital": [0, 1],
                "icu_los_hours": [2.0, 3.0],
                "center_id": ["c1", "c2"],
                "icu_type": ["med", "surg"],
                "data_source": ["demo", "demo"],
            }
        )
        static.to_csv(s0_dir / "static.csv", index=False)
        pd.DataFrame({"patient_id": ["p1", "p2"]}).to_csv(treatment_dir / "cohort_static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump({"train": [], "val": [], "test": [0, 1]}, f)

        events = list(
            iter_stage5_bedside_feed(
                s0_dir=s0_dir,
                treatment_dir=treatment_dir,
                split="test",
                order="round_robin",
            )
        )
        assert [(event["patient_id"], event["hour_index"]) for event in events] == [
            ("p1", 1),
            ("p2", 1),
            ("p1", 2),
            ("p2", 2),
            ("p2", 3),
        ]
        assert events[0]["context"]["label"] == 0
        assert events[1]["context"]["label"] == 1

        feed_path = root / "feed.jsonl"
        assert write_bedside_feed_jsonl(feed_path, events) == 5
        roundtrip = list(load_bedside_feed_jsonl(feed_path))
        assert len(roundtrip) == 5
        assert roundtrip[-1]["patient_id"] == "p2"
        assert roundtrip[-1]["hour_index"] == 3
