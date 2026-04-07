from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s5.finetune_prep import build_s5_finetune_prep_bundle, write_s5_finetune_prep_artifacts


def test_s5_finetune_prep_materializes_run_config_and_manifest():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "config").mkdir()
        (root / "data" / "external_temporal" / "mimic" / "s0").mkdir(parents=True)
        (root / "data" / "s4" / "mimic_treatments").mkdir(parents=True)
        (root / "data" / "external_temporal" / "mimic" / "s15").mkdir(parents=True)
        (root / "data" / "s5_v2_cloud" / "realtime_mimic_transformer_v2_tempcal_20260401").mkdir(parents=True)
        (root / "outputs" / "reports" / "s5_adaptation_trigger_20260402" / "mimic_v2").mkdir(parents=True)

        (root / "data" / "external_temporal" / "mimic" / "s0" / "static.csv").write_text("x\n1\n", encoding="utf-8")
        (root / "data" / "s4" / "mimic_treatments" / "cohort_static.csv").write_text("x\n1\n", encoding="utf-8")
        (root / "data" / "external_temporal" / "mimic" / "s15" / "embeddings_s15.npy").write_bytes(b"npy")
        (root / "data" / "s5_v2_cloud" / "realtime_mimic_transformer_v2_tempcal_20260401" / "realtime_student.pt").write_bytes(b"pt")
        (root / "data" / "s5_v2_cloud" / "realtime_mimic_transformer_v2_tempcal_20260401" / "realtime_student_report.json").write_text(
            json.dumps({"ok": True}),
            encoding="utf-8",
        )

        trigger_report = root / "outputs" / "reports" / "s5_adaptation_trigger_20260402" / "mimic_v2" / "s5_adaptation_trigger_report.json"
        trigger_report.write_text(
            json.dumps(
                {
                    "source": "mimic_v2",
                    "source_key": "mimic",
                    "triggered": True,
                    "next_step": "start_source_specific_full_finetune",
                    "production_policy_ready": False,
                    "shadow_policy_ready": True,
                    "offline_model_ready": True,
                    "search_exhausted": True,
                    "metrics_snapshot": {
                        "offline_validation": {"auroc": 0.8747},
                    },
                }
            ),
            encoding="utf-8",
        )

        template_config = root / "config" / "s5_mimic_full_finetune_prep.yaml"
        template_config.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "s0_dir": "data/external_temporal/mimic/s0",
                        "treatment_dir": "data/s4/mimic_treatments",
                        "teacher_embeddings": "data/external_temporal/mimic/s15/embeddings_s15.npy",
                        "output_dir": "data/s5_mimic_adapt/_pending_materialization",
                        "base_student_artifact": "data/s5_v2_cloud/realtime_mimic_transformer_v2_tempcal_20260401/realtime_student.pt",
                        "base_student_report": "data/s5_v2_cloud/realtime_mimic_transformer_v2_tempcal_20260401/realtime_student_report.json",
                        "trigger_report": "outputs/reports/s5_adaptation_trigger_20260402/mimic_v2/s5_adaptation_trigger_report.json",
                    },
                    "adaptation": {
                        "source": "mimic_v2",
                        "source_key": "mimic",
                        "output_root": "data/s5_mimic_adapt",
                        "run_name_prefix": "mimic_v2_full_finetune",
                    },
                    "training": {
                        "batch_size": 128,
                        "init_checkpoint_strict": True,
                    },
                    "model": {
                        "student_arch": "transformer",
                    },
                    "runtime": {
                        "device": "auto",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        bundle = build_s5_finetune_prep_bundle(
            template_config_path=template_config,
            project_root=root,
            run_name="mimic_v2_full_finetune_20260402",
        )
        artifacts = write_s5_finetune_prep_artifacts(bundle)

        train_config_path = Path(artifacts["train_config_path"])
        manifest_path = Path(artifacts["prep_manifest_path"])
        assert train_config_path.exists()
        assert manifest_path.exists()
        assert (Path(artifacts["run_dir"]) / "checkpoints").exists()

        written_config = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
        written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert written_config["paths"]["output_dir"] == "data/s5_mimic_adapt/mimic_v2_full_finetune_20260402"
        assert written_config["paths"]["init_checkpoint"].endswith("realtime_student.pt")
        assert written_config["adaptation"]["trigger_snapshot"]["triggered"] is True
        assert written_manifest["launch"]["command"].endswith(
            "--config data/s5_mimic_adapt/mimic_v2_full_finetune_20260402/train_config.yaml"
        )
        assert written_manifest["base_artifacts"]["base_student_artifact"].endswith("realtime_student.pt")


def test_s5_finetune_prep_requires_trigger_by_default():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / "config").mkdir()
        (root / "data" / "external_temporal" / "mimic" / "s0").mkdir(parents=True)
        (root / "data" / "s4" / "mimic_treatments").mkdir(parents=True)
        (root / "data" / "external_temporal" / "mimic" / "s15").mkdir(parents=True)
        (root / "outputs").mkdir()

        template_config = root / "config" / "prep.yaml"
        trigger_report = root / "outputs" / "trigger.json"
        trigger_report.write_text(
            json.dumps(
                {
                    "triggered": False,
                    "next_step": "continue_shadow_monitoring",
                }
            ),
            encoding="utf-8",
        )
        template_config.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "s0_dir": "data/external_temporal/mimic/s0",
                        "treatment_dir": "data/s4/mimic_treatments",
                        "teacher_embeddings": "data/external_temporal/mimic/s15",
                        "output_dir": "data/s5_mimic_adapt/_pending_materialization",
                        "trigger_report": "outputs/trigger.json",
                    },
                    "adaptation": {
                        "output_root": "data/s5_mimic_adapt",
                    },
                    "training": {
                        "init_checkpoint_strict": True,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        try:
            build_s5_finetune_prep_bundle(
                template_config_path=template_config,
                project_root=root,
                run_name="blocked_run",
            )
        except ValueError as exc:
            assert "Fine-tune prep requires a triggered adaptation report" in str(exc)
        else:
            raise AssertionError("Expected fine-tune prep to reject an untriggered report")
