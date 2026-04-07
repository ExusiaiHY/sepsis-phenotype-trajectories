from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s4.reporting import build_s4_closeout_summary, write_s4_closeout_artifacts
from s5.reporting import (
    build_s5_adaptation_trigger_report,
    build_s5_validation_summary,
    write_s5_adaptation_trigger_artifacts,
    write_s5_validation_artifacts,
)


def test_s4_closeout_reporting_writes_expected_artifacts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        mimic_treatment = root / "mimic_treatment.json"
        eicu_treatment = root / "eicu_treatment.json"
        mimic_causal = root / "mimic_causal.json"
        eicu_causal = root / "eicu_causal.json"

        common_treatment = {
            "splits": {
                "test": {
                    "n_samples": 100,
                    "positive_rate": 0.2,
                    "predicted_positive_rate": 0.3,
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.79,
                    "precision": 0.4,
                    "recall": 0.75,
                    "f1": 0.52,
                    "auroc": 0.88,
                }
            },
            "calibration": {"test": {"brier": 0.09, "ece": 0.01}},
        }
        mimic_treatment.write_text(json.dumps(common_treatment), encoding="utf-8")
        eicu_treatment.write_text(
            json.dumps(
                {
                    **common_treatment,
                    "splits": {
                        "test": {
                            **common_treatment["splits"]["test"],
                            "auroc": 0.9,
                            "balanced_accuracy": 0.82,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        common_treatments = {
            "vasopressor_on_any_6h": {
                "psm": {"ate": -0.01},
                "causal_forest_dml": {"cate_mean": -0.02, "cate_std": 0.1},
                "rdd": {"local_effect": None},
            },
            "antibiotic_on_any_6h": {
                "psm": {"ate": 0.002},
                "causal_forest_dml": {"cate_mean": 0.001, "cate_std": 0.05},
                "rdd": {"local_effect": None},
            },
        }
        mimic_causal.write_text(
            json.dumps(
                {
                    "treatments": common_treatments,
                    "recommendations": [
                        {
                            "treatment_col": "vasopressor_on_any_6h",
                            "direction": "candidate_beneficial",
                            "clinical_note": "benefit",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        eicu_causal.write_text(
            json.dumps(
                {
                    "treatments": {
                        "vasopressor_on_any_6h": {
                            "psm": {"ate": 0.03},
                            "causal_forest_dml": {"cate_mean": 0.02, "cate_std": 0.1},
                            "rdd": {"local_effect": None},
                        },
                        "antibiotic_on_any_6h": {
                            "psm": {"ate": -0.001},
                            "causal_forest_dml": {"cate_mean": 0.0, "cate_std": 0.04},
                            "rdd": {"local_effect": None},
                        },
                    },
                    "recommendations": [
                        {
                            "treatment_col": "vasopressor_on_any_6h",
                            "direction": "candidate_harmful",
                            "clinical_note": "harm",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        bundle = build_s4_closeout_summary(
            mimic_treatment_report_path=mimic_treatment,
            eicu_treatment_report_path=eicu_treatment,
            mimic_causal_report_path=mimic_causal,
            eicu_causal_report_path=eicu_causal,
        )
        artifacts = write_s4_closeout_artifacts(
            bundle,
            reports_dir=root / "reports_s4",
            figures_dir=root / "figures_s4",
        )

        assert bundle["summary"]["n_sources"] == 2
        assert len(bundle["cross_source_comparison"]) == 2
        assert Path(artifacts["summary_json"]).exists()
        assert Path(artifacts["performance_csv"]).exists()
        assert Path(artifacts["causal_fig"]).exists()


def test_s5_validation_reporting_writes_expected_artifacts():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        mimic_report = root / "mimic_realtime.json"
        eicu_report = root / "eicu_realtime.json"
        common = {
            "threshold_selection": {"selected_threshold": 0.1},
            "training": {"epochs_trained": 5},
            "splits": {
                "test": {
                    "n_samples": 100,
                    "positive_rate": 0.2,
                    "predicted_positive_rate": 0.25,
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.79,
                    "precision": 0.4,
                    "recall": 0.8,
                    "f1": 0.53,
                    "auroc": 0.88,
                }
            },
            "calibration": {"test": {"brier": 0.09, "ece": 0.01}},
            "deployment": {
                "cpu_latency_ms_per_sample": 0.8,
                "float_n_parameters": 91000,
                "dynamic_quantization_ok": False,
            },
        }
        mimic_report.write_text(json.dumps(common), encoding="utf-8")
        eicu_report.write_text(
            json.dumps(
                {
                    **common,
                    "splits": {
                        "test": {
                            **common["splits"]["test"],
                            "auroc": 0.9,
                            "balanced_accuracy": 0.82,
                        }
                    },
                    "deployment": {
                        **common["deployment"],
                        "cpu_latency_ms_per_sample": 1.2,
                    },
                }
            ),
            encoding="utf-8",
        )

        bundle = build_s5_validation_summary(
            mimic_report_path=mimic_report,
            eicu_report_path=eicu_report,
        )
        artifacts = write_s5_validation_artifacts(
            bundle,
            reports_dir=root / "reports_s5",
            figures_dir=root / "figures_s5",
        )

        assert bundle["summary"]["sources_passing_all_gates"] == 2
        assert len(bundle["realtime_validation_metrics"]) == 2
        assert Path(artifacts["summary_json"]).exists()
        assert Path(artifacts["metrics_csv"]).exists()
        assert Path(artifacts["deployment_fig"]).exists()


def test_s5_adaptation_trigger_reporting_detects_escalation_condition():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        validation_summary = root / "s5_validation_summary.json"
        policy_best_json = root / "mimic_best_policy.json"
        policy_candidates_csv = root / "mimic_candidates.csv"
        shadow_summary = root / "mimic_shadow_summary.json"
        trigger_config = root / "mimic_trigger_config.json"

        validation_summary.write_text(
            json.dumps(
                {
                    "realtime_validation_metrics": [
                        {
                            "source": "mimic",
                            "source_label": "MIMIC-IV",
                            "auroc": 0.8747,
                            "balanced_accuracy": 0.7859,
                            "ece": 0.0172,
                            "cpu_latency_ms_per_sample": 1.117,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        policy_best_json.write_text(
            json.dumps(
                {
                    "constraints": {
                        "grid_profile": "tight",
                        "ranking_mode": "burden_first",
                    },
                    "best_policy": {
                        "feasible": False,
                        "negative_patient_alert_rate": 0.3173,
                        "alert_events_per_patient_day": 0.2053,
                        "positive_patient_alert_rate": 0.5671,
                        "positive_alert_rate_at_24h": 0.5388,
                    },
                }
            ),
            encoding="utf-8",
        )
        policy_candidates_csv.write_text(
            "\n".join(
                [
                    "policy_name,feasible",
                    "policy_0001,False",
                    "policy_0002,False",
                    "policy_0003,False",
                    "policy_0004,False",
                    "policy_0005,False",
                ]
            ),
            encoding="utf-8",
        )
        shadow_summary.write_text(
            json.dumps(
                {
                    "negative_patient_alert_rate": 0.3173,
                    "positive_patient_alert_rate": 0.5671,
                    "alert_events_per_patient_day": 0.2053,
                    "alert_state_hours_per_patient_day": 2.1493,
                    "patient_alert_rate": 0.3619,
                    "median_first_alert_hour_positive": 7.0,
                    "cumulative_alert_metrics": [
                        {"hour": 12, "positive_alert_rate": 0.5364},
                        {"hour": 24, "positive_alert_rate": 0.5388},
                    ],
                }
            ),
            encoding="utf-8",
        )
        trigger_config.write_text(
            json.dumps(
                {
                    "source": "mimic_v2",
                    "offline_quality_gates": {
                        "min_auroc": 0.87,
                        "min_balanced_accuracy": 0.78,
                        "max_ece": 0.02,
                        "max_latency_ms_per_sample": 2.0,
                    },
                    "production_policy_gates": {
                        "max_negative_patient_alert_rate": 0.25,
                        "max_alert_events_per_patient_day": 1.0,
                        "min_positive_patient_alert_rate": 0.6,
                        "min_positive_alert_rate_24h": 0.5,
                    },
                    "shadow_policy_gates": {
                        "max_negative_patient_alert_rate": 0.35,
                        "max_alert_events_per_patient_day": 0.25,
                        "min_positive_patient_alert_rate": 0.55,
                        "min_positive_alert_rate_24h": 0.5,
                        "max_median_first_alert_hour_positive": 12.0,
                    },
                    "search_exhaustion_gates": {
                        "min_candidate_count": 5,
                        "max_feasible_count": 0,
                        "required_grid_profile": "tight",
                        "required_ranking_mode": "burden_first",
                    },
                    "trigger_policy": {
                        "require_shadow_ready": True,
                        "require_offline_quality_ready": True,
                        "require_production_policy_failure": True,
                        "require_search_exhausted": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        bundle = build_s5_adaptation_trigger_report(
            validation_summary_path=validation_summary,
            policy_best_json_path=policy_best_json,
            policy_candidates_csv_path=policy_candidates_csv,
            shadow_replay_summary_path=shadow_summary,
            trigger_config_path=trigger_config,
            source_key="mimic",
        )
        artifacts = write_s5_adaptation_trigger_artifacts(
            bundle,
            reports_dir=root / "trigger_reports",
        )

        assert bundle["triggered"] is True
        assert bundle["next_step"] == "start_source_specific_full_finetune"
        assert bundle["shadow_policy_ready"] is True
        assert bundle["offline_model_ready"] is True
        assert bundle["production_policy_ready"] is False
        assert bundle["search_exhausted"] is True
        assert Path(artifacts["summary_json"]).exists()
