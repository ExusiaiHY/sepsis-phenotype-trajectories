from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from subtype_label_engine import build_subtype_labels


def _build_static_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [40, 78, 75, 55],
            "mortality_28d": [1, 1, 0, 0],
            "fd_ferritin_max": [6000.0, 150.0, 300.0, 200.0],
            "fd_crp_max": [220.0, 90.0, 170.0, 60.0],
            "fd_platelet_min": [70.0, 180.0, 210.0, 160.0],
            "fd_bilirubin_max": [4.0, 0.8, 1.1, 4.2],
            "fd_alt_max": [180.0, 40.0, 50.0, 180.0],
            "fd_ddimer_max": [6500.0, 500.0, 900.0, 800.0],
            "fd_lymphocytes_min": [0.9, 0.3, 1.1, 1.5],
            "fd_lymphocytes_pct_min": [12.0, 6.0, 15.0, 18.0],
            "fd_lactate_max": [4.2, 1.4, 3.8, 1.8],
            "fd_sbp_min": [82.0, 92.0, 84.0, 118.0],
            "fd_mbp_min": [58.0, 68.0, 60.0, 75.0],
            "fd_spo2_min": [92.0, 95.0, 82.0, 96.0],
            "fd_rr_min": [24.0, 9.0, 30.0, 16.0],
            "first_day_sofa": [14.0, 8.0, 6.0, 7.0],
            "sofa_resp": [2.0, 1.0, 4.0, 1.0],
            "sofa_coag": [2.0, 0.0, 0.0, 0.0],
            "sofa_liver": [3.0, 0.0, 0.0, 4.0],
            "sofa_cardio": [3.0, 4.0, 1.0, 1.0],
            "sofa_cns": [1.0, 0.0, 0.0, 0.0],
            "sofa_renal": [3.0, 4.0, 1.0, 4.0],
            "mech_vent_first24h": [0, 0, 1, 0],
            "blood_culture_positive": [0, 1, 0, 0],
            "resp_culture_positive": [0, 0, 1, 0],
            "any_culture_positive": [0, 1, 1, 0],
            "mhla_dr": [np.nan, 3000.0, np.nan, np.nan],
        }
    )


def _build_timeseries_frame() -> pd.DataFrame:
    rows: list[dict] = []
    patient_specs = {
        1: {"heart_rate": 120.0, "resp_rate": 30.0, "temperature": 39.0, "sbp": 85.0, "map": 60.0},
        2: {"heart_rate": 50.0, "resp_rate": 9.0, "temperature": 35.2, "sbp": 82.0, "map": 55.0},
        3: {"heart_rate": 110.0, "resp_rate": 26.0, "temperature": 39.0, "sbp": 160.0, "map": 95.0},
        4: {"heart_rate": 80.0, "resp_rate": 16.0, "temperature": 36.8, "sbp": 120.0, "map": 75.0},
    }
    for stay_id, spec in patient_specs.items():
        for hr in range(4):
            rows.append({"stay_id": stay_id, "hr": hr, **spec})
    return pd.DataFrame(rows)


def test_build_subtype_labels_emits_split_multitask_schema(tmp_path):
    static_path = tmp_path / "patient_static_enhanced.csv"
    timeseries_path = tmp_path / "patient_timeseries_enhanced.csv"
    _build_static_frame().to_csv(static_path, index=False)
    _build_timeseries_frame().to_csv(timeseries_path, index=False)

    result = build_subtype_labels(
        static_path=static_path,
        timeseries_path=timeseries_path,
        output_dir=tmp_path,
        output_format="csv",
    )

    required_cols = {
        "gold_mals_label",
        "mask_gold_mals_label",
        "gold_immunoparalysis_label",
        "mask_gold_immunoparalysis_label",
        "proxy_immune_state",
        "proxy_clinical_phenotype",
        "proxy_trajectory_phenotype",
        "proxy_fluid_strategy",
        "score_mals",
        "score_immunoparalysis",
        "score_delta",
        "score_trajectory_a",
        "score_restrictive_fluid_benefit",
        "immune_subtype",
        "organ_subtype",
        "fluid_benefit_proxy",
    }
    assert required_cols.issubset(set(result.columns))

    row_1 = result.loc[result["stay_id"] == 1].iloc[0]
    row_2 = result.loc[result["stay_id"] == 2].iloc[0]
    row_3 = result.loc[result["stay_id"] == 3].iloc[0]
    row_4 = result.loc[result["stay_id"] == 4].iloc[0]

    assert row_1["gold_mals_label"] == 1
    assert row_1["mask_gold_mals_label"] == 1.0
    assert row_1["proxy_immune_state"] == "MALS-like"
    assert row_1["immune_subtype"] == "MAS-like"

    assert row_2["gold_immunoparalysis_label"] == 1
    assert row_2["mask_gold_immunoparalysis_label"] == 1.0
    assert row_2["proxy_immune_state"] == "immunoparalysis-like"
    assert row_2["immune_subtype"] == "EIL-like"

    assert row_1["proxy_trajectory_phenotype"] == "group-a"
    assert row_2["proxy_trajectory_phenotype"] == "group-d"
    assert row_3["proxy_trajectory_phenotype"] == "group-b"
    assert row_4["proxy_trajectory_phenotype"] == "group-c"

    assert row_1["proxy_fluid_strategy"] == "resuscitation-fluid-benefit-like"
    assert row_2["proxy_fluid_strategy"] == "restrictive-fluid-benefit-like"
    assert row_1["fluid_benefit_proxy"] == "high_benefit"
    assert row_2["fluid_benefit_proxy"] == "low_benefit"

    saved_csv = tmp_path / "patient_static_with_subtypes.csv"
    targets_npz = tmp_path / "sepsis_multitask_targets.npz"
    schema_json = tmp_path / "sepsis_multitask_schema.json"
    assert saved_csv.exists()
    assert targets_npz.exists()
    assert schema_json.exists()

    bundle = np.load(targets_npz, allow_pickle=True)
    assert bundle["classification_labels"].shape == (4, 6)
    assert bundle["classification_masks"].shape == (4, 6)
    assert bundle["regression_targets"].shape[0] == 4
    assert "proxy_trajectory_phenotype" in bundle["classification_task_names"].tolist()
    assert "score_restrictive_fluid_benefit" in bundle["regression_task_names"].tolist()

    schema = json.loads(schema_json.read_text(encoding="utf-8"))
    assert schema["schema_version"] == "2.0.0"
    assert any(task["name"] == "proxy_fluid_strategy" for task in schema["classification_tasks"])
