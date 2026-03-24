"""
test_s4_treatment_causal.py - Stage 4 bundle/model/causal smoke tests.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s4.causal_analysis import build_causal_frame, run_causal_suite
from s4.treatment_aware_model import train_treatment_aware_classifier
from s4.treatment_features import build_treatment_feature_bundle


def test_build_eicu_treatment_bundle_extracts_events():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        prepared_dir = root / "prepared"
        raw_dir = root / "raw"
        output_dir = root / "bundle"
        prepared_dir.mkdir()
        raw_dir.mkdir()

        patient_info = pd.DataFrame(
            {
                "patient_id": ["1", "2"],
                "stay_id": [1, 2],
                "mortality_28d": [0, 1],
            }
        )
        patient_info.to_csv(prepared_dir / "patient_info_eicu_demo.csv", index=False)
        with open(prepared_dir / "feature_names_eicu_demo.json", "w", encoding="utf-8") as f:
            json.dump(["heart_rate", "vasopressor", "mechanical_vent", "rrt"], f)

        ts = np.zeros((2, 48, 4), dtype=np.float32)
        ts[0, 2:6, 1] = 1.0
        ts[1, 3:5, 2] = 1.0
        ts[1, 10:14, 3] = 1.0
        np.save(prepared_dir / "time_series_eicu_demo.npy", ts)

        pd.DataFrame(
            {
                "patientunitstayid": [1, 2],
                "drugstartoffset": [120, 60],
                "drugorderoffset": [120, 60],
                "drugstopoffset": [360, 180],
                "drugname": ["Vancomycin", "Metoprolol"],
            }
        ).to_csv(raw_dir / "medication.csv", index=False)
        pd.DataFrame(
            {
                "patientunitstayid": [1, 1, 2],
                "intakeoutputoffset": [60, 120, 120],
                "cellpath": ["Crystalloids saline", "Crystalloids saline", "P.O."],
                "celllabel": ["saline bolus", "saline bolus", "oral"],
                "cellvaluenumeric": [500.0, 200.0, 100.0],
            }
        ).to_csv(raw_dir / "intakeOutput.csv", index=False)

        report = build_treatment_feature_bundle(
            source="eicu",
            prepared_dir=prepared_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
            n_hours=48,
            tag="eicu_demo",
        )

        treatments = np.load(output_dir / "treatments.npy")
        feature_names = json.loads((output_dir / "treatment_feature_names.json").read_text(encoding="utf-8"))
        idx = {name: i for i, name in enumerate(feature_names)}

        assert treatments.shape == (2, 48, len(feature_names))
        assert treatments[0, 2, idx["vasopressor_on"]] == 1.0
        assert treatments[0, 2, idx["antibiotic_on"]] == 1.0
        assert treatments[0, 1, idx["fluid_bolus_ml"]] > 0.0
        assert report["feature_exposure_rate"]["antibiotic_on"] > 0.0


def test_treatment_aware_classifier_and_causal_suite():
    rng = np.random.default_rng(42)
    n_samples = 120
    n_hours = 48
    n_cont = 21
    n_treat = 7

    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2), dtype=int)
    rng.shuffle(labels)

    continuous = rng.normal(scale=0.2, size=(n_samples, n_hours, n_cont)).astype(np.float32)
    masks_cont = np.ones_like(continuous, dtype=np.float32)
    treatments = np.zeros((n_samples, n_hours, n_treat), dtype=np.float32)
    masks_treat = np.ones_like(treatments, dtype=np.float32)

    continuous[labels == 1, :12, 0] += 1.5
    continuous[labels == 0, :12, 0] -= 1.5
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    treated_pos = positive_idx[: int(len(positive_idx) * 0.75)]
    treated_neg = negative_idx[: int(len(negative_idx) * 0.25)]
    treated_idx = np.concatenate([treated_pos, treated_neg])
    treatments[treated_idx, :12, 0] = 1.0
    treatments[treated_idx, :12, 1] = 0.8
    treatments[treated_idx, :6, 3] = 300.0
    untreated_idx = np.setdiff1d(np.arange(n_samples), treated_idx)
    treatments[untreated_idx, :6, 3] = 50.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        s0_dir = root / "s0"
        proc_dir = s0_dir / "processed"
        treatment_dir = root / "treatments"
        proc_dir.mkdir(parents=True)
        treatment_dir.mkdir(parents=True)

        np.save(proc_dir / "continuous.npy", continuous)
        np.save(proc_dir / "masks_continuous.npy", masks_cont)
        np.save(treatment_dir / "treatments.npy", treatments)
        np.save(treatment_dir / "masks_treatments.npy", masks_treat)
        pd.DataFrame(
            {
                "patient_id": np.arange(n_samples).astype(str),
                "age": 65 + labels * 5,
                "sex": np.where(np.arange(n_samples) % 2 == 0, "male", "female"),
                "mortality_inhospital": labels,
                "charlson_comorbidity_index": 1 + labels,
                "first_day_sofa": 2 + labels,
            }
        ).to_csv(s0_dir / "static.csv", index=False)
        pd.DataFrame(
            {
                "patient_id": np.arange(n_samples).astype(str),
                "mortality_inhospital": labels,
            }
        ).to_csv(treatment_dir / "cohort_static.csv", index=False)
        with open(s0_dir / "splits.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": list(range(0, 80)),
                    "val": list(range(80, 100)),
                    "test": list(range(100, 120)),
                },
                f,
            )

        report = train_treatment_aware_classifier(
            s0_dir=s0_dir,
            treatment_dir=treatment_dir,
            output_dir=root / "out",
            batch_size=16,
            epochs=4,
            lr_encoder=1.0e-3,
            lr_head=1.0e-3,
            patience=2,
            device="cpu",
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=64,
            dropout=0.1,
            treatment_layers=1,
            head_hidden_dim=32,
            head_dropout=0.1,
        )
        assert report["splits"]["test"]["auroc"] >= 0.7

        frame = build_causal_frame(
            cohort_static=pd.read_csv(s0_dir / "static.csv"),
            treatments=treatments,
            treatment_names=[
                "vasopressor_on",
                "vasopressor_rate",
                "antibiotic_on",
                "crystalloid_fluid_ml",
                "fluid_bolus_ml",
                "mechanical_vent_on",
                "rrt_on",
            ],
            continuous=continuous,
            masks_continuous=masks_cont,
            outcome_col="mortality_inhospital",
        )
        results = run_causal_suite(
            frame,
            treatment_cols=["vasopressor_on_any_6h"],
            outcome_col="mortality_inhospital",
            covariate_cols=["age", "charlson_comorbidity_index", "first_day_sofa", "map_mean_6h"],
            effect_modifier_cols=["age", "map_mean_6h"],
            rdd_specs={
                "vasopressor_on_any_6h": {
                    "running_col": "map_mean_6h",
                    "threshold": 0.0,
                    "treated_when": "above",
                    "covariate_cols": ["age"],
                }
            },
        )

        assert "vasopressor_on_any_6h" in results["treatments"]
        assert results["treatments"]["vasopressor_on_any_6h"]["psm"]["ate"] is not None


if __name__ == "__main__":
    test_build_eicu_treatment_bundle_extracts_events()
    test_treatment_aware_classifier_and_causal_suite()
    print("2 passed, 0 failed")
